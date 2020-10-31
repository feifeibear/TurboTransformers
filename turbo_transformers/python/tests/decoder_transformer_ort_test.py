# Copyright (C) 2020 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.
import turbo_transformers

import unittest
import sys
import torch
import os

from onmt.decoders.transformer import TransformerDecoderLayer
from onmt_tranformer_copy import TransformerDecoderLayer as ModifiedTransformerDecoderLayer

sys.path.append(os.path.dirname(__file__))
import test_helper


def create_test(batch_size, src_length, T):
    class TestDecoder(unittest.TestCase):
        def init_data(self, use_cuda):
            self.test_device = torch.device('cuda:0') if use_cuda else \
                torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(4)
                turbo_transformers.set_num_threads(4)

            torch.set_grad_enabled(False)
            self.model_dim = 1024

            self.onmt_decoder = ModifiedTransformerDecoderLayer(
                d_model=self.model_dim,
                heads=8,
                d_ff=self.model_dim,
                dropout=0.,
                attention_dropout=0.)
            self.onmt_decoder.eval()
            if use_cuda:
                self.onmt_decoder.to(self.test_device)

            # build ort model
            dummy_T = 10
            dummy_src_len = 20
            dummy_batch = 4
            dummy_input = {
                'input_tensor':
                torch.rand(dummy_batch,
                           dummy_T,
                           self.model_dim,
                           dtype=torch.float32).to(self.test_device),
                'memory_bank':
                torch.rand(dummy_batch,
                           dummy_src_len,
                           self.model_dim,
                           dtype=torch.float32).to(self.test_device),
                'src_pad_mask':
                torch.zeros(dummy_batch, 1, dummy_src_len,
                            dtype=torch.bool).to(self.test_device),
                'tgt_pad_mask':
                torch.zeros(dummy_batch, dummy_T, dummy_T,
                            dtype=torch.bool).to(self.test_device)
            }
            symbolic_names_type_1 = {0: 'batch_size', 1: 'T'}
            symbolic_names_type_2 = {0: 'batch_size', 2: 'src_len'}
            symbolic_names_type_3 = {0: 'batch_size', 1: 'T_1', 2: 'T_2'}
            self.onnx_model_path = "/tmp/temp_turbo_onnx.model"
            with open(self.onnx_model_path, 'wb') as f:
                torch.onnx.export(
                    self.onmt_decoder,
                    (dummy_input['input_tensor'], dummy_input['memory_bank'],
                     dummy_input['src_pad_mask'], dummy_input['tgt_pad_mask']),
                    f,
                    input_names=[
                        'input_tensor', 'memory_bank', 'src_pad_mask',
                        'tgt_pad_mask'
                    ],
                    output_names=['output'],
                    opset_version=11,
                    dynamic_axes={
                        'input_tensor': symbolic_names_type_1,
                        'memory_bank': symbolic_names_type_1,
                        'src_pad_mask': symbolic_names_type_2,
                        'tgt_pad_mask': symbolic_names_type_3
                    })
            import onnxruntime.backend
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = onnxruntime.InferenceSession(
                self.onnx_model_path, sess_options)
            if use_cuda:
                self.session.set_providers(['CUDAExecutionProvider'])

        def check_torch_and_turbo(self, use_cuda, backend="turbo", num_iter=1):
            deivce_type = "GPU" if use_cuda else "CPU"
            info = f"\"({deivce_type}, {batch_size}, {src_length}, {T})\""

            step = 2
            self.init_data(use_cuda=use_cuda)

            self.inputs = torch.rand(batch_size,
                                     T,
                                     self.model_dim,
                                     dtype=torch.float32,
                                     device=self.test_device)
            self.memory_bank = torch.rand(batch_size,
                                          src_length,
                                          self.model_dim,
                                          dtype=torch.float32,
                                          device=self.test_device)

            self.src_pad_mask = torch.zeros(batch_size,
                                            1,
                                            src_length,
                                            dtype=torch.float32,
                                            device=self.test_device).bool()
            self.tgt_pad_mask = torch.zeros(batch_size,
                                            1,
                                            T,
                                            dtype=torch.float32,
                                            device=self.test_device).bool()

            onmt_model = lambda: self.onmt_decoder(self.inputs,
                                                   self.memory_bank,
                                                   self.src_pad_mask,
                                                   self.tgt_pad_mask,
                                                   layer_cache=None,
                                                   step=step,
                                                   future=False)

            onmt_result, torch_qps, torch_time_consume = \
                test_helper.run_model(onmt_model, use_cuda, num_iter)

            print(
                f"ONMT Deocder {info} {backend}",
                f"{deivce_type} QPS, {torch_qps}, time, {torch_time_consume}")

            # run ort model
            ort_inputs = {
                'input_tensor': self.inputs.cpu().numpy(),
                'memory_bank': self.memory_bank.cpu().numpy(),
                'src_pad_mask': self.src_pad_mask.cpu().numpy(),
                'tgt_pad_mask': self.tgt_pad_mask.cpu().numpy()
            }
            # return self.onnx_model.run(inputs=ort_inputs)
            onnxrt_output = self.session.run(None, ort_inputs)
            # onnxrt_output = torch.tensor(onnxrt_output, device=self.test_device)
            # for idx in range(len(onnxrt_output)):
            #     onnxrt_output[idx] = torch.tensor(onnxrt_output[idx],
            #                                     device=self.test_device)
            # turbo_mid = onmt_result[0]
            # turbo_attns = onmt_result[1]

            # TODO(jiaruifang) why FP16 error is so large?
            err = 1e-3
            # self.assertTrue(torch.max(torch.abs(onnxrt_output - onmt_result)) < err)
            # self.assertTrue(torch.max(torch.abs(attns - turbo_attns)) < err)

        def test_decoder(self, backend="onnxrt"):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                    turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True, backend=backend)

    globals()[f"TestDecoder{batch_size}_{src_length}_{T}"] = TestDecoder


create_test(4, src_length=43, T=40)
# #quantize test
# for batch_size in [4]:
#     for src_length in [10, 60, 100]:
#         for T in range(10, src_length, 10):
#             create_test(batch_size, src_length, T, True, "onnxrt")
#FP32 test
# for batch_size in [4]:
#     for src_length in [10, 40, 100]:
#         for T in [10, 40, 100]:
#             create_test(batch_size, src_length, T, False, "turbo")

if __name__ == '__main__':
    unittest.main()

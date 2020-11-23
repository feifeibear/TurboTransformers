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

import unittest

import sys
import torch
import turbo_transformers
from transformers.modeling_albert import AlbertConfig, AlbertModel
import numpy
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


class TestAlbertModel(unittest.TestCase):
    def init_data(self, use_cuda: bool) -> None:
        self.test_device = torch.device('cuda:0') if use_cuda else \
            torch.device('cpu:0')
        if not use_cuda:
            torch.set_num_threads(4)
            turbo_transformers.set_num_threads(4)

        torch.set_grad_enabled(False)
        self.cfg = AlbertConfig()

        self.torch_model = AlbertModel(self.cfg)
        if torch.cuda.is_available():
            self.torch_model.to(self.test_device)
        self.torch_model.eval()
        self.hidden_size = self.cfg.hidden_size
        self.turbo_model = turbo_transformers.AlbertModel.from_torch(
            self.torch_model)

    def check_torch_and_turbo(self,
                              use_cuda,
                              batch_size,
                              seq_length,
                              use_memory_opt=False):
        self.init_data(use_cuda=use_cuda)
        self.input_tensor = torch.randint(low=0,
                                          high=self.cfg.vocab_size - 1,
                                          size=(batch_size, seq_length),
                                          device=self.test_device)

        device = "GPU" if use_cuda else "CPU"
        num_iter = 1
        turbo_model = lambda: self.turbo_model(
            self.input_tensor, attention_mask=None, head_mask=None)
        turbo_result, turbo_qps, turbo_time = \
            test_helper.run_model(turbo_model, use_cuda, num_iter)

        print(
            f"AlbertLayer \"({batch_size},{seq_length:03})\" ",
            f"{device} TurboTransform QPS,  {turbo_qps}, time, {turbo_time}")
        torch_model = lambda: self.torch_model(
            input_ids=self.input_tensor, attention_mask=None, head_mask=None)

        if use_memory_opt:
            turbo_transformers.bert_opt_mem_allocate_api(
                input_ids.size()[0],  # batch
                input_ids.size()[1],  # seq_len
                self.cfg.num_attention_heads,
                self.cfg.hidden_size,
                self.cfg.num_hidden_layers,
                "GPU" if 'cuda' in input_ids.device.type else "CPU")

        with turbo_transformers.pref_guard("albert_perf") as perf:
            torch_result, torch_qps, torch_time = \
                test_helper.run_model(torch_model, use_cuda, num_iter)

        print(f"AlbertModel \"({batch_size},{seq_length:03})\" ",
              f"{device} Torch QPS,  {torch_qps}, time, {torch_time}")

        # print(turbo_result[-1])
        # print(turbo_result, torch_result[0])
        # TODO(jiaruifang) Error is too high. Does tensor core introduce more differences?
        tolerate_error = 1e-2
        self.assertTrue(
            torch.max(torch.abs(torch_result[0] -
                                turbo_result[0])) < tolerate_error)

        with open("albert_model_res.txt", "a") as fh:
            fh.write(
                f"\"({batch_size},{seq_length:03})\", {torch_qps}, {torch_qps}\n"
            )

    def test_albert_model(self, use_memory_opt=False):
        if use_memory_opt:
            turbo_transformers.reset_allocator_schema("model-aware")

        for batch_size in [1, 2]:
            for seq_length in [50, 4, 16]:
                if torch.cuda.is_available() and \
                        turbo_transformers.config.is_compiled_with_cuda():
                    self.check_torch_and_turbo(use_cuda=True,
                                               batch_size=batch_size,
                                               seq_length=seq_length,
                                               use_memory_opt=use_memory_opt)
                self.check_torch_and_turbo(use_cuda=False,
                                           batch_size=batch_size,
                                           seq_length=seq_length,
                                           use_memory_opt=use_memory_opt)

        if use_memory_opt:
            turbo_transformers.reset_allocator_schema("naive")

    def test_bert_model(self):
        # self.test_albert_model(True)
        self.test_albert_model(False)


if __name__ == '__main__':
    unittest.main()

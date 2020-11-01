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

import torch.nn as nn
from transformers.modeling_bert import BertConfig
from bert_model_copy import BertModel as ModifiedBertModel
from bert_model_copy import reset_timer, print_timer
import torch
import math
import time
import turbo_transformers


def profile_hf_batch_reduction(batch_size, seq_length, use_cuda,
                               use_memory_opt):
    test_device = torch.device('cuda:0') if use_cuda else \
        torch.device('cpu:0')
    cfg = BertConfig()
    model = ModifiedBertModel(cfg)
    model.eval()
    model.to(test_device)
    torch.set_grad_enabled(False)
    niter = 10

    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(batch_size, seq_length),
                              dtype=torch.long,
                              device=test_device)

    model(input_ids)

    reset_timer()
    start_time = time.time()
    for i in range(niter):
        model(input_ids)
    bert_time = time.time() - start_time
    softmax_time, layernorm_time = print_timer()
    # print(out)
    if use_memory_opt:
        turbo_transformers.reset_allocator_schema("model-aware")

    if use_memory_opt:
        turbo_transformers.bert_opt_mem_allocate_api(
            input_ids.size()[0],  # batch
            input_ids.size()[1],  # seq_len
            cfg.num_attention_heads,
            cfg.hidden_size,
            cfg.num_hidden_layers,
            "GPU" if 'cuda' in input_ids.device.type else "CPU")

    tt_model = turbo_transformers.BertModel.from_torch(model, test_device,
                                                       "turbo")

    tt_model(input_ids)
    start_time = time.time()
    with turbo_transformers.pref_guard("info") as perf:
        for i in range(niter):
            if use_memory_opt:
                turbo_transformers.bert_opt_mem_allocate_api(
                    input_ids.size()[0],  # batch
                    input_ids.size()[1],  # seq_len
                    cfg.num_attention_heads,
                    cfg.hidden_size,
                    cfg.num_hidden_layers,
                    "GPU" if 'cuda' in input_ids.device.type else "CPU")
            tt_model(input_ids)
    end_time = time.time()
    turbo_elapse = end_time - start_time
    print(
        f"{batch_size} {seq_len} torch: {bert_time} {softmax_time} {softmax_time/bert_time*100}% {layernorm_time} {layernorm_time/bert_time*100}%"
    )
    print(f"turbo: {turbo_elapse}")


if __name__ == '__main__':
    use_gpu = True
    for batch_size in [1, 20]:
        for seq_len in [10, 50, 100, 200, 400]:
            profile_hf_batch_reduction(batch_size, seq_len, True, True)

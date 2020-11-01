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
from transformers.modeling_bert import BertConfig, BertAttention
import torch
import math
import time
import turbo_transformers

BertLayerNorm = torch.nn.LayerNorm

g_softmax_cost = 0.
g_ln_cost = 0.


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        global g_softmax_cost
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        end.record()
        torch.cuda.synchronize()
        elapse = start.elapsed_time(end) / 1e3
        g_softmax_cost += elapse
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,
                   attention_probs) if output_attentions else (context_layer, )
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size,
                                       eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        global g_ln_cost
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        end.record()
        torch.cuda.synchronize()
        elapse = start.elapsed_time(end) / 1e3
        g_ln_cost += elapse

        return hidden_states


def build_turbo_attn(batch_size, seq_length):
    test_device = torch.device('cuda:0')
    cfg = BertConfig()

    input_tensor = torch.rand(size=(batch_size, seq_length, cfg.hidden_size),
                              dtype=torch.float32,
                              device=test_device)

    torch_attention_base = BertAttention(cfg)
    torch_attention_base.to(test_device)
    turbo_attention = turbo_transformers.BertAttention.from_torch(
        torch_attention_base)
    turbo_attention(input_tensor)
    start_time = time.time()
    with turbo_transformers.pref_guard("info") as perf:
        for i in range(10):
            turbo_attention(input_tensor)
    end_time = time.time()
    turbo_elapse = end_time - start_time
    return turbo_elapse


def profile_ln(batch_size=1, seq_length=200):
    cfg = BertConfig()
    torch_bert_out = BertSelfOutput(cfg)
    test_device = torch.device('cuda:0')
    torch_bert_out.eval()
    torch_bert_out.to(test_device)

    intermediate_output = torch.rand(size=(batch_size, seq_length,
                                           cfg.intermediate_size),
                                     dtype=torch.float32,
                                     device=test_device)
    attention_output = torch.rand(size=(batch_size, seq_length,
                                        cfg.hidden_size),
                                  dtype=torch.float32,
                                  device=test_device)

    torch_bert_out(attention_output, attention_output)

    global g_ln_cost
    g_ln_cost = 0.
    start_time = time.time()

    for i in range(10):
        torch_bert_out(attention_output, attention_output)

    end_time = time.time()
    torch_elapse = end_time - start_time

    turbo_elapsed = build_turbo_attn(batch_size, seq_length)
    print(turbo_elapsed, g_ln_cost,
          g_ln_cost / (turbo_elapsed + g_ln_cost) * 100)


def profile_softmax(batch_size=1, seq_length=200):
    cfg = BertConfig()
    torch_attention = BertSelfAttention(cfg)

    hidden_size = cfg.hidden_size
    test_device = torch.device('cuda:0')

    torch_attention.to(test_device)
    input_tensor = torch.rand(size=(batch_size, seq_length, hidden_size),
                              dtype=torch.float32,
                              device=test_device)
    torch_attention_base = BertAttention(cfg)
    torch_attention_base.to(test_device)
    turbo_attention = turbo_transformers.BertAttention.from_torch(
        torch_attention_base)

    torch_attention(input_tensor)
    global g_softmax_cost
    g_softmax_cost = 0.
    start_time = time.time()
    for i in range(10):
        torch_attention(input_tensor)
    end_time = time.time()
    torch_elapse = end_time - start_time

    turbo_attention(input_tensor)
    start_time = time.time()
    with turbo_transformers.pref_guard("info") as perf:
        for i in range(10):
            turbo_attention(input_tensor)
    end_time = time.time()
    turbo_elapse = end_time - start_time
    print(batch_size, seq_length, turbo_elapse, g_softmax_cost,
          g_softmax_cost / (turbo_elapse + g_softmax_cost) * 100)


if __name__ == "__main__":
    profile_ln(1, 10)
    profile_ln(1, 100)
    profile_ln(1, 500)
    profile_ln(20, 10)
    profile_ln(20, 100)
    profile_ln(20, 500)

    # profile_softmax(1, 10)
    # profile_softmax(1, 100)
    # profile_softmax(1, 500)
    # profile_softmax(20, 10)
    # profile_softmax(20, 100)
    # profile_softmax(20, 500)

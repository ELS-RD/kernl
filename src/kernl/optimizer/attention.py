#  Copyright 2022 Lefebvre Sarrut
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from typing import Optional

import torch

from kernl.implementations.attention import attention_forward, attention_reference
from kernl.implementations.attention_skinny import skinny_attention_forward
from kernl.utils.extended_matcher import replace_pattern


def attention_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    # When tensors are shaped for bmm, first dimension is used for both batch and heads. Our kernel supports tensors
    # with 4 dimensions, so we add another dimension of size 1 for heads.

    extend_head = q.dim() == 3
    if extend_head:
        q = q.unsqueeze(dim=1)
        k = k.unsqueeze(dim=1)
        v = v.unsqueeze(dim=1)
        output = output.unsqueeze(dim=1)

    # When there is a large difference between those dimensions, our kernel become inefficient
    # (almost no parallelization), so we use pytorch instead
    if q.size(-2) == 1 and k.size(-2) > 50:
        if (attention_mask is None) and (not is_causal):
            skinny_attention_forward(q, k, v, output, sm_scale, is_causal=is_causal, attention_mask=attention_mask)
        else:
            attention_reference(q, k, v, output, sm_scale, is_causal=is_causal, attention_mask=attention_mask)
    else:
        attention_forward(q, k, v, output, sm_scale, is_causal=is_causal, attention_mask=attention_mask)

    if extend_head:
        output = output.squeeze(dim=1)
    return output


torch.fx.wrap("attention_wrapper")


def fuse_attention_pattern_1(gm: torch.fx.GraphModule, is_causal: bool):
    def pattern(q, k, attention_mask, v):
        transpose_10 = k.transpose(-1, -2)
        matmul_20 = torch.matmul(q, transpose_10)
        truediv_10 = matmul_20 / 8.0
        add_30 = truediv_10 + attention_mask
        softmax_10 = torch.nn.functional.softmax(add_30, dim=-1)
        matmul_21 = torch.matmul(softmax_10, v)
        return matmul_21

    def replace(q, k, attention_mask, v):
        output = torch.empty_like(q)
        output = attention_wrapper(q, k, v, output, 1 / 8.0, is_causal, attention_mask)
        return output

    replace_pattern(gm, pattern, replace)


def fuse_attention_pattern_2(gm: torch.fx.GraphModule, is_causal: bool):
    def pattern(q, k, encoder_decoder_position_bias, v):
        transpose_3 = k.transpose(3, 2)
        matmul = torch.matmul(q, transpose_3)
        add_2 = torch.add(matmul, encoder_decoder_position_bias)
        float_1 = add_2.float()
        softmax = torch.nn.functional.softmax(float_1, dim=-1)
        type_as = softmax.type_as(add_2)
        matmul_1 = torch.matmul(type_as, v)
        return matmul_1

    def replace(q, k, encoder_decoder_position_bias, v):
        output = torch.empty_like(q)
        output = attention_wrapper(q, k, v, output, 1.0, is_causal, encoder_decoder_position_bias)
        return output

    replace_pattern(gm, pattern, replace)


def fuse_attention_pattern_3(gm: torch.fx.GraphModule, is_causal: bool):
    def pattern(q, k, v):
        transpose_46 = k.transpose(1, 2)
        bmm_22 = torch.bmm(q, transpose_46)
        softmax_11 = torch.nn.functional.softmax(bmm_22, dim=-1)
        bmm_23 = torch.bmm(softmax_11, v)

        return bmm_23

    def replace(q, k, v):
        output = torch.empty_like(q)
        output = attention_wrapper(q, k, v, output, 1.0, is_causal, None)
        return output

    replace_pattern(gm, pattern, replace)

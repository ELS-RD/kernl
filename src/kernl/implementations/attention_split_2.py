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
import math
from typing import Optional, Union

import torch
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx

@triton.jit
def _fwd_kernel_split_2(
    heads_count,
    intermediates_count,
    size_m,

    input,
    input_batch_stride,
    input_head_stride,
    input_intermediate_stride,
    input_m_stride,
    input_n_stride,

    maximums,
    maximums_batch_stride,
    maximums_head_stride,
    maximums_intermediate_stride,
    maximums_m_stride,

    sums,
    sums_batch_stride,
    sums_head_stride,
    sums_intermediate_stride,
    sums_m_stride,

    output,
    output_batch_stride,
    output_head_stride,
    output_m_stride,
    output_n_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_DHEAD: tl.constexpr
):
    m_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    current_batch_idx = head_idx // heads_count
    current_head_idx = head_idx % heads_count

    range_offs_m = tl.arange(0, BLOCK_M)
    range_offs_d = tl.arange(0, BLOCK_DHEAD)

    offs_m = m_block_idx * BLOCK_M + range_offs_m

    acc = tl.zeros((BLOCK_M, BLOCK_DHEAD), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n_intermediate_idx in range(0, intermediates_count):
        offs_input = (
                current_batch_idx * input_batch_stride
                + current_head_idx * input_head_stride
                + n_intermediate_idx * input_intermediate_stride
                + (offs_m[:, None] * input_m_stride + range_offs_d[None, :] * input_n_stride)
        )
        ptrs_input = input + offs_input
        numerators = tl.load(ptrs_input, mask=offs_m[:, None] < size_m, other=0.0)

        offs_sums = (
                current_batch_idx * sums_batch_stride
                + current_head_idx * sums_head_stride
                + n_intermediate_idx * sums_intermediate_stride
                + offs_m * sums_m_stride
        )
        ptrs_sums = sums + offs_sums
        d_j = tl.load(ptrs_sums, mask=offs_m < size_m, other=0.0)

        offs_maximums = (
                current_batch_idx * maximums_batch_stride
                + current_head_idx * maximums_head_stride
                + n_intermediate_idx * maximums_intermediate_stride
                + offs_m * maximums_m_stride
        )
        ptrs_maximums = maximums + offs_maximums
        l_j = tl.load(ptrs_maximums, mask=offs_m < size_m, other=0.0)

        l_new = tl.maximum(l_i, l_j)
        alpha = tl.exp(l_i - l_new)
        beta = tl.exp(l_j - l_new)
        d_new = alpha * d_i + beta * d_j

        p_scale = beta / d_new

        acc_scale = d_i / d_new * alpha
        acc *= acc_scale[:, None]

        acc += numerators * p_scale[:, None]

        d_i = d_new
        l_i = l_new

    offs_output = (
            current_batch_idx * output_batch_stride
            + current_head_idx * output_head_stride
            + (offs_m[:, None] * output_m_stride + range_offs_d[None, :] * output_n_stride)
    )
    ptrs_output = output + offs_output
    tl.store(ptrs_output, acc, mask=offs_m[:, None] < size_m)

class AttentionSplit2(torch.autograd.Function):
    @staticmethod
    # @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: FunctionCtx,
        input: torch.Tensor,
        maximums: torch.Tensor,
        sums: torch.Tensor,
        output: torch.Tensor,
    ):
        batch, heads, steps, size_m, dhead = input.size()
        # Attention hardcode
        BLOCK_M = 16
        grid = (triton.cdiv(size_m, BLOCK_M), batch * heads)
        _fwd_kernel_split_2[grid](
            heads,
            steps,
            size_m,
            input,
            *input.stride(),
            maximums,
            *maximums.stride(),
            sums,
            *sums.stride(),
            output,
            *output.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_DHEAD=dhead,
            num_warps=4,
            num_stages=1,
        )
        return output


def attention_split_2_forward(
    input: torch.Tensor,
    maximums: torch.Tensor,
    sums: torch.Tensor,
    output: torch.Tensor,
):
    return AttentionSplit2.apply(input, maximums, sums, output)
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
from typing import Optional, Union

import torch
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx
from torch.cuda.amp import custom_fwd


def attention_split_1_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
    attention_mask: Union[torch.Tensor, None],
) -> torch.Tensor:
    BLOCK_N = 128
    size_n = k.size(2)
    n_divisions = triton.cdiv(size_n, BLOCK_N)

    p = torch.matmul(q.float(), k.transpose(2, 3).float()) * sm_scale

    if attention_mask is not None:
        p += attention_mask
    if is_causal:
        size_m = q.size(2)
        size_n = k.size(2)
        M = torch.tril(torch.ones((size_m, size_n), device="cuda"))
        p = torch.where(M == 0, float("-inf"), p)

    p = p.view(p.size(0), p.size(1), p.size(2), n_divisions, p.size(3) // n_divisions).permute(0, 1, 3, 2, 4)

    maximums, _ = torch.max(p, 4)
    numerators = torch.exp(p - maximums.view(*maximums.size(), 1))
    sums = torch.sum(numerators, 4)

    v = v.view(v.size(0), v.size(1), n_divisions, v.size(2) // n_divisions, v.size(3))
    blocks = torch.matmul(numerators.half(), v)

    return blocks, maximums, sums


@triton.jit
def _fwd_part_1(
    heads,
    size_m,
    size_n,
    Q,
    K,
    V,
    sm_scale,
    attention_mask,
    output,
    maximums,
    sums,
    q_batch_stride,
    q_head_stride,
    q_m_stride,
    q_k_stride,
    k_batch_stride,
    k_head_stride,
    k_n_stride,
    k_k_stride,  # axis named n,k instead of k,n because of the transpose of K matrix
    v_batch_stride,
    v_head_stride,
    v_k_stride,
    v_n_stride,
    sums_batch_stride,
    sums_head_stride,
    sums_step_stride,
    sums_m_stride,
    maximums_batch_stride,
    maximums_head_stride,
    maximums_step_stride,
    maximums_m_stride,
    o_batch_stride,
    o_head_stride,
    o_step_stride,
    o_m_stride,
    o_n_stride,
    attention_mask_batch_stride,
    attention_mask_head_stride,
    attention_mask_m_stride,
    attention_mask_k_stride,
    min_clamp_value,
    NEED_LOAD_MASK_SIZE_N: tl.constexpr,
    NEED_LOAD_MASK_SIZE_M: tl.constexpr,
    MASK_BATCH_SIZE: tl.constexpr,
    MASK_HEAD_SIZE: tl.constexpr,
    MASK_M_SIZE: tl.constexpr,
    MASK_K_SIZE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DHEAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Index of the block on M axis (M axis is the rows of matrix K)
    n_block_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # offsets
    range_offs_m = tl.arange(0, BLOCK_M)  # first block on M dimension
    range_offs_n = tl.arange(0, BLOCK_N)  # first block on N dimension
    range_offs_d = tl.arange(0, BLOCK_DHEAD)  # full head

    offs_m = m_block_idx * BLOCK_M + range_offs_m  # rows offsets on M axis

    current_batch_idx = head_idx // heads
    current_head_idx = head_idx % heads
    # memory offsets matrices on whole Q, K, V matrices
    # Offsets for the current block on matrix Q
    offs_q = (
        current_batch_idx * q_batch_stride
        + current_head_idx * q_head_stride
        + (offs_m[:, None] * q_m_stride + range_offs_d[None, :] * q_k_stride)
    )

    # Offsets for the first block on matrix K
    offs_k = (
        current_batch_idx * k_batch_stride
        + current_head_idx * k_head_stride
        + (range_offs_n[:, None] * k_n_stride + range_offs_d[None, :] * k_k_stride)
    )

    # Offsets for the first block on matrix V
    offs_v = (
        current_batch_idx * v_batch_stride
        + current_head_idx * v_head_stride
        + (range_offs_n[:, None] * v_k_stride + range_offs_d[None, :] * v_n_stride)
    )

    # pointers to blocks in Q, K, V
    ptrs_q = Q + offs_q
    ptrs_k = K + offs_k
    ptrs_v = V + offs_v

    # load q, a block of full rows of matrix q
    # it will stay in SRAM throughout
    if NEED_LOAD_MASK_SIZE_M:
        q = tl.load(ptrs_q, mask=offs_m[:, None] < size_m, eviction_policy="", other=0.0)
    else:
        q = tl.load(ptrs_q, eviction_policy="")

    if HAS_MASK:
        mask_batch_idx = (current_batch_idx,)
        if MASK_BATCH_SIZE == 1:
            mask_batch_idx = 0

        mask_head_idx = current_head_idx
        if MASK_HEAD_SIZE == 1:
            mask_head_idx = 0

        offs_base_mask = mask_batch_idx * attention_mask_batch_stride + mask_head_idx * attention_mask_head_stride

    block_start_index_n = n_block_idx * BLOCK_N
    offs_n = block_start_index_n + range_offs_n
    # We load the current block in K in SRAM
    # We do the first multiplication between the block in Q and the current block in K
    # We finish with the scaling (sqrt(BLOCK_DHEAD) in Vaswani et al. but sm_scale here)
    if NEED_LOAD_MASK_SIZE_N:
        load_mask = offs_n[:, None] < size_n
        k = tl.load(ptrs_k + block_start_index_n * k_n_stride, mask=load_mask, eviction_policy="", other=0.0)
    else:
        k = tl.load(ptrs_k + block_start_index_n * k_n_stride, eviction_policy="")

    qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # required to fix a Triton compiler bug, if not done, there is a precision issue
    if NEED_LOAD_MASK_SIZE_N:
        qk = tl.where(range_offs_n[None, :] < size_n, qk, float("-inf"))
    qk += tl.dot(q, k, trans_b=True)
    qk *= sm_scale
    if IS_CAUSAL:
        qk += tl.where(offs_m[:, None] >= offs_n[None, :], 0, float("-inf"))

    if HAS_MASK:
        # we assume mask has a vector shape
        offs_mask = offs_base_mask + offs_n[None, :] * attention_mask_k_stride
        if MASK_M_SIZE != 1:  # mask has a square shape, we load (BLOCK_M, BLOCK_N) elements
            offs_mask += offs_m[:, None] * attention_mask_m_stride

        if NEED_LOAD_MASK_SIZE_N & MASK_M_SIZE == 1:  # mask has a vector shape need a load mask
            attention_load_mask = offs_n[None, :] < size_n
        if MASK_M_SIZE != 1:  # mask has a matrix shape
            if NEED_LOAD_MASK_SIZE_M & (not NEED_LOAD_MASK_SIZE_N):  # load mask on M axis
                attention_load_mask = offs_m[:, None] < size_m
            elif (not NEED_LOAD_MASK_SIZE_M) & NEED_LOAD_MASK_SIZE_N:  # load mask on N axis
                attention_load_mask = offs_n[None, :] < size_n
            elif NEED_LOAD_MASK_SIZE_M & NEED_LOAD_MASK_SIZE_N:  # load mask on both axis
                attention_load_mask = (offs_n[None, :] < size_n) & (offs_m[:, None] < size_m)

        if NEED_LOAD_MASK_SIZE_M | NEED_LOAD_MASK_SIZE_N:
            m = tl.load(
                attention_mask + offs_mask,
                eviction_policy="",
                mask=attention_load_mask,
                other=float("-inf"),
            )
        else:
            m = tl.load(
                attention_mask + offs_mask,
                eviction_policy="",
            )
        # Avoids NaN
        m = tl.where(m == float("-inf"), min_clamp_value, m)
        qk += m

    l_j = tl.max(qk, 1)
    numerators = tl.exp(qk - l_j[:, None])
    d_j = tl.sum(numerators, 1)

    offs_maximums = (
        current_batch_idx * maximums_batch_stride
        + current_head_idx * maximums_head_stride
        + n_block_idx * maximums_step_stride
        + offs_m * maximums_m_stride
    )
    maximums_ptrs = maximums + offs_maximums
    tl.store(maximums_ptrs, l_j, mask=offs_m < size_m)

    offs_sums = (
        current_batch_idx * sums_batch_stride
        + current_head_idx * sums_head_stride
        + n_block_idx * sums_step_stride
        + offs_m * sums_m_stride
    )
    sums_ptrs = sums + offs_sums
    tl.store(sums_ptrs, d_j, mask=offs_m < size_m)

    if NEED_LOAD_MASK_SIZE_N:
        load_mask = offs_n[:, None] < size_n  # repeated otherwise triton segfault
        v = tl.load(ptrs_v + block_start_index_n * v_k_stride, mask=load_mask, other=0.0, eviction_policy="evict_first")
    else:
        v = tl.load(ptrs_v + block_start_index_n * v_k_stride, eviction_policy="evict_first")

    result = tl.dot(numerators.to(Q.dtype.element_ty), v)

    off_o = (
        current_batch_idx * o_batch_stride
        + current_head_idx * o_head_stride
        + n_block_idx * o_step_stride
        + (offs_m[:, None] * o_m_stride + range_offs_d[None, :] * o_n_stride)
    )

    out_ptrs = output + off_o

    if NEED_LOAD_MASK_SIZE_M:
        out_store_mask = offs_m[:, None] < size_m
        tl.store(out_ptrs, result, mask=out_store_mask)
    else:
        tl.store(out_ptrs, result)


@triton.jit
def _fwd_part_2(
    heads_count,
    intermediates_count,
    size_m,
    inputs,
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
    BLOCK_DHEAD: tl.constexpr,
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
        offs_inputs = (
            current_batch_idx * input_batch_stride
            + current_head_idx * input_head_stride
            + n_intermediate_idx * input_intermediate_stride
            + (offs_m[:, None] * input_m_stride + range_offs_d[None, :] * input_n_stride)
        )
        ptrs_inputs = inputs + offs_inputs
        numerators = tl.load(ptrs_inputs, mask=offs_m[:, None] < size_m, other=0.0)

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


class SkinnyAttention(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        sm_scale: float,
        is_causal: bool,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Computes attention.
        FP32 input and output are not supported.
        https://github.com/openai/triton/issues/674
        Not an issue as the function is annotated with @custom_fwd(cast_inputs=torch.float16) so the input is casted to
        float16 before the function is called.

        @param ctx: context for autograd
        @param q: Query matrix size (batch, heads, size_m, dhead)
        @param k: Key matrix size (batch, heads, size_n, dhead)
        @param v: Value matrix size (batch, heads, size_n, dhead)
        @param output: Output matrix size (batch, heads, size_m, dhead)
        @param sm_scale: Scaling factor applied after operation QxK
        @param is_causal: Autoregressive decoder attention
        @param attention_mask: Attention mask matrix broadcastable to (batch, heads, size_m, size_n)
        @return:
        """
        # Constraints
        # Queries and keys have the same d_k size
        assert q.shape[-1] == k.shape[-1]
        # assert (
        #     q.dtype == k.dtype == v.dtype == output.dtype
        # ), f"All tensors must have the same dtype: {q.dtype}, {k.dtype}, {v.dtype}, {output.dtype}"
        assert q.dtype in [torch.float16, torch.bfloat16], f"Only float16 and bfloat16 are supported, got {q.dtype}"
        batch, heads, size_m, dhead = q.size()
        size_n = k.size(2)

        BLOCK_M = 16
        BLOCK_N = 128
        # load mask is needed if one dim (size_n / size_m) of tensors do not align with block size
        NEED_LOAD_MASK_SIZE_N = size_n % BLOCK_N != 0
        NEED_LOAD_MASK_SIZE_M = size_m % BLOCK_M != 0

        n_divisions = triton.cdiv(size_n, BLOCK_N)
        splitted_qkt = torch.empty(
            q.size(0), q.size(1), n_divisions, q.size(2), q.size(3), dtype=torch.float16, device="cuda"
        )

        grid = (n_divisions, triton.cdiv(size_m, BLOCK_M), batch * heads)

        maximums = torch.zeros(
            (
                batch,
                heads,
                n_divisions,
                size_m,
            ),
            device=q.device,
            dtype=torch.float32,
        )
        sums = torch.zeros(
            (
                batch,
                heads,
                n_divisions,
                size_m,
            ),
            device=q.device,
            dtype=torch.float32,
        )

        HAS_MASK = False
        if attention_mask is not None:
            assert (
                attention_mask.size(0) == batch or attention_mask.size(0) == 1
            ), "Incompatible broadcast batch dimension"
            assert (
                attention_mask.size(1) == heads or attention_mask.size(1) == 1
            ), "Incompatible broadcast heads dimension"
            assert (
                attention_mask.size(2) == size_m or attention_mask.size(2) == 1
            ), "Incompatible broadcast size_m dimension"
            assert attention_mask.size(3) == size_n, "Last size of mask must broadcast on QK^t"

            HAS_MASK = True

        _fwd_part_1[grid](
            heads=heads,
            size_m=size_m,
            size_n=size_n,
            Q=q,
            K=k,
            V=v,
            sm_scale=sm_scale,
            attention_mask=attention_mask,
            output=splitted_qkt,
            maximums=maximums,
            sums=sums,
            q_batch_stride=q.stride(0),
            q_head_stride=q.stride(1),
            q_m_stride=q.stride(2),
            q_k_stride=q.stride(3),
            k_batch_stride=k.stride(0),
            k_head_stride=k.stride(1),
            k_n_stride=k.stride(2),
            k_k_stride=k.stride(3),
            v_batch_stride=v.stride(0),
            v_head_stride=v.stride(1),
            v_k_stride=v.stride(2),
            v_n_stride=v.stride(3),
            sums_batch_stride=sums.stride(0),
            sums_head_stride=sums.stride(1),
            sums_step_stride=sums.stride(2),
            sums_m_stride=sums.stride(3),
            maximums_batch_stride=maximums.stride(0),
            maximums_head_stride=maximums.stride(1),
            maximums_step_stride=maximums.stride(2),
            maximums_m_stride=maximums.stride(3),
            o_batch_stride=splitted_qkt.stride(0),
            o_head_stride=splitted_qkt.stride(1),
            o_step_stride=splitted_qkt.stride(2),
            o_m_stride=splitted_qkt.stride(3),
            o_n_stride=splitted_qkt.stride(4),
            attention_mask_batch_stride=attention_mask.stride(0) if HAS_MASK else 0,
            attention_mask_head_stride=attention_mask.stride(1) if HAS_MASK else 0,
            attention_mask_m_stride=attention_mask.stride(2) if HAS_MASK else 0,
            attention_mask_k_stride=attention_mask.stride(3) if HAS_MASK else 0,
            NEED_LOAD_MASK_SIZE_N=NEED_LOAD_MASK_SIZE_N,
            NEED_LOAD_MASK_SIZE_M=NEED_LOAD_MASK_SIZE_M,
            min_clamp_value=torch.finfo(attention_mask.dtype).min if HAS_MASK else 0,
            MASK_BATCH_SIZE=attention_mask.size(0) if HAS_MASK else 0,
            MASK_HEAD_SIZE=attention_mask.size(1) if HAS_MASK else 0,
            MASK_M_SIZE=attention_mask.size(2) if HAS_MASK else 0,
            MASK_K_SIZE=attention_mask.size(3) if HAS_MASK else 0,
            HAS_MASK=HAS_MASK,
            IS_CAUSAL=is_causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DHEAD=dhead,
            num_warps=1,
            num_stages=8,
        )

        batch, heads, steps, size_m, dhead = splitted_qkt.size()
        # Attention hardcode
        BLOCK_M = 16
        grid_part2 = (triton.cdiv(size_m, BLOCK_M), batch * heads)
        _fwd_part_2[grid_part2](
            heads,
            steps,
            size_m,
            splitted_qkt,
            *splitted_qkt.stride(),
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


def skinny_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    is_causal: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
):
    return SkinnyAttention.apply(q, k, v, output, sm_scale, is_causal, attention_mask)

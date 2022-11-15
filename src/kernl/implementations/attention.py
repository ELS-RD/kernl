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
from torch.cuda.amp import custom_fwd


# CREDITS: Initially inspired by the Triton tutorial


# Similar to https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L213
def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
    attention_mask: Union[torch.Tensor, None],
) -> torch.Tensor:
    """
    Reference implementation for attention
    @param q: Query matrix size (batch, heads, size_m, BLOCK_DHEAD)
    @param k: Key matrix size (batch, heads, size_n, BLOCK_DHEAD)
    @param v: Value matrix size (batch, heads, size_n, BLOCK_DHEAD)
    @param output: Output matrix size (batch, heads, size_m, BLOCK_DHEAD)
    @param sm_scale: Scaling factor applied after operation QxK
    @param is_causal: Whether to apply causal attention
    @param attention_mask: Attention mask broadcastable to (batch, heads, size_m, size_n). Warning the mask
    isn't a binary mask like the one you use normally. This mask is directly added to QxK.
    @return:
    """
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    if attention_mask is not None:
        p += attention_mask
    if is_causal:
        size_m = q.size(2)
        size_n = k.size(2)
        M = torch.tril(torch.ones((size_m, size_n), device="cuda"))
        p = torch.where(M == 0, float("-inf"), p)
    p = torch.nn.functional.softmax(p, dim=-1)
    ref_out = torch.matmul(p, v, out=output)
    return ref_out


@triton.jit
def _fwd_kernel(
    heads,
    size_m,
    size_n,
    size_m_rounded,
    Q,
    K,
    V,
    sm_scale,
    attention_mask,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    output,
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
    o_batch_stride,
    o_head_stride,
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
    """
    Computes attention

    Q•K^T Naming conventions. V multiply not represented here.

                                    N Dimension
                                       size_n
                                   ───────────────
                                  ┌───┬───┬───────┐
                                  │   │   │       │
                                  │   │   │       │
                     K Dimension  │   │   │       │
                                  │   │   │       │
                                  │   │   │       │
                                  │   │   │       │
                    BLOCK_DHEAD   └───┴───┴───────┘
                   ┌────────────┐
               │   │            │
    M Dimension│   ├────────────┤     ┌───┐
      size_m   │   │            │     │   │ BLOCK_M
               │   ├────────────┤     └───┘
               │   │            │    BLOCK_N
               │   │            │
                   └────────────┘

    @param heads: number of heads per batch
    @param size_m: size of M axis
    @param size_n: size of N axis
    @param Q: query matrix size (batch, heads, size_m, BLOCK_DHEAD)
    @param K: key matrix size (batch, heads, size_n, BLOCK_DHEAD)
    @param V: value matrix size (batch, heads, size_n, BLOCK_DHEAD)
    @param sm_scale: scaling factor applied after operation QxK
    @param TMP: temporary variable to fix a compiler bug
    @param output: output matrix size (batch, heads, size_m, BLOCK_DHEAD)
    @param q_batch_stride: matrix q stride for batch dimension
    @param q_head_stride: matrix q stride for head dimension
    @param q_m_stride: matrix q stride for rows, called "M dimension"
    @param q_k_stride: matrix q stride for columns, called "K dimension"
    @param k_batch_stride: matrix k stride for batch dimension
    @param k_head_stride: matrix k stride for head dimension
    @param k_n_stride: matrix k stride for rows, called "N dimension", will be columns after transpose
    @param k_k_stride: matrix k stride for columns, called "K dimension", will be rows after transpose
    @param v_batch_stride: matrix v stride for batch dimension
    @param v_head_stride: matrix v stride for head dimension
    @param v_k_stride: matrix v stride for columns
    @param v_n_stride: matrix v stride for rows
    @param o_batch_stride: output matrix stride for batch dimension
    @param o_head_stride: output matrix stride for head dimension
    @param o_m_stride: output matrix stride for rows
    @param o_n_stride: output matrix stride for columns
    @param attention_mask: attention mask matrix broadcastable to (batch, heads, size_m, size_n)
    @param attention_mask_batch_stride: matrix mask stride for batch dimension
    @param attention_mask_head_stride: matrix mask stride for head dimension
    @param attention_mask_m_stride: matrix mask stride for rows
    @param attention_mask_k_stride: matrix mask stride for columns
    @param NEED_LOAD_MASK_SIZE_N: use boundary check when loading K/V/Attention mask tensors
    @param NEED_LOAD_MASK_SIZE_M: use boundary check when loading/saving from/to Q/Output tensors
    @param MASK_BATCH_SIZE: matrix mask size for batch dimension
    @param MASK_HEAD_SIZE: matrix mask size for head dimension
    @param MASK_M_SIZE: matrix mask size for rows
    @param MASK_K_SIZE: matrix mask size for columns
    @param HAS_MASK: whether the mask is applied
    @param IS_CAUSAL: whether the mask is applied
    @param BLOCK_M: number of rows computed in a single instance for matrix Q
    @param BLOCK_DHEAD: number of columns per head
    @param BLOCK_N:  number of rows computed at each loop in the main loop for matrix K and V
    """
    # Index of the block on M axis (M axis is the rows of matrix K)
    m_block_idx = tl.program_id(0)
    # Global index of the current head
    head_idx = tl.program_id(1)

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

    # Temporary pointer to memory to fix bug in triton compiler
    ptrs_t = TMP + head_idx * size_m_rounded + offs_m

    # initialize pointer to m and d used to compute normalizer for softmax
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # initialize the main loop accumulator, it is the size of a block of full rows, written to the output for the
    # current head
    acc = tl.zeros((BLOCK_M, BLOCK_DHEAD), dtype=tl.float32)

    # load q, a block of full rows of matrix q
    # it will stay in SRAM throughout
    if NEED_LOAD_MASK_SIZE_M:
        q = tl.load(ptrs_q, mask=offs_m[:, None] < size_m, other=0.0)
    else:
        q = tl.load(ptrs_q)

    n_end = size_n
    if IS_CAUSAL:
        n_end = ((m_block_idx + 1) * BLOCK_M,)

    if HAS_MASK:
        mask_batch_idx = (current_batch_idx,)
        if MASK_BATCH_SIZE == 1:
            mask_batch_idx = 0

        mask_head_idx = current_head_idx
        if MASK_HEAD_SIZE == 1:
            mask_head_idx = 0

        offs_base_mask = mask_batch_idx * attention_mask_batch_stride + mask_head_idx * attention_mask_head_stride

    # loop over k, v and update accumulator
    # n_row_start is the row offset on dimension N of the current block
    # It's used for both the N dimension of K and V because they are handled at the same time
    for n_row_start in range(0, n_end, BLOCK_N):
        n_row_start = tl.multiple_of(n_row_start, BLOCK_N)
        n_row_offset = n_row_start + range_offs_n
        # We load the current block in K in SRAM
        # We do the first multiplication between the block in Q and the current block in K
        # We finish with the scaling (sqrt(BLOCK_DHEAD) in Vaswani et al. but sm_scale here)
        if NEED_LOAD_MASK_SIZE_N:
            load_mask = n_row_offset[:, None] < size_n
            k = tl.load(ptrs_k + n_row_start * k_n_stride, mask=load_mask, other=0.0)
        else:
            k = tl.load(ptrs_k + n_row_start * k_n_stride)
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # required to fix a Triton compiler bug, if not done, there is a precision issue
        if NEED_LOAD_MASK_SIZE_N:
            qk = tl.where(range_offs_n[None, :] < size_n, qk, float("-inf"))
        qk += tl.dot(q, k, trans_b=True)
        qk *= sm_scale
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= n_row_offset[None, :], 0, float("-inf"))

        if HAS_MASK:
            # we assume mask has a vector shape
            offs_mask = offs_base_mask + n_row_offset[None, :] * attention_mask_k_stride
            if MASK_M_SIZE != 1:  # mask has a square shape, we load (BLOCK_M, BLOCK_N) elements
                offs_mask += offs_m[:, None] * attention_mask_m_stride

            if NEED_LOAD_MASK_SIZE_N & MASK_M_SIZE == 1:  # mask has a vector shape need a load mask
                attention_load_mask = n_row_offset[None, :] < size_n
            if MASK_M_SIZE != 1:  # mask has a matrix shape
                if NEED_LOAD_MASK_SIZE_M & (not NEED_LOAD_MASK_SIZE_N):  # load mask on M axis
                    attention_load_mask = offs_m[:, None] < size_m
                elif (not NEED_LOAD_MASK_SIZE_M) & NEED_LOAD_MASK_SIZE_N:  # load mask on N axis
                    attention_load_mask = n_row_offset[None, :] < size_n
                elif NEED_LOAD_MASK_SIZE_M & NEED_LOAD_MASK_SIZE_N:  # load mask on both axis
                    attention_load_mask = (n_row_offset[None, :] < size_n) & (offs_m[:, None] < size_m)

            if NEED_LOAD_MASK_SIZE_M | NEED_LOAD_MASK_SIZE_N:
                m = tl.load(
                    attention_mask + offs_mask,
                    eviction_policy="evict_first",
                    mask=attention_load_mask,
                    other=float("-inf"),
                )
            else:
                m = tl.load(
                    attention_mask + offs_mask,
                    eviction_policy="evict_first",
                )
            # Avoids NaN
            m = tl.where(m == float("-inf"), min_clamp_value, m)
            qk += m

        # We compute softmax normalization like in Milakov et al.
        # We renamed m (in the original article) to l to avoid confusions
        # We start with the current block qk
        l_j = tl.max(qk, 1)

        numerators = tl.exp(qk - l_j[:, None])
        d_j = tl.sum(numerators, 1)

        l_new = tl.maximum(l_i, l_j)
        alpha = tl.exp(l_i - l_new)
        beta = tl.exp(l_j - l_new)
        d_new = alpha * d_i + beta * d_j

        # We correct the numerator for the current softmax (*beta) -> exp(l_j - l_new) * exp(qk - mj) = exp(l_new) We
        # divide by the normalization. It's strange to do it this way instead of simply computing the softmax for qk,
        # but since all needed operations are already done for updating m and d, it seems faster
        p_scale = beta / d_new

        qk_softmax = numerators * p_scale[:, None]

        # From here, qk_softmax is correct related to all over previously done block
        # However it is wrong related to the full output row if the qk isn't the last block
        # And at this stage all the previously done qk_softmax blocks are also wrong and needs to be corrected
        # To correct previous blocks we will scale the accumulator
        # d_i / d_new is for correcting denominator
        # alpha is for correcting numerator
        acc_scale = d_i / d_new * alpha

        # This isn't useful in the algorithm, simply to fix a compiler bug
        # BUG: have to store and immediately load
        tl.store(ptrs_t, acc_scale)
        acc_scale = tl.load(ptrs_t)

        # acc scaling
        acc = acc * acc_scale[:, None]

        # We now apply the last operation, the multiplication by a block of matrix V
        if NEED_LOAD_MASK_SIZE_N:
            load_mask = n_row_offset[:, None] < size_n  # repeated otherwise triton segfault
            v = tl.load(ptrs_v + n_row_start * v_k_stride, mask=load_mask, other=0.0)
        else:
            v = tl.load(ptrs_v + n_row_start * v_k_stride)
        qk_softmax = qk_softmax.to(Q.dtype.element_ty)
        acc += tl.dot(qk_softmax, v)

        # We update the normalizer for the next iteration
        d_i = d_new
        l_i = l_new

    off_o = (
        current_batch_idx * o_batch_stride
        + current_head_idx * o_head_stride
        + (offs_m[:, None] * o_m_stride + range_offs_d[None, :] * o_n_stride)
    )

    out_ptrs = output + off_o
    if NEED_LOAD_MASK_SIZE_M:
        out_store_mask = offs_m[:, None] < size_m
        tl.store(out_ptrs, acc, mask=out_store_mask)
    else:
        tl.store(out_ptrs, acc)


class Attention(torch.autograd.Function):
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
        assert (
            q.dtype == k.dtype == v.dtype == output.dtype
        ), f"All tensors must have the same dtype: {q.dtype}, {k.dtype}, {v.dtype}, {output.dtype}"
        assert q.dtype in [torch.float16, torch.bfloat16], f"Only float16 and bfloat16 are supported, got {q.dtype}"
        batch, heads, size_m, dhead = q.size()
        size_n = k.size(2)

        # use fix size, because triton otherwise race condition
        # if not power of 2
        # size_m_pow_2 = triton.next_power_of_2(size_m) if size_m & (size_m - 1) else size_m
        # BLOCK_M = max(min(128, size_m_pow_2), 16)  # minimal size
        # if BLOCK_M == 32:
        #     BLOCK_M = 64  # there is a strange triton segfault with 32
        BLOCK_M = 128
        # load mask is needed if one dim (size_n / size_m) of tensors do not align with block size
        NEED_LOAD_MASK_SIZE_N = size_n % BLOCK_M != 0
        NEED_LOAD_MASK_SIZE_M = size_m % BLOCK_M != 0

        grid = (triton.cdiv(size_m, BLOCK_M), batch * heads)
        # following 2 ops required to fix race condition in Triton compiler
        size_m_rounded = math.ceil(size_m / BLOCK_M) * BLOCK_M
        tmp = torch.empty((batch * heads, size_m_rounded), device=q.device, dtype=torch.float32)

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

        _fwd_kernel[grid](
            heads=heads,
            size_m=size_m,
            size_n=size_n,
            size_m_rounded=size_m_rounded,
            Q=q,
            K=k,
            V=v,
            sm_scale=sm_scale,
            attention_mask=attention_mask,
            TMP=tmp,
            output=output,
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
            o_batch_stride=output.stride(0),
            o_head_stride=output.stride(1),
            o_m_stride=output.stride(2),
            o_n_stride=output.stride(3),
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
            BLOCK_N=BLOCK_M,
            BLOCK_DHEAD=dhead,
            num_warps=4,
            num_stages=1,
        )
        ctx.save_for_backward(q, k, v, output)
        return output


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    is_causal: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
):
    return Attention.apply(q, k, v, output, sm_scale, is_causal, attention_mask)

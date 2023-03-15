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
from typing import List, Optional, Union

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
    @param q: Query matrix size (batch, heads, m_size, BLOCK_DHEAD)
    @param k: Key matrix size (batch, heads, n_size, BLOCK_DHEAD)
    @param v: Value matrix size (batch, heads, n_size, BLOCK_DHEAD)
    @param output: Output matrix size (batch, heads, m_size, BLOCK_DHEAD)
    @param sm_scale: SM (softmax) scaling factor applied on Q•K^T just before the softmax
    @param is_causal: Whether to apply causal attention
    @param attention_mask: Attention mask broadcastable to (batch, heads, m_size, n_size). Warning the mask
    isn't a binary mask like the one you use normally. This mask is directly added to QxK.
    @return:
    """
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    if attention_mask is not None:
        p += attention_mask
    if is_causal:
        m_size = q.size(2)
        n_size = k.size(2)
        M = torch.tril(torch.ones((m_size, n_size), device="cuda"))
        p = torch.where(M == 0, float("-inf"), p)
    p = torch.nn.functional.softmax(p, dim=-1)
    ref_out = torch.matmul(p.to(v.dtype), v, out=output)
    return ref_out


def closest_power_of_2(n: int, min_range: int = 16, max_range: int = 128) -> List[int]:
    """return the closests power of 2 for n, in 16-128 range"""
    n = max(min(n, max_range), min_range)
    min_range = math.floor(math.log2(n - 1))
    max_range = math.ceil(math.log2(n + 1))
    ranges = [2**i for i in range(min_range, max_range + 1)]
    return ranges


def prune(configs, named_args):
    """remove block shapes unlikely to provide optimal speedup"""
    pruned_configs = []
    sizes_m = closest_power_of_2(named_args["m_size"])
    sizes_n = closest_power_of_2(named_args["n_size"])
    is_causal = named_args["IS_CAUSAL"]
    for c in configs:
        if is_causal and c.kwargs["BLOCK_M_SIZE"] != c.kwargs["BLOCK_N_SIZE"]:
            continue
        if c.kwargs["BLOCK_M_SIZE"] in sizes_m and c.kwargs["BLOCK_N_SIZE"] in sizes_n:
            pruned_configs.append(c)

    assert len(pruned_configs) > 0
    return pruned_configs


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M_SIZE": 16, "BLOCK_N_SIZE": 16}, num_stages=1, num_warps=4),
#         triton.Config({"BLOCK_M_SIZE": 16, "BLOCK_N_SIZE": 32}, num_stages=1, num_warps=1),
#         triton.Config({"BLOCK_M_SIZE": 16, "BLOCK_N_SIZE": 64}, num_stages=1, num_warps=1),
#         triton.Config({"BLOCK_M_SIZE": 16, "BLOCK_N_SIZE": 128}, num_stages=1, num_warps=1),
#         triton.Config({"BLOCK_M_SIZE": 32, "BLOCK_N_SIZE": 16}, num_stages=1, num_warps=4),
#         triton.Config({"BLOCK_M_SIZE": 32, "BLOCK_N_SIZE": 32}, num_stages=1, num_warps=2),
#         triton.Config({"BLOCK_M_SIZE": 32, "BLOCK_N_SIZE": 64}, num_stages=1, num_warps=2),
#         triton.Config({"BLOCK_M_SIZE": 32, "BLOCK_N_SIZE": 128}, num_stages=1, num_warps=2),
#         triton.Config({"BLOCK_M_SIZE": 64, "BLOCK_N_SIZE": 16}, num_stages=1, num_warps=4),
#         triton.Config({"BLOCK_M_SIZE": 64, "BLOCK_N_SIZE": 32}, num_stages=1, num_warps=4),
#         triton.Config({"BLOCK_M_SIZE": 64, "BLOCK_N_SIZE": 64}, num_stages=1, num_warps=4),
#         triton.Config({"BLOCK_M_SIZE": 64, "BLOCK_N_SIZE": 128}, num_stages=1, num_warps=4),
#         triton.Config({"BLOCK_M_SIZE": 128, "BLOCK_N_SIZE": 16}, num_stages=1, num_warps=4),
#         triton.Config({"BLOCK_M_SIZE": 128, "BLOCK_N_SIZE": 32}, num_stages=1, num_warps=4),
#         triton.Config({"BLOCK_M_SIZE": 128, "BLOCK_N_SIZE": 64}, num_stages=1, num_warps=4),
#         triton.Config({"BLOCK_M_SIZE": 128, "BLOCK_N_SIZE": 128}, num_stages=1, num_warps=4),
#         triton.Config({"BLOCK_M_SIZE": 128, "BLOCK_N_SIZE": 128}, num_stages=1, num_warps=8),
#         # triton.Config({"BLOCK_M_SIZE": 128, "BLOCK_N_SIZE": 256}, num_stages=1, num_warps=8),
#         # triton.Config({"BLOCK_M_SIZE": 256, "BLOCK_N_SIZE": 128}, num_stages=1, num_warps=8),
#         # triton.Config({"BLOCK_M_SIZE": 256, "BLOCK_N_SIZE": 256}, num_stages=1, num_warps=16),
#     ],
#     prune_configs_by={"early_config_prune": prune, "perf_model": None, "top_k": None},
#     key=["m_size", "n_size", "head_size", "HAS_MASK", "IS_MATRIX_MASK", "IS_CAUSAL"],
# )
# @triton.heuristics(  # order should be the same as in function args, otherwise expect strange bugs
#     {
#         # load mask is needed if one dim (n_size / m_size) of tensors do not align with block size
#         "M_LOAD_MASK_NEEDED": lambda args: args["m_size"] % args["BLOCK_M_SIZE"] != 0,
#         "N_LOAD_MASK_NEEDED": lambda args: args["n_size"] % args["BLOCK_N_SIZE"] != 0,
#     }
# )
@triton.jit
def _fwd_kernel(
    head_size,
    m_size,
    n_size,
    cache_key_m_size,
    cache_key_n_size,
    q_ptr,
    k_ptr,
    v_ptr,
    sm_scale,
    attention_mask_ptr,
    output_ptr,
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
    output_batch_stride,
    output_head_stride,
    output_row_stride,
    output_col_stride,
    attention_mask_batch_stride,
    attention_mask_head_stride,
    attention_mask_m_stride,
    attention_mask_n_stride,
    min_clamp_value,
    attention_mask_batch_size,
    attention_mask_head_size,
    attention_mask_m_size,
    attention_mask_n_size,
    HAS_MASK: tl.constexpr,
    IS_MATRIX_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_DHEAD_SIZE: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr,  # this parameter and below are managed by the autotune and need to be at the end
    BLOCK_N_SIZE: tl.constexpr,
    M_LOAD_MASK_NEEDED: tl.constexpr,
    N_LOAD_MASK_NEEDED: tl.constexpr,
):
    """
    Computes attention

    Q•K^T Naming conventions. V multiply not represented here.

                                    N Dimension
                                       n_size
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
      m_size   │   │            │     │   │ BLOCK_M_SIZE
               │   ├────────────┤     └───┘
               │   │            │    BLOCK_N_SIZE
               │   │            │
                   └────────────┘

    @param head_size: number of heads per batch
    @param m_size: size of M axis
    @param n_size: size of N axis
    @param q_ptr: query matrix size (batch, head_size, m_size, BLOCK_DHEAD)
    @param k_ptr: key matrix size (batch, head_size, n_size, BLOCK_DHEAD)
    @param v_ptr: value matrix size (batch, head_size, n_size, BLOCK_DHEAD)
    @param sm_scale: scaling factor applied after operation QxK
    @param output_ptr: output matrix size (batch, head_size, m_size, BLOCK_DHEAD)
    @param q_batch_stride: matrix Q stride for batch dimension
    @param q_head_stride: matrix Q stride for head dimension
    @param q_m_stride: matrix Q stride for rows, called "M dimension"
    @param q_k_stride: matrix Q stride for columns, called "K dimension"
    @param k_batch_stride: matrix K stride for batch dimension
    @param k_head_stride: matrix K stride for head dimension
    @param k_n_stride: matrix K stride for rows, called "N dimension", will be columns after transpose
    @param k_k_stride: matrix K stride for columns, called "K dimension", will be rows after transpose
    @param v_batch_stride: matrix V stride for batch dimension
    @param v_head_stride: matrix V stride for head dimension
    @param v_k_stride: matrix V stride for columns
    @param v_n_stride: matrix V stride for rows
    @param output_batch_stride: matrix OUTPUT stride for batch dimension
    @param output_head_stride: matrix OUTPUT stride for head dimension
    @param output_row_stride: matrix OUTPUT stride for rows
    @param output_col_stride: matrix OUTPUT stride for columns
    @param attention_mask_ptr: attention mask matrix broadcastable to (batch, head_size, m_size, n_size)
    @param attention_mask_batch_stride: matrix mask stride for batch dimension
    @param attention_mask_head_stride: matrix mask stride for head dimension
    @param attention_mask_m_stride: matrix mask stride for rows
    @param attention_mask_n_stride: matrix mask stride for columns
    @param attention_mask_batch_size: matrix mask size for batch dimension
    @param attention_mask_head_size: matrix mask size for head dimension
    @param attention_mask_m_size: matrix mask size for rows (equal to m_size)
    @param attention_mask_n_size: matrix mask size for columns (equal to n_size)
    @param HAS_MASK: whether the mask is applied
    @param IS_MATRIX_MASK: whether the mask is a vector or a matrix
    @param IS_CAUSAL: whether the mask is applied
    @param BLOCK_DHEAD_SIZE: number of columns per head
    @param BLOCK_M_SIZE: number of rows computed in a single instance for matrix Q
    @param BLOCK_N_SIZE:  number of rows computed at each loop in the main loop for matrix K and V
    @param M_LOAD_MASK_NEEDED: use boundary check when loading/saving from/to Q/Output tensors
    @param N_LOAD_MASK_NEEDED: use boundary check when loading K/V/Attention mask tensors
    """

    # Index of the block on M axis (M axis is the rows of matrix K)
    block_m_idx = tl.program_id(0)

    # Global index of the current head (batch and heads are mixed into one program id)
    head_idx = tl.program_id(1)

    # Index of the current batch
    current_batch_idx = head_idx // head_size
    # Index of the head inside current batch
    current_head_idx = head_idx % head_size

    m_range_offs = tl.arange(0, BLOCK_M_SIZE)  # first block on M dimension
    n_range_offs = tl.arange(0, BLOCK_N_SIZE)  # first block on N dimension
    dhead_range_offs = tl.arange(0, BLOCK_DHEAD_SIZE)  # full head

    m_offs = block_m_idx * BLOCK_M_SIZE + m_range_offs  # rows offsets on M axis

    # memory offsets matrices on whole Q, K, V, output matrices
    # offsets for the current block on matrix Q
    q_offs = (
        current_batch_idx * q_batch_stride
        + current_head_idx * q_head_stride
        + (m_offs[:, None] * q_m_stride + dhead_range_offs[None, :] * q_k_stride)
    )

    # offsets for the first block on matrix K
    k_offs = (
        current_batch_idx * k_batch_stride
        + current_head_idx * k_head_stride
        + (n_range_offs[:, None] * k_n_stride + dhead_range_offs[None, :] * k_k_stride)
    )

    # offsets for the first block on matrix V
    v_offs = (
        current_batch_idx * v_batch_stride
        + current_head_idx * v_head_stride
        + (n_range_offs[:, None] * v_k_stride + dhead_range_offs[None, :] * v_n_stride)
    )

    # offsets for the current block on matrix Output
    output_offs = (
        current_batch_idx * output_batch_stride
        + current_head_idx * output_head_stride
        + (m_offs[:, None] * output_row_stride + dhead_range_offs[None, :] * output_col_stride)
    )

    # pointers to blocks in Q, K, V
    q_ptrs = q_ptr + q_offs
    k_ptrs = k_ptr + k_offs
    v_ptrs = v_ptr + v_offs
    output_ptrs = output_ptr + output_offs

    # initialize pointer to m and d used to compute normalizer for softmax
    l_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)

    # initialize the main loop accumulator, it is the size of a block of full rows, written to the output for the
    # current head
    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32)

    # load q, a block of full rows of matrix q
    # there is a bug on n_size, it is not related to Q tensor but if a load mask is needed and BLOCK_N > n_size,
    # output is wrong.
    if M_LOAD_MASK_NEEDED | N_LOAD_MASK_NEEDED:
        q = tl.load(q_ptrs, mask=m_offs[:, None] < m_size, other=0.0)
    else:
        q = tl.load(q_ptrs)

    block_n_end = n_size
    if IS_CAUSAL:
        # in causal mode, we expect that BLOCK_M_SIZE == BLOCK_N_SIZE
        # autotune will prune shapes not matching this rule
        block_n_end = (block_m_idx + 1) * BLOCK_N_SIZE

    if HAS_MASK:
        attention_mask_batch_idx = (current_batch_idx,)
        if attention_mask_batch_size == 1:
            attention_mask_batch_idx = 0

        attention_mask_head_idx = current_head_idx
        if attention_mask_head_size == 1:
            attention_mask_head_idx = 0

        attention_mask_off = (
            attention_mask_batch_idx * attention_mask_batch_stride
            + attention_mask_head_idx * attention_mask_head_stride
        )

    # loop over k, v and update accumulator
    # block_n_start_idx is the row offset on dimension N of the current block
    # It's used for both the N dimension of K and V because they are handled at the same time
    for block_n_start_idx in range(0, block_n_end, BLOCK_N_SIZE):
        # block_n_start_idx = tl.multiple_of(block_n_start_idx, BLOCK_N_SIZE)
        block_n_offs = block_n_start_idx + n_range_offs
        # We load the current block in K in SRAM
        # We do the first multiplication between the block in Q and the current block in K
        # We finish with the scaling (sqrt(BLOCK_DHEAD) in Vaswani et al. but sm_scale here)
        if N_LOAD_MASK_NEEDED:
            k_ptr_mask = block_n_offs[:, None] < n_size
            k = tl.load(k_ptrs + block_n_start_idx * k_n_stride, mask=k_ptr_mask, other=0.0)
        else:
            k = tl.load(k_ptrs + block_n_start_idx * k_n_stride)
        qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)

        # required to fix a Triton compiler bug, if not done, there is a precision issue
        if N_LOAD_MASK_NEEDED:
            qk = tl.where(n_range_offs[None, :] < n_size, qk, float("-inf"))
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        if IS_CAUSAL:
            qk += tl.where(m_offs[:, None] >= block_n_offs[None, :], 0, float("-inf"))

        if HAS_MASK:
            # we assume mask has a vector shape
            attention_mask_offs = attention_mask_off + block_n_offs * attention_mask_n_stride
            if IS_MATRIX_MASK:  # mask has a matrix shape, we load (BLOCK_M, BLOCK_N) elements
                attention_mask_offs = attention_mask_offs[None, :] + m_offs[:, None] * attention_mask_m_stride

            if N_LOAD_MASK_NEEDED & (not IS_MATRIX_MASK):  # mask has a vector shape + need a load mask
                attention_mask_ptr_mask = block_n_offs < attention_mask_n_size
            if IS_MATRIX_MASK:  # mask has a matrix shape
                if M_LOAD_MASK_NEEDED & (not N_LOAD_MASK_NEEDED):  # load mask on M axis
                    attention_mask_ptr_mask = m_offs[:, None] < attention_mask_m_size
                elif (not M_LOAD_MASK_NEEDED) & N_LOAD_MASK_NEEDED:  # load mask on N axis
                    attention_mask_ptr_mask = block_n_offs[None, :] < attention_mask_n_size
                elif M_LOAD_MASK_NEEDED & N_LOAD_MASK_NEEDED:  # load mask on both axis
                    attention_mask_ptr_mask = (block_n_offs[None, :] < attention_mask_n_size) & (
                        m_offs[:, None] < attention_mask_m_size
                    )

            if (M_LOAD_MASK_NEEDED & IS_MATRIX_MASK) | N_LOAD_MASK_NEEDED:
                attention_mask = tl.load(
                    attention_mask_ptr + attention_mask_offs,
                    eviction_policy="evict_first",
                    mask=attention_mask_ptr_mask,
                    other=float("-inf"),
                )
            else:
                attention_mask = tl.load(
                    attention_mask_ptr + attention_mask_offs,
                    eviction_policy="evict_first",
                )
            # Avoids NaN
            attention_mask = tl.where(attention_mask == float("-inf"), min_clamp_value, attention_mask)
            # if IS_MATRIX_MASK we already added the dimensions, else we need to add one
            if IS_MATRIX_MASK:
                qk += attention_mask
            else:  # related to https://github.com/openai/triton/issues/1273
                qk += attention_mask[None, :]

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

        # acc scaling
        acc = acc * acc_scale[:, None]

        # We now apply the last operation, the multiplication by a block of matrix V
        if N_LOAD_MASK_NEEDED:
            v_ptr_mask = block_n_offs[:, None] < n_size  # repeated otherwise triton segfault
            v = tl.load(v_ptrs + block_n_start_idx * v_k_stride, mask=v_ptr_mask, other=0.0)
        else:
            v = tl.load(v_ptrs + block_n_start_idx * v_k_stride)
        qk_softmax = qk_softmax.to(q_ptr.dtype.element_ty)
        acc += tl.dot(qk_softmax, v)

        # We update the normalizer for the next iteration
        d_i = d_new
        l_i = l_new

    if M_LOAD_MASK_NEEDED:
        output_ptr_mask = m_offs[:, None] < m_size
        tl.store(output_ptrs, acc, mask=output_ptr_mask)
    else:
        tl.store(output_ptrs, acc)


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
        @param q: Query matrix size (batch, head_size, m_size, dhead)
        @param k: Key matrix size (batch, head_size, n_size, dhead)
        @param v: Value matrix size (batch, head_size, n_size, dhead)
        @param output: Output matrix size (batch, head_size, m_size, dhead)
        @param sm_scale: SM (softmax) scaling factor applied on Q•K^T just before the softmax
        @param is_causal: Autoregressive decoder attention
        @param attention_mask: Attention mask matrix broadcastable to (batch, head_size, m_size, n_size)
        @return:
        """
        # Constraints
        # Queries and keys have the same d_k size
        assert q.shape[-1] == k.shape[-1]
        assert (
            q.dtype == k.dtype == v.dtype == output.dtype
        ), f"All tensors must have the same dtype: {q.dtype}, {k.dtype}, {v.dtype}, {output.dtype}"
        assert q.dtype in [torch.float16, torch.bfloat16], f"Only float16 and bfloat16 are supported, got {q.dtype}"
        batch, head_size, m_size, dhead = q.size()
        n_size = k.size(2)

        grid = lambda args: (triton.cdiv(m_size, args["BLOCK_M_SIZE"]), batch * head_size)  # noqa: E731
        # tmp should match m_size rounded to the next multiple of block_m
        # if unknown because of autotune, we put 128 as a safe value

        HAS_MASK = False
        IS_MATRIX_MASK = False
        if attention_mask is not None:
            assert (
                attention_mask.size(0) == batch or attention_mask.size(0) == 1
            ), "Incompatible broadcast batch dimension"
            assert (
                attention_mask.size(1) == head_size or attention_mask.size(1) == 1
            ), "Incompatible broadcast heads dimension"
            assert (
                attention_mask.size(2) == m_size or attention_mask.size(2) == 1
            ), "Incompatible broadcast m_size dimension"
            assert attention_mask.size(3) == n_size, "Last size of mask must broadcast on QK^t"

            HAS_MASK = True
            IS_MATRIX_MASK = attention_mask.size(2) != 1

        _fwd_kernel[grid](  # can't use name args because of the way autotune is implemented :-(
            head_size,  # heads
            m_size,  # m_size
            n_size,  # n_size
            m_size // 32,  # cache_key_m_size
            n_size // 32,  # cache_key_n_size
            q,  # Q
            k,  # K
            v,  # V
            sm_scale,  # sm_scale
            attention_mask,  # attention_mask
            output,  # output
            *q.stride(),  # (batch, heads, m_size, size_k)
            *k.stride(),  # (batch, heads, n_size, size_k)
            *v.stride(),  # (batch, heads, size_k, n_size)
            *output.stride(),  # (batch, heads, m_size, n_size)
            *attention_mask.stride() if HAS_MASK else (0, 0, 0, 0),  # (batch, heads, m_size, size_k)
            torch.finfo(attention_mask.dtype).min if HAS_MASK else 0,  # min_clamp_value
            *attention_mask.size() if HAS_MASK else (0, 0, 0, 0),  # (batch, heads, m_size, size_k)
            HAS_MASK,  # HAS_MASK
            IS_MATRIX_MASK,  # IS_MATRIX_MASK
            is_causal,  # IS_CAUSAL
            dhead,  # BLOCK_DHEAD
            128,  # BLOCK_M_SIZE
            128,  # BLOCK_N_SIZE
            m_size % 128 != 0,  # M_LOAD_MASK_NEEDED
            n_size % 128 != 0,  # N_LOAD_MASK_NEEDED
            num_warps=4 if k.size(3) <= 64 else 8,
            num_stages=2,
        )
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

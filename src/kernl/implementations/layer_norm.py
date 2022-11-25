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
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx
from torch.cuda.amp import custom_fwd
from triton import JITFunction


# CREDITS: Initially inspired by the Triton tutorial and xformers implementation


def pytorch_naive_layernorm(a: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    """
    Naive implementation of layer norm in PyTorch
    -> only used in benchmarks
    """
    mean = a.mean(dim=-1, keepdim=True)
    var = a.var(dim=-1, keepdim=True)
    rstd = 1 / torch.sqrt(var + eps)
    a_hat = (a - mean) * rstd
    out = a_hat * weight + bias
    return out


def pytorch_naive_rmsnorm(a: torch.Tensor, weight: torch.Tensor, eps: float):
    """
    Naive implementation of rmsnorm in PyTorch.
    Basically it's a layernorm without bias and subtraction of mean.
    Implementation follows HF one:
    https://github.com/huggingface/transformers/blob/d92e22d1f28324f513f3080e5c47c071a3916721/src/transformers/models/t5/modeling_t5.py#L239
    Paper: https://arxiv.org/pdf/1910.07467.pdf
    -> only used in benchmarks
    """
    variance = a.to(torch.float32).pow(2).mean(-1, keepdim=True)
    a *= torch.rsqrt(variance + eps)

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        a = a.to(weight.dtype)

    return weight * a


@triton.jit
def layer_norm_xformers(
    Out,
    A,
    Weight,
    Bias,
    Mean,
    Rstd,
    stride_row_out,
    stride_col_out,
    stride_row_a,
    stride_col_a,
    N,
    eps,
    HAS_BIAS: tl.constexpr,  # not used, just to make the signature similar to single pass
    IS_RMSNORM: tl.constexpr,  # not used, just to make the signature similar to single pass
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm forward pass for a single feature.
    Requires that a whole row of X is loaded into shared memory -> won't work for large tensors.
    based on:
    https://github.com/facebookresearch/xformers/blob/main/xformers/triton/k_layer_norm.py
    (arg names modified to match other implementation)
    -> only used in benchmarks
    """

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_ptrs = A + row * stride_row_a + cols * stride_col_a

    x = tl.load(x_ptrs, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    w = tl.load(Weight + cols, mask=mask, other=1.0)
    b = tl.load(Bias + cols, mask=mask, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N
    x_zm = tl.where(mask, x - mean, 0.0)
    tl.store(Mean + row, mean)

    x_var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(x_var + eps)

    # Normalize
    y = x_zm * rstd
    tl.store(Rstd + row, rstd)

    y = y * w + b
    y_ptrs = Out + row * stride_row_out + cols * stride_col_out
    tl.store(y_ptrs, y, mask=mask)


@triton.jit
def _layer_norm_fwd_fused_single_pass(
    Out,
    A,
    Weight,
    Bias,
    Mean,
    Rstd,
    stride_row_out,
    stride_col_out,
    stride_row_a,
    stride_col_a,
    N,
    eps,
    HAS_BIAS: tl.constexpr,
    IS_RMSNORM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layernorm based on Welford's variance computation algorithm.
    https://changyaochen.github.io/welford/
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    :param Out: output tensor
    :param A: input tensor
    :param Weight: weights applied to the normalized input
    :param Bias: bias added to the normalized input
    :param Mean: save mean tensor for backward
    :param Rstd: save standard deviation tensor for backward
    :param stride_row_a: stride of the input tensor
    :param N: number of elements per row in the input tensor
    :param eps: epsilon value to avoid division by zero
    :param HAS_BIAS: whether the bias is provided
    :param IS_RMSNORM: whether the normalization is rmsnorm (False == layernorm)
    :param BLOCK_SIZE: number of threads per block
    :return: None
    """
    # position of elements processed by this program
    _idx = tl.program_id(0)
    a_ptr = A + _idx * stride_row_a
    out_ptr = Out + _idx * stride_row_out
    # compute mean
    mean = 0.0
    var = 0.0
    for start_n_offset in range(0, N, BLOCK_SIZE):
        end_n_offset = min((start_n_offset + BLOCK_SIZE), N)
        nb_block_cols = end_n_offset - start_n_offset
        column_offset = start_n_offset + tl.arange(0, BLOCK_SIZE)
        mask = column_offset < N
        # eviction policy below have little impact now because of new implementation. Kept as is.
        # float32 is used to avoid overflow because of the square operation
        a = tl.load(a_ptr + column_offset * stride_col_a, mask=mask, other=0.0, eviction_policy="evict_last").to(
            tl.float32
        )
        if IS_RMSNORM:
            var += tl.sum(a * a, axis=0)
        else:
            block_mean = tl.sum(a, axis=0) / nb_block_cols
            # mean is 0 or has a mask applied to it, no need to mask delta_mean!
            delta_mean = block_mean - mean
            delta_mean_sqr = delta_mean * delta_mean

            block_delta = tl.sum((a - block_mean) * a, axis=0)
            # mean has a mask
            mean += tl.sum((a - mean) * mask, axis=0) / end_n_offset
            var += block_delta + delta_mean_sqr * (start_n_offset * nb_block_cols) / end_n_offset

    var /= N
    rstd = 1 / tl.sqrt(var + eps)

    # write-back mean/rstd for backward pass
    tl.store(Mean + _idx, mean)
    tl.store(Rstd + _idx, rstd)

    # multiply by weight and add bias
    for off in range(0, N, BLOCK_SIZE):
        column_offset = off + tl.arange(0, BLOCK_SIZE)
        mask = column_offset < N
        weight = tl.load(Weight + column_offset, mask=mask)

        # eviction policy helps to keep weights in cache (reused by other threads)
        a = tl.load(a_ptr + column_offset * stride_col_a, mask=mask, other=0.0, eviction_policy="evict_first").to(
            tl.float32
        )
        a_hat = (a - mean) * rstd
        out = a_hat * weight
        if HAS_BIAS:
            bias = tl.load(Bias + column_offset, mask=mask)
            out = out + bias
        # write-back
        tl.store(out_ptr + column_offset * stride_col_out, out, mask=mask)


@triton.jit
def _layer_norm_fwd_fused_multi_pass(
    Out,
    A,
    Weight,
    Bias,
    Mean,
    Rstd,
    stride_row_out,
    stride_col_out,
    stride_row_a,
    stride_col_a,
    N,
    eps,
    IS_RMSNORM: tl.constexpr,  # not used, just to have the same signature than the single pass
    HAS_BIAS: tl.constexpr,  # not used, just to have the same signature than the single pass
    BLOCK_SIZE: tl.constexpr,
):
    """
    Implementation from triton tutorial:
    https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py
    It requires multiple passes on the data to compute mean and variance, it is slower than the single pass version.
    -> only used in benchmarks
    """
    # position of elements processed by this program
    row = tl.program_id(0)
    Out += row * stride_row_out
    A += row * stride_row_a
    # compute mean
    mean = 0
    _mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + cols * stride_col_a, mask=cols < N, other=0.0, eviction_policy="evict_last").to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # compute variance
    _var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + cols * stride_col_a, mask=cols < N, other=0.0, eviction_policy="evict_last").to(tl.float32)
        a = tl.where(cols < N, a - mean, 0.0)
        _var += a * a
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # write-back mean/rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # multiply by weight and add bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        weight = tl.load(Weight + cols, mask=mask)
        bias = tl.load(Bias + cols, mask=mask)
        a = tl.load(A + cols * stride_col_a, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        a_hat = (a - mean) * rstd
        out = a_hat * weight + bias
        # write-back
        tl.store(Out + cols * stride_col_out, out, mask=mask)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
        implementation: JITFunction,
        use_rms_norm: bool,
    ):
        assert x.dtype == weight.dtype, f"input and weight bias must have the same dtype: {x.dtype}, {weight.dtype}"
        if bias is not None:
            assert x.dtype == bias.dtype, f"input and bias must have the same dtype: {x.dtype}, {bias.dtype}"
        # catch eps being too small if the tensors are fp16
        if x.dtype == torch.float16:
            eps = max(eps, 1.6e-5)
        # allocate output
        out = torch.empty_like(x)
        # reshape input data into 2D tensor
        a_arg = x.reshape(-1, x.shape[-1])
        M, N = a_arg.shape
        # tensors below for backward pass
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        std = torch.empty((M,), dtype=torch.float32, device="cuda")
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        BLOCK_SIZE = max(BLOCK_SIZE, 128)
        if implementation == layer_norm_xformers:
            assert N <= 4096, "LayerNorm: N is too large for xformers implementation"
        BLOCK_SIZE = min(BLOCK_SIZE, 4096)
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        implementation[(M,)](
            Out=out,
            A=a_arg,
            Weight=weight,
            Bias=bias if bias is not None else a_arg,
            Mean=mean,
            Rstd=std,
            stride_row_out=out.stride(0),
            stride_col_out=out.stride(1),
            stride_row_a=a_arg.stride(0),
            stride_col_a=a_arg.stride(1),
            N=N,
            eps=eps,
            HAS_BIAS=bias is not None,
            IS_RMSNORM=use_rms_norm,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x, mean, std, weight)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        return out


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    implementation: JITFunction = _layer_norm_fwd_fused_single_pass,
    use_rms_norm: bool = False,
):
    return LayerNorm.apply(x, weight, bias, eps, implementation, use_rms_norm)

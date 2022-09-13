import torch

import triton
import triton.language as tl
from triton import JITFunction


# CREDITS: Initially inspired by the Triton tutorial


@triton.jit
def _layer_norm_fwd_fused_single_pass(
        Out,
        A,
        Weight,
        Bias,
        Mean, Rstd,
        stride, N, eps,
        BLOCK_SIZE: tl.constexpr,
):
    """
    Based on Welford's variance computation algorithm.
    https://changyaochen.github.io/welford/
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    # position of elements processed by this program
    row = tl.program_id(0)
    Out += row * stride
    A += row * stride
    # compute mean
    mean = 0.0
    var = 0.0
    for start in range(0, N, BLOCK_SIZE):
        end = min((start + BLOCK_SIZE), N)
        nb_block_col = end - start
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(A + cols, mask=mask, other=0., eviction_policy="evict_last").to(tl.float32)

        block_mean = tl.sum(a, axis=0) / nb_block_col
        # mean is 0 or has a mask applied to it, no need to mask delta_mean!
        delta_mean = block_mean - mean
        delta_mean_sqr = delta_mean * delta_mean

        block_delta = tl.sum((a - block_mean) * a, axis=0)
        # mean has a mask!
        mean += tl.sum((a - mean) * mask, axis=0) / end
        var += block_delta + delta_mean_sqr * (start * nb_block_col) / end

    var = var / N
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
        a = tl.load(A + cols, mask=mask, other=0., eviction_policy="evict_first").to(tl.float32)
        a_hat = (a - mean) * rstd
        out = a_hat * weight + bias
        # write-back
        tl.store(Out + cols, out, mask=mask)


@triton.jit
def _layer_norm_fwd_fused_multi_pass(
        Out,
        A,
        Weight,
        Bias,
        Mean, Rstd,
        stride, N, eps,
        BLOCK_SIZE: tl.constexpr,
):
    # position of elements processed by this program
    row = tl.program_id(0)
    Out += row * stride
    A += row * stride
    # compute mean
    mean = 0
    _mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + cols, mask=cols < N, other=0., eviction_policy="evict_last").to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # compute variance
    _var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + cols, mask=cols < N, other=0., eviction_policy="evict_last").to(tl.float32)
        a = tl.where(cols < N, a - mean, 0.)
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
        a = tl.load(A + cols, mask=mask, other=0., eviction_policy="evict_first").to(tl.float32)
        a_hat = (a - mean) * rstd
        out = a_hat * weight + bias
        # write-back
        tl.store(Out + cols, out, mask=mask)


def layer_norm_forward(a: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, implementation: JITFunction = _layer_norm_fwd_fused_single_pass):
    # allocate output
    out = torch.empty_like(a)
    # reshape input data into 2D tensor
    a_arg = a.reshape(-1, a.shape[-1])
    M, N = a_arg.shape
    mean = torch.empty((M,), dtype=torch.float32, device="cuda")
    rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // a.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    eps = min(eps, 1e-6)  # >= 1e-5 may decrease Bert accuracy
    implementation[(M,)](
        out,
        a_arg,
        weight,
        bias,
        mean, rstd,
        a_arg.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out

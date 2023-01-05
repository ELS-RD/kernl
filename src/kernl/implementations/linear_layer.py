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
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx
from torch.cuda.amp import custom_fwd
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

from kernl.implementations import activation_func


# CREDITS: Initially inspired by the Triton tutorial on matrix multiplications


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": 1},
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
                    # split_k not used
                    # for split_k in [2, 4, 8, 16]:
                    #     configs.append(triton.Config(
                    #         {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                    #         num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=5, num_warps=2),
        # good for int8
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=5, num_warps=2),
    ]
    + get_configs_io_bound(),
    key=["CACHE_KEY_M", "CACHE_KEY_N", "CACHE_KEY_K"],
    prune_configs_by={"early_config_prune": early_config_prune, "perf_model": estimate_matmul_time, "top_k": 10},
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def kernel_fma(
    C,  # Pointers to matrices
    ACT_INPUTS,
    A,
    B,
    bias,
    # Matrix dimensions
    M,
    N,
    K,
    CACHE_KEY_M,
    CACHE_KEY_N,
    CACHE_KEY_K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_om,
    stride_on,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # split k not used, not performant with activation, kept because early_config_prune is expecting it
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    BIAS: tl.constexpr,
    SAVE_ACT_INPUTS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):

    """
    Kernel for computing Out = activation(A x W + C)

    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Bias has shape (N,)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)

    'ActInputs' optionally saves the A x W + C intermediate for backward computations

    This kernel will consolidate over K
    """

    assert ACTIVATION in ["tanh", "gelu", "fast_gelu", "relu"], f"{ACTIVATION} is not supported"
    pid = tl.program_id(axis=0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # trick to avoid masking on M and N axis
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A = A + (ram[:, None] * stride_im + rk[None, :] * stride_ik)
    B = B + (rk[:, None] * stride_wk + rbn[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if BIAS:
        bias = tl.load(bias + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)

        A += BLOCK_K * stride_ik
        B += BLOCK_K * stride_wk

    # optional: save the activation inputs
    if SAVE_ACT_INPUTS:
        act_in_ptrs = ACT_INPUTS + ram[:, None] * stride_om + rbn[None, :]
        tl.store(act_in_ptrs, acc)

    # optional: fused activation (while the data is in shared memory)
    if ACTIVATION == "tanh":
        acc = activation_func.tanh(acc)
    if ACTIVATION == "gelu":
        acc = activation_func.gelu(acc)
    if ACTIVATION == "fast_gelu":
        acc = activation_func.fast_gelu(acc)
    if ACTIVATION == "relu":
        acc = activation_func.relu(acc)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # write back result
    C = C + rm[:, None] * stride_om + rn[None, :] * stride_on
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


class LinearLayerFwd(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: str,
        act_inputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute e = activation(x @ weight + bias).
        This wrapper kicks the `kernel_fma` Triton kernel
        :param ctx: context for autograd
        :param x: input tensor
        :param weight: weight matrix
        :param bias: an optional bias tensor
        :param activation: Activation name. Needs to be a Triton kernel.
        :param act_inputs: an optional tensor to save the activation inputs (for backward)
        :return: result tensor
        """
        x_ = x if x.ndim == 2 else x.flatten(0, 1)

        assert x.dtype == weight.dtype, f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"
        if bias is not None:
            assert x.dtype == bias.dtype, f"Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}"
        assert x_.shape[1] == weight.shape[1], f"Incompatible dimensions: {x_.shape} - {weight.shape}"

        assert bias is None or bias.is_contiguous()
        assert bias is None or bias.shape[0] == weight.shape[0], "Incompatible dimensions in between weight and bias"
        assert weight.is_contiguous()

        M, K = x_.shape
        N, K = weight.shape

        outputs = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa

        kernel_fma[grid](
            outputs,
            act_inputs,
            x_,
            weight,  # data ptrs
            bias if bias is not None else x,  # auto skip bias if not present
            M,  # shapes
            N,
            K,
            M // 32,  # key for triton cache (limit number of compilations)
            N // 32,
            K // 32,
            stride_om=outputs.stride(0),  # strides
            stride_on=outputs.stride(1),
            stride_im=x_.stride(0),
            stride_ik=x_.stride(1),
            stride_wn=weight.stride(0),
            stride_wk=weight.stride(1),
            BIAS=bias is not None,  # optional fused bias
            SAVE_ACT_INPUTS=act_inputs is not None,  # optional save activation inputs
            ACTIVATION=activation if not None else x,  # optional fused activation
            GROUP_M=8,  # speed optimization: group the programs
        )

        outputs = outputs if x.ndim == 2 else outputs.reshape(x.shape[0], -1, N)
        ctx.save_for_backward(weight, bias, x)
        return outputs


def linear_layer(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation="",
    act_inputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return LinearLayerFwd.apply(x, weight, bias, activation, act_inputs)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=5, num_warps=2),
        # good for int8
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=5, num_warps=2),
    ]
    + get_configs_io_bound(),
    key=["CACHE_KEY_M", "CACHE_KEY_N", "CACHE_KEY_K"],
    prune_configs_by={"early_config_prune": early_config_prune, "perf_model": estimate_matmul_time, "top_k": 10},
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def kernel_bwd(
    C,  # Pointers to matrices
    ACT_INPUT,
    A,
    B,
    # Matrix dimensions
    M,
    N,
    K,
    CACHE_KEY_M,
    CACHE_KEY_N,
    CACHE_KEY_K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_om,
    stride_on,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # split k not used, not performant with activation, kept because early_config_prune is expecting it
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # trick to avoid masking on M and N axis
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A = A + (ram[:, None] * stride_im + rk[None, :] * stride_ik)
    B = B + (rk[:, None] * stride_wk + rbn[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)

        A += BLOCK_K * stride_ik
        B += BLOCK_K * stride_wk

    # optional: fused activation (while the data is in shared memory)
    if ACTIVATION != "id":
        act_in_ptrs = ACT_INPUT + ram[:, None] * stride_om + rbn[None, :] * stride_on
        act_input = tl.load(act_in_ptrs).to(acc.dtype)
    if ACTIVATION == "tanh":
        acc *= activation_func.tanh_grad(act_input)
    if ACTIVATION == "gelu":
        acc *= activation_func.gelu_grad(act_input)
    if ACTIVATION == "fast_gelu":
        acc *= activation_func.fast_gelu_grad(act_input)
    if ACTIVATION == "relu":
        acc *= activation_func.relu_grad(act_input)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # write back result
    C = C + rm[:, None] * stride_om + rn[None, :] * stride_on
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


def LinearlayerBwd(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    activation: str = "id",
    act_input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute e = activation(grad_output @ weight + bias).
    This wrapper kicks the `kernel_fwd` Triton kernel
    :param grad_output: input tensor
    :param weight: weight matrix
    :param activation: Activation name. Needs to be a Triton kernel.
    :param act_input: an optional tensor to save the activation inputs (for backward)
    :return: result tensor
    """
    assert activation in ["id", "gelu", "gelu_approx", "squared_relu"]

    batch_shape, n = grad_output.shape[:-1], grad_output.shape[-1]
    batch_dim = batch_shape.numel()
    grad_output_reshaped = grad_output.reshape(batch_dim, n)

    if grad_output_reshaped.stride(0) > 1 and grad_output_reshaped.stride(1) > 1:
        grad_output_reshaped = grad_output_reshaped.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()

    assert (
        grad_output.dtype == weight.dtype
    ), f"grad_output and weight must have the same dtype, got {grad_output.dtype} and {weight.dtype}"
    assert (
        grad_output_reshaped.shape[1] == weight.shape[0]
    ), f"Incompatible dimensions: {grad_output_reshaped.shape} - {weight.shape}"
    if activation != "id":
        assert act_input is not None, f"act_input is required for activation {activation}"

    # M, N, K in bwd are different from M, N, K in fwd
    M, K = grad_output_reshaped.shape
    K, N = weight.shape

    grad_input = torch.empty((M, N), device=grad_output.device, dtype=grad_output.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa

    kernel_bwd[grid](
        grad_input,
        act_input,
        grad_output_reshaped,
        weight,  # data ptrs
        M,  # shapes
        N,
        K,
        M // 32,  # key for triton cache (limit number of compilations)
        N // 32,
        K // 32,
        stride_cm=grad_input.stride(0),  # strides
        # stride_cn=grad_input.stride(1),
        stride_am=grad_output_reshaped.stride(0),
        stride_ak=grad_output_reshaped.stride(1),
        stride_bk=weight.stride(0),
        stride_bn=weight.stride(1),
        ACTIVATION=activation,  # optional fused activation
        GROUP_M=8,  # speed optimization: group the programs
    )

    return grad_input.reshape(*batch_shape, grad_input.shape[-1])

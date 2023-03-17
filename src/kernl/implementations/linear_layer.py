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
        "K_LOAD_MASK_NEEDED": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
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
    output_m_stride,
    output_n_stride,
    act_inputs_m_stride,
    act_inputs_n_stride,
    a_m_stride,
    a_k_stride,
    b_n_stride,
    b_k_stride,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # split k not used, not performant with activation, kept because early_config_prune is expecting it
    SPLIT_K: tl.constexpr,
    K_LOAD_MASK_NEEDED: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SHOULD_SAVE_ACT_INPUTS: tl.constexpr,
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
    program_idx = tl.program_id(axis=0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_idx = program_idx // width
    group_size = min(grid_m - group_idx * GROUP_M, GROUP_M)
    block_m_idx = group_idx * GROUP_M + (program_idx % group_size)
    block_n_idx = (program_idx % width) // group_size

    # now compute the block that each program will go through
    # m_offs (resp. n_offs) denotes a range of indices
    # for rows (resp. col) of C
    m_offs_untagged = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs_untagged = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    # trick to avoid masking on M and N axis
    # m_offs_untagged and n_offs_untagged can contains addresses outside matrix boundaries
    # modulo operation is used to wrap around the indices that go beyond the matrix boundaries
    # The value loaded are not ok but at least we are not reading outside the A/B matrices
    # Then, during storing in C a mask is used and the results related to these wrong values is discarded!
    # Regarding max_contiguous and multiple_of, they are used to force the compiler to vectorize loads
    # multiple_of indicates that the first element of rm / rn is a multiple of BLOCK_M / BLOCK_N
    # max_contiguous indicates that the range is a block of BLOCK_M / BLOCK_N contiguous elements
    m_offs = tl.max_contiguous(tl.multiple_of(m_offs_untagged % M, BLOCK_M), BLOCK_M)
    n_offs = tl.max_contiguous(tl.multiple_of(n_offs_untagged % N, BLOCK_N), BLOCK_N)

    k_range_offs = tl.arange(0, BLOCK_K)

    A = A + (m_offs[:, None] * a_m_stride + k_range_offs[None, :] * a_k_stride)
    B = B + (k_range_offs[:, None] * b_k_stride + n_offs[None, :] * b_n_stride)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if HAS_BIAS:
        bias = tl.load(bias + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    for k in range(K, 0, -BLOCK_K):
        if K_LOAD_MASK_NEEDED:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=k_range_offs[None, :] < k, other=0.0)
            b = tl.load(B, mask=k_range_offs[:, None] < k, other=0.0)
        acc += tl.dot(a, b)

        A += BLOCK_K * a_k_stride
        B += BLOCK_K * b_k_stride

    # optional: save the activation inputs
    if SHOULD_SAVE_ACT_INPUTS:
        act_in_ptrs = ACT_INPUTS + m_offs[:, None] * act_inputs_m_stride + n_offs[None, :] * act_inputs_n_stride
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

    # write back result
    C = C + m_offs[:, None] * output_m_stride + n_offs[None, :] * output_n_stride
    c_ptr_mask = (m_offs < M)[:, None] & (n_offs < N)[None, :]
    tl.store(C, acc, mask=c_ptr_mask)


class LinearLayer(torch.autograd.Function):
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
            output_m_stride=outputs.stride(0),  # strides
            output_n_stride=outputs.stride(1),
            act_inputs_m_stride=act_inputs.stride(0) if act_inputs is not None else 0,
            act_inputs_n_stride=act_inputs.stride(1) if act_inputs is not None else 0,
            a_m_stride=x_.stride(0),
            a_k_stride=x_.stride(1),
            b_n_stride=weight.stride(0),
            b_k_stride=weight.stride(1),
            HAS_BIAS=bias is not None,  # optional fused bias
            SHOULD_SAVE_ACT_INPUTS=act_inputs is not None,  # optional save activation inputs
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
    return LinearLayer.apply(x, weight, bias, activation, act_inputs)

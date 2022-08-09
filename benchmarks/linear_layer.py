# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import Optional

import torch
import triton
import triton.language as tl

# CREDITS: Initially inspired by the Triton tutorial on matrix multiplications


# fmt: off
from torch.utils import benchmark


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_stages=5, num_warps=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=5, num_warps=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=3, num_warps=4),
        # requires a GPU with enough shared memory
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def kernel_fma(
        # Pointers to matrices
        OUT, ACT_INPUTS, INPUT, WEIGHT, bias,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
        # by to get the element one row down (A has M rows)
        stride_om, stride_im,
        stride_wn,
        # Meta-parameters
        BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        BIAS: tl.constexpr,
        SAVE_ACT_INPUTS: tl.constexpr,
        ACTIVATION: tl.constexpr,
):
    # fmt: on

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

    # programs are grouped together to improve L2 hit rate
    # the logic is that we'll consolidate over K. If the programs were not grouped,
    # then multiple cols/rows in the result would end up pulling in the same row and lines
    # from the inputs. By grouping the computation we ensure some data reuse, which the hardware
    # covers via the L2 cache
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)  # number of program ids along the M axis
    num_pid_n = tl.cdiv(N, BLOCK_N)  # number of programs ids along the N axis
    num_pid_in_group = GROUP_M * num_pid_n  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_m = group_id * GROUP_M  # row-id of the first program in the group
    GROUP_M = min(
        num_pid_m - first_pid_m, GROUP_M
    )  # if `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    # row-id /col-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M

    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # the memory addresses of elements can follow numpy broadcasting
    input_ptrs = INPUT + rm[:, None] * stride_im
    weight_ptrs = WEIGHT + rn[None, :] * stride_wn

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if BIAS:
        bias = tl.load(bias + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # block level matrix multiplication.
    # We fetch a block memory block from both inputs, matmul and accumulate, then repeat
    mask_rn = rn < N
    mask_rm = rm < M

    for i in range(0, K, BLOCK_K):
        rk = tl.arange(0, BLOCK_K) + i
        a = tl.load(input_ptrs + rk[None, :], mask=((rk[None, :] < K) & mask_rm[:, None]), other=0.0)
        w = tl.load(weight_ptrs + rk[:, None], mask=((rk[:, None] < K) & mask_rn[None, :]), other=0.0)

        acc += tl.dot(a, w)

    # optional: save the activation inputs
    if SAVE_ACT_INPUTS:
        act_in_ptrs = ACT_INPUTS + rm[:, None] * stride_om + rn[None, :]
        tl.store(act_in_ptrs, acc, mask=mask_rm[:, None] & mask_rn[None, :])

    # optional: fused activation (while the data is in shared memory)
    if ACTIVATION:
        acc = ACTIVATION(acc)

    # write back result
    out_ptrs = OUT + rm[:, None] * stride_om + rn[None, :]
    tl.store(out_ptrs, acc, mask=mask_rm[:, None] & mask_rn[None, :])


# Activation needs to be a triton kernel
def fused_matmul(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation=None,
        save_act_inputs: bool = False
):
    """
    Compute e = activation(x @ weight + bias).
    This wrapper kicks the `kernel_fma` Triton kernel
    """

    if not x.is_contiguous():
        print("make it contiguous")
        x = x.contiguous()

    x_ = x if x.ndim == 2 else x.flatten(0, 1)

    assert (
            x_.shape[1] == weight.shape[1]
    ), f"Incompatible dimensions in between inputs and weight, {x_.shape} - {weight.shape}"
    assert bias is None or bias.is_contiguous()
    assert (
            bias is None or bias.shape[0] == weight.shape[0]
    ), "Incompatible dimensions in between weight and bias"
    assert weight.is_contiguous()

    M, K = x_.shape
    N, K = weight.shape

    outputs = torch.empty((M, N), device=x.device, dtype=x.dtype)
    act_inputs = torch.empty_like(outputs) if save_act_inputs else x  # will not be used in that case

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),) # noqa
    BLOCK_K = 32 if K < 1024 else 64

    # fmt: off
    kernel_fma[grid](
        outputs, act_inputs, x_, weight,            # data ptrs
        bias if bias is not None else x,            # auto skip bias if not present
        M, N, K,                                    # shapes
        outputs.stride(0), x_.stride(0),            # strides
        weight.stride(0),
        ACTIVATION=activation,                      # optional fused activation
        BIAS=bias is not None,                      # optional fused bias
        GROUP_M=8,                                  # speed optimization: group the programs
        BLOCK_K=BLOCK_K,
        SAVE_ACT_INPUTS=save_act_inputs
    )
    # fmt: on

    outputs = outputs if x.ndim == 2 else outputs.reshape(x.shape[0], -1, N)

    return outputs, act_inputs if save_act_inputs else None


batch = 8
M = 512
K = 256


print("batch M K", batch, M, K)

torch.manual_seed(0)
a = torch.randn((batch, M, K), device='cuda', dtype=torch.float16, requires_grad=False)
layer_weight = torch.randn((K*4, K), device='cuda', dtype=torch.float16, requires_grad=False)

linear_layer = torch.nn.Linear(K, K*4, bias=False, device="cuda", dtype=torch.float16)
linear_layer.weight.data = layer_weight

print("a", a.shape)
print("linear_layer.weight", linear_layer.weight.shape)
torch_output = linear_layer(a)

triton_output, _ = fused_matmul(x=a, weight=layer_weight, bias=None)

print("Triton output shape:", triton_output.shape)
print("Torch output shape:", torch_output.shape)

assert torch.allclose(torch_output, triton_output)
cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')
nb_repeat = 10
start_event = [torch.cuda.Event(enable_timing=True) for _ in range(nb_repeat)]
end_event = [torch.cuda.Event(enable_timing=True) for _ in range(nb_repeat)]
for _ in range(nb_repeat):
    fused_matmul(x=a, weight=layer_weight, bias=None)

torch.cuda.synchronize()
for start, end in zip(start_event, end_event):
    cache.zero_()
    start.record()
    fused_matmul(x=a, weight=layer_weight, bias=None)
    end.record()
torch.cuda.synchronize()

times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
triton_time = torch.median(times).tolist()
print(f"Triton time: {triton_time:.2f} ms")

for _ in range(nb_repeat):
    linear_layer(a)


torch.cuda.synchronize()
for start, end in zip(start_event, end_event):
    cache.zero_()
    start.record()
    linear_layer(a)
    end.record()
torch.cuda.synchronize()

times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
torch_time = torch.median(times).tolist()
print(f"Torch time: {torch_time:.4f} ms")
print(f"speedup: {torch_time / triton_time:.4f}")

#
# t0 = benchmark.Timer(
#     stmt='fused_matmul(x=a, weight=layer_weight, bias=None); torch.cuda.synchronize()',
#     setup='from __main__ import fused_matmul',
#     globals={'a': a, 'layer_weight': layer_weight})
#
# print(t0.timeit(10))

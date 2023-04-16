# (echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"') | sudo tee -a /etc/modprobe.d/RestrictedProfiling.conf >/dev/null
# sudo update-initramfs -u -k all
# cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
# sudo apt-get install zlib1g-dev
# for reproductible experiments
# sudo nvidia-smi -pm 1 -i 0
# sudo nvidia-smi -i 0 -pl 350  # 400 for A100
# sudo nvidia-smi -i 0 -lgc 1005
from typing import Optional

import torch
import triton
import triton.language as tl
import random

# from kernl.debugger.debugger import triton_debug
from triton.runtime.driver import CudaUtils

torch.manual_seed(123)
random.seed(123)

device = torch.cuda.current_device()
cuda_utils = CudaUtils()
total_sm = cuda_utils.get_device_properties(device)["multiprocessor_count"]
print(f"total SMs: {total_sm}")

# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit()
def swizzle_tile(tile_id,
                 M, N, K,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 GROUP_M: tl.constexpr
                 ):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


@triton.jit()
def linear_tile(tile_id,
                M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr
                ):
    pid_m = tile_id // tl.cdiv(N, BLOCK_N)
    pid_n = tile_id % tl.cdiv(N, BLOCK_N)
    return pid_m, pid_n


# iterate, multiply and accumulate over K axis
@triton.jit()
def mac_loop(A, B, C,
             M, N, K,
             locks,
             stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
             iters_per_tile,
             start_iter, end_iter,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):

    # where are we in the grid
    tile_id = start_iter // iters_per_tile
    # print(f"tile_id: {tile_id.tensor.item()}, start_iter: {start_iter.tensor.item()}, end_iter: {end_iter.tensor.item()}")
    if GROUP_M  > 0:
        pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak) + BLOCK_K * stride_ak * (start_iter % iters_per_tile)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn) + BLOCK_K * stride_bk * (start_iter % iters_per_tile)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for current_iter in range(start_iter, end_iter):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    if end_iter % iters_per_tile == 0:  # last iteration of the tile always happens before its start on another SM
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)  # compute inside the if/else to avoid spilling!
        tl.store(C_, acc)
        if start_iter % iters_per_tile != 0:  # only if tile has been partially processed
            tl.atomic_xchg(locks + tile_id, 1)
    else:
        while tl.atomic_cas(locks + tile_id, 1, 1) != 1:
            pass
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)  # compute inside the if/else to avoid spilling!
        tl.atomic_add(C_, acc)


# @triton_debug
@triton.jit()
def first_wave(
        A, B, C,
        M, N, K,
        locks,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        total_full_tiles_streamk, total_partial_tiles_streamk, iters_per_tile,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr,
        GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    # print(f"pid: {pid.tensor.item()}")
    start_iter = pid * total_full_tiles_streamk + tl.minimum(pid, total_partial_tiles_streamk)
    last_iter = (pid + 1) * total_full_tiles_streamk + tl.minimum(pid + 1, total_partial_tiles_streamk)

    while start_iter < last_iter:
        end_iter = tl.minimum(start_iter + (iters_per_tile - start_iter % iters_per_tile), last_iter)
        mac_loop(A, B, C,
                 M, N, K,
                 locks,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 iters_per_tile,
                 start_iter, end_iter,
                 BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
                 GROUP_M,
                 )

        start_iter = end_iter


# similar to the reference matmul kernel
@triton.jit()
def full_tiles(
        A, B, C,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        total_tiles_streamk,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr,
        GROUP_M: tl.constexpr,
):
    # first wave has done more tiles than there are SMs, we adjust pid
    tile_id = tl.program_id(0) + total_tiles_streamk
    if GROUP_M > 0:
        pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(tl.float16)  # restore C.dtype.element_ty
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(C, acc)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class matmul(torch.autograd.Function):

    _debug = False

    @staticmethod
    def set_debug(debug: bool):
        matmul._debug = debug

    @staticmethod
    def _call(a: torch.Tensor, b: torch.Tensor, total_programs_streamk: int, BLK_M: int, BLK_N: int, BLK_K: int, two_tiles: bool, num_stages: int, num_warps: int):
        device = a.device

        assert a.is_contiguous() and b.is_contiguous(), "non-contiguous inputs are not supported"
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # accumulator types
        ACC_TYPE = tl.float32 if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # compute grid (work to do per SM on the first wave)
        total_blocks_M = triton.cdiv(M, BLK_M)
        total_blocks_N = triton.cdiv(N, BLK_N)
        iters_per_tile = triton.cdiv(K, BLK_K)
        GROUP_M = 8  # 0 to disable swizzling
        total_tiles = total_blocks_M * total_blocks_N

        if total_programs_streamk > 0:  # Stream-K
            # last wave may occupy less than total_programs_streamk SMs
            total_tiles_streamk = total_tiles % total_programs_streamk
            # for two-tile Stream-K + data-parallel from original paper
            if two_tiles and total_tiles - total_tiles_streamk > total_programs_streamk:
                total_tiles_streamk += total_programs_streamk
            # remaining tiles are computed using classical blocking
            total_blocking_tiles = total_tiles - total_tiles_streamk
            total_iters_streamk = total_tiles_streamk * iters_per_tile
            # iterations related to full waves
            total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
            # iterations related to last (partial) wave
            total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk

        else:  # all tiles are computed using classical blocking
            total_blocking_tiles = total_tiles
            total_tiles_streamk = 0
            total_full_tiles_streamk = 0
            total_partial_tiles_streamk = 0
            total_iters_streamk = 0

        if matmul._debug:
            print(f"M,N,K={M},{N},{K} ; BLK_M,N,K={BLK_M},{BLK_N},{BLK_K}")
            print(f"{total_blocks_M=} x {total_blocks_N=} = {total_tiles=}")
            print(f"{total_tiles_streamk=} + {total_blocking_tiles=} = {total_tiles=}")
            print(f"{total_programs_streamk=}")
            print(f"{total_blocking_tiles=}")
            print(f"{iters_per_tile=}")
            print(f"{total_iters_streamk=}")

        # allocates output
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        # allocates locks to sync work accross SMs
        locks = torch.zeros((total_tiles_streamk,), device=device, dtype=torch.int32)
        k1 = first_wave[(total_programs_streamk,)](
            a,
            b,
            c,
            M,
            N,
            K,
            locks,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            total_full_tiles_streamk=total_full_tiles_streamk,
            total_partial_tiles_streamk=total_partial_tiles_streamk,
            iters_per_tile=iters_per_tile,
            BLOCK_M=BLK_M,
            BLOCK_N=BLK_N,
            BLOCK_K=BLK_K,
            ACC_TYPE=ACC_TYPE,
            GROUP_M=GROUP_M,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        if matmul._debug:
            print(f"{k1.n_regs} registers used, {k1.n_spills} spills")
        k2 = full_tiles[(total_blocking_tiles,)](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            total_tiles_streamk=total_tiles_streamk,
            BLOCK_M=BLK_M,
            BLOCK_N=BLK_N,
            BLOCK_K=BLK_K,
            ACC_TYPE=ACC_TYPE,
            GROUP_M=GROUP_M,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        if matmul._debug:
            print(f"{k2.n_regs} registers used, {k2.n_spills} spills")
        return c

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, grid: int, BLK_M=128, BLK_N=128, BLK_K=32, two_tiles=True, num_stages=3, num_warps=4):
        return matmul._call(a=a, b=b, total_programs_streamk=grid, BLK_M=BLK_M, BLK_N=BLK_N, BLK_K=BLK_K, two_tiles=two_tiles, num_warps=num_warps, num_stages=num_stages)


# ---------------------------------------------------------------------------
# Example and Benchmark
# ---------------------------------------------------------------------------


m, n, k = 1536, 1792, 6016  # some problem size to test
A = torch.randn(m, k, device="cuda", dtype=torch.float16)
B = torch.randn(k, n, device="cuda", dtype=torch.float16)

matmul.set_debug(True)
C = matmul.apply(A, B, total_sm, 128, 128, 32, 4, 4)
matmul.set_debug(False)
expected = A @ B

assert torch.allclose(C, expected, atol=1), f"max: {(C - expected).abs().max().item()}\n{C}\n{expected}"

# for debugging, uncomment the following line
# exit(0)

triton_ms = triton.testing.do_bench(lambda: torch.matmul(A, B))
print("PyTorch", triton_ms)

triton_ms = triton.testing.do_bench(lambda: matmul.apply(A, B, total_sm, 128, 128, 32, True, 4, 4))
print(f"hybrid stream-k (grid={total_sm})", triton_ms)

triton_ms = triton.testing.do_bench(lambda: matmul.apply(A, B, total_sm * 2, 128, 128, 32, True, 4, 4))
print(f"hybrid stream-k (grid={total_sm * 2})", triton_ms)

triton_ms = triton.testing.do_bench(lambda: matmul.apply(A, B, 0, 128, 128, 32, True, 4, 4))
print("tile matmul (grid=0)", triton_ms)

# ---------------------------------------------------------------------------
# Log-sampled benchmark
# ---------------------------------------------------------------------------

# tried to reproduce the tests described in the paper
num_samples = 1000  # 32768
step = 256
values = ((torch.logspace(torch.tensor(step).log2(), torch.tensor(8192).log2(), num_samples,
                          base=2) / step).round() * step).unique().tolist()
shapes = [(int(m), int(n), int(k)) for m in values for n in values for k in values]
shapes = random.sample(shapes, num_samples)
assert len(shapes) == num_samples

results = []
for idx, (m, n, k) in enumerate(shapes):
    # print progress bar
    if idx % 10 == 0 and idx > 0:
        speedups = [ratio for *_, ratio in results]
        print(f"{idx}/{num_samples} - average speedup: {sum(speedups) / len(speedups):.3f}")

    A = torch.randn(m, k, device="cuda", dtype=torch.float16)
    B = torch.randn(k, n, device="cuda", dtype=torch.float16)
    output: Optional[torch.Tensor] = None


    def wrapper_matmul(*args, **kwargs):
        global output
        output = matmul.apply(*args, **kwargs)
        return output


    expected = A @ B
    pytorch_ms = triton.testing.do_bench(lambda: A @ B)
    measures = list()
    for two_tiles in [True, False]:
        for sm in [total_sm, total_sm * 2]:
            triton_ms = triton.testing.do_bench(lambda: wrapper_matmul(A, B, sm, 128, 128, 32, two_tiles, 4, 4))
            measures.append(triton_ms)
            max_disc = (output - expected).abs().max().item()
            # large tolerance to accomodate for large K (rounding due to half precision), we just want to catch bugs.
            assert max_disc <= 5., f"pb size: {m}x{n}x{k} - max discrepancy: {max_disc} - sm: {sm}, 2 tiles: {two_tiles}\n{output}\n{expected}"

    results.append((m, n, k, sm, max_disc, measures, pytorch_ms / min(measures)))
    measures.clear()

results.sort(key=lambda x: x[-1], reverse=False)

# ---------------------------------------------------------------------------
# Benchmark export
# ---------------------------------------------------------------------------

import json

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

# 990/1000 - average speedup: 0.965 (A100)
# 990/1000 - average speedup: 1.060 (3090 RTX no while loop)
# 990/1000 - average speedup: 1.053 (3090 RTX with while loop)
# 990/1000 - average speedup: 1.063 (3090 RTX with while loop and 2 tiles disabled / enabled)

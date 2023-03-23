# for reproductible experiments
# sudo nvidia-smi -pm 1 -i 0
# sudo nvidia-smi -i 0 -pl 350  # 400 for A100
# sudo nvidia-smi -i 0 -lgc 1005

import torch
import triton
import triton.language as tl
from triton.compiler import init_cuda_utils

from matmul_ref import matmul_ref

torch.manual_seed(123)

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit()
def _kernel(
        # input and output matrices
        A, B, C,
        # matrix dimensions
        M, N, K,
        # strides
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        # total number of iterations for Stream-K and other variables
        total_sm, total_iters_streamk, total_tiles_streamk, iters_per_tile,
        # block dimensions and accumulator type
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    # First wave: each SM/triton program does more than one tile of work
    if pid < total_sm:
        process_first_wave(
            A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            pid, total_sm, total_iters_streamk, total_tiles_streamk, iters_per_tile,
            BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
        )
    else:
        # After first wave: classic blocking matmul
        process_classic_blocking(
            A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            pid, total_sm, total_tiles_streamk, iters_per_tile,
            BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
        )


@triton.jit()
def process_first_wave(
        A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        pid, total_sm, total_iters_streamk, total_tiles_streamk, iters_per_tile,
        BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
):
    full = total_iters_streamk // total_sm
    remaining = total_iters_streamk % total_sm
    start_iter = pid * full + tl.minimum(pid, remaining)
    end_iter = (pid + 1) * full + tl.minimum(pid + 1, remaining)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for current_iter in range(start_iter, end_iter):  # iterate over K axis, M/N may change during iteration
        tile_id = current_iter // iters_per_tile
        pid_m = tile_id // tl.cdiv(N, BLOCK_N)
        pid_n = tile_id % tl.cdiv(N, BLOCK_N)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        A_ = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak) + BLOCK_K * stride_ak * (
                current_iter % iters_per_tile)
        B_ = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn) + BLOCK_K * stride_bk * (
                current_iter % iters_per_tile)
        a = tl.load(A_)
        b = tl.load(B_)
        acc += tl.dot(a, b)
        if (current_iter + 1) % iters_per_tile == 0:  # (current_iter + 1) check if next iter is for a new tile
            C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
            if current_iter + 1 - iters_per_tile >= start_iter:
                tl.store(C_, acc)
            else:
                tl.atomic_add(C_, acc)
            if end_iter != current_iter:
                acc *= 0.

    # save last tile if there are some iterations leftovers
    if end_iter % iters_per_tile != 0:
        tile_id = tl.cdiv(end_iter, iters_per_tile) - 1
        pid_m = tile_id // tl.cdiv(N, BLOCK_N)
        pid_n = tile_id % tl.cdiv(N, BLOCK_N)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        tl.atomic_add(C_, acc)


@triton.jit()
def process_classic_blocking(
        A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        pid, total_sm, total_tiles_streamk, iters_per_tile,
        BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
):
    pid = pid + (total_tiles_streamk - total_sm)  # first wave has done more tiles than there are SMs, we adjust pid
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A_ = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc_ = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A_)
        b = tl.load(B_)
        acc_ += tl.dot(a, b)
        A_ += BLOCK_K * stride_ak
        B_ += BLOCK_K * stride_bk
    acc_ = acc_.to(tl.float16)  # restore C.dtype.element_ty
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(C_, acc_)


class _matmul(torch.autograd.Function):
    kernel = _kernel

    @staticmethod
    def _call(a: torch.Tensor, b: torch.Tensor, grid_to_use: int, debug: bool, BLK_M: int, BLK_N: int, BLK_K: int):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
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
        total_tiles = total_blocks_M * total_blocks_N
        total_iters = total_tiles * iters_per_tile  # total work to do
        # tiles to be computed using classical blocking approach (data parallel in the paper)
        # if more SMs than tiles, will be 0
        total_blocking_tiles = (total_tiles // grid_to_use) * grid_to_use if grid_to_use > 0 else total_tiles
        if total_tiles >= grid_to_use:
            # for two-tile Stream-K + data-parallel in the paper
            total_blocking_tiles -= grid_to_use
        total_tiles_streamk = total_tiles - total_blocking_tiles
        total_iters_streamk = total_tiles_streamk * iters_per_tile

        total_programs = grid_to_use + (total_tiles - total_tiles_streamk)  # grid

        if debug:
            print(f"m,n,k={M},{N},{K} ; BLK_M,BLK_N,BLK_K={BLK_M},{BLK_N},{BLK_K}")
            print(f"{total_blocks_M=} x {total_blocks_N=} = {total_tiles=}")
            print(f"{total_tiles_streamk=} + {total_blocking_tiles=} = {total_tiles=}")
            print(f"{total_tiles=} * {iters_per_tile=} = {total_iters=}")
            print(f"{total_iters_streamk=}")
            print(f"{total_programs=}")
        # allocates output
        if total_tiles_streamk > 0:
            # atomic add requires zero-initialized output
            c = torch.zeros((M, N), device=device, dtype=a.dtype)
        else:
            c = torch.empty((M, N), device=device, dtype=a.dtype)
        assert c.dtype == torch.float16
        _kernel[(total_programs,)](
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
            total_iters_streamk=total_iters_streamk,
            total_tiles_streamk=total_tiles_streamk,
            iters_per_tile=iters_per_tile,
            total_sm=grid_to_use,
            BLOCK_M=BLK_M,
            BLOCK_N=BLK_N,
            BLOCK_K=BLK_K,
            ACC_TYPE=ACC_TYPE,
            num_warps=16,
        )
        return c

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, grid: int, debug: bool = False, BLK_M=128, BLK_N=128, BLK_K=32):
        return _matmul._call(a=a, b=b, grid_to_use=grid, debug=debug, BLK_M=BLK_M, BLK_N=BLK_N, BLK_K=BLK_K)


matmul = _matmul.apply

# ---------------------------------------------------------------------------
# Example and Benchmark
# ---------------------------------------------------------------------------

device = torch.cuda.current_device()
init_cuda_utils()
total_sm = triton.compiler.cuda_utils.get_device_properties(device)["multiprocessor_count"]
m, n, k = 256, 512, 2048*16
A = torch.randn(m, k, device="cuda", dtype=torch.float16)
B = torch.randn(k, n, device="cuda", dtype=torch.float16)

debug = False
C = matmul(A, B, total_sm, debug, 64, 64, 64)
expected = A @ B
C_ref = matmul_ref(A, B, 8)
# assert torch.allclose(C, expected, atol=5e-1), f"max: {(C - expected).abs().max().item()}\n{C}\n{expected}"
# assert torch.allclose(C_ref, expected, atol=5e-1), f"max: {(C_ref - expected).max().item()}\n{C_ref}\n{expected}"

if not debug:
    ms, *_ = triton.testing.do_bench(lambda: torch.matmul(A, B))
    print("PyTorch", ms)

    ms, *_ = triton.testing.do_bench(lambda: matmul(A, B, total_sm, False, 64, 64, 64))
    print("Triton", ms, total_sm)

    ms, *_ = triton.testing.do_bench(lambda: matmul(A, B, 0, False, 64, 64, 64))
    print("Triton grid=0", ms)

    ms, *_ = triton.testing.do_bench(lambda: matmul_ref(A, B, 1))
    print("Triton ref", ms)

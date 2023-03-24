import random
from typing import Optional

import torch
import triton
import triton.language as tl

from experimental.misc.matmul_ref import matmul_ref
from kernl.debugger.debugger import triton_debug

torch.manual_seed(123)


@triton.jit()
# @triton_debug
def _kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    total_sm,
    total_iters_streamk,
    total_tiles_streamk,
    iters_per_tile,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < total_sm:  # during first wave, each SM does more than one tile of work
        full = total_iters_streamk // total_sm
        remaining = total_iters_streamk % total_sm
        start_iter = pid * full + tl.minimum(pid, remaining)
        end_iter = (pid + 1) * full + tl.minimum(pid + 1, remaining)
        # print("-- stream k pid", pid.tensor.item(), "start", start_iter.tensor.item(), "end", end_iter.tensor.item(), "nb iter", (end_iter.tensor - start_iter.tensor).item())

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
            A_ = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak) + BLOCK_K * stride_ak * (current_iter % iters_per_tile)
            B_ = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn) + BLOCK_K * stride_bk * (current_iter % iters_per_tile)
            a = tl.load(A_)
            b = tl.load(B_)
            acc += tl.dot(a, b)
            if (current_iter + 1) % iters_per_tile == 0:  # (current_iter + 1) check if next iter is for a new tile
                C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
                if current_iter + 1 - iters_per_tile >= start_iter:
                    tl.store(C_, acc)
                else:
                    tl.atomic_add(C_, acc)
                # print("tile_id first", tile_id.tensor.item(), "to save", acc.tensor.sum().item(), "tile iter", (current_iter % iters_per_tile).tensor.item(), "iter", current_iter, "is full" if (current_iter + 1 - iters_per_tile >= start_iter).tensor.item() else "is_partial")
                if end_iter != current_iter:
                    acc *= 0.0

        # save last tile if there are some iterations leftovers
        if end_iter % iters_per_tile != 0:
            tile_id = tl.cdiv(end_iter, iters_per_tile) - 1
            # print("tile_id bis", tile_id.tensor.item(), "to save", acc.tensor.sum().item(), "iter", end_iter.tensor.item())
            pid_m = tile_id // tl.cdiv(N, BLOCK_N)
            pid_n = tile_id % tl.cdiv(N, BLOCK_N)
            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
            tl.atomic_add(C_, acc)
    else:  # classic matmul part
        pid = pid + (total_tiles_streamk - total_sm)  # first wave has done more tiles than there are SMs
        # print("-- classic pid", pid.tensor.item(), "tile id", pid.tensor.item(), "nb iter", (BLOCK_K.tensor / K.tensor).item())
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
        for k in range(0, K, BLOCK_K):
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
        c = torch.zeros((M, N), device=device, dtype=a.dtype)  # mandatory it's made of zeros!
        assert c.dtype == torch.float16
        _kernel[(total_programs, )](
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
            num_stages=3,
            num_warps=4,
        )
        return c

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, grid: Optional[int] = None, debug: bool=False, BLK_M=64, BLK_N=64, BLK_K=64):  # 82 for 3090 RTX, 108 for A100
        if grid is None:
            device = torch.cuda.current_device()
            grid = triton.compiler.cuda_utils.get_device_properties(device)["multiprocessor_count"]
        return _matmul._call(a=a, b=b, grid_to_use=grid, debug=debug, BLK_M=BLK_M, BLK_N=BLK_N, BLK_K=BLK_K)


matmul = _matmul.apply


# Example
# m, n, k, g = 256, 3584, 8192, 8  # works! with warp set to 16 and 64 64 64 for block mnk
# m, n, k, g = 896, 384, 128, 4
# m, n, k, g = 768, 384, 96, 82
# m, n, k, g = 384, 384, 64, 82
# m, n, k, g = 5120, 3840, 160, 82
# m, n, k, g = 1024, 1024, 1024, 82
# m, n, k, g = 512, 512, 1024, 64
# m, n, k, g = 256, 128, 3968, 82  # marche
m, n, k = 256, 512, 2048*16
g = 32
A = torch.randn(m, k, device="cuda", dtype=torch.float16)
B = torch.randn(k, n, device="cuda", dtype=torch.float16)

debug = False
C = matmul(A, B, g, debug, 64, 64, 64)
expected = A @ B
C_ref = matmul_ref(A, B, 1)
C_grid_0 = matmul(A, B, 0, debug, 64, 64, 64)
for i, res in enumerate([C, C_ref, C_grid_0]):
    print("check", i)
    assert torch.allclose(res, expected, atol=1), f"max: {(res - expected).max().item()}\n{res}\n{expected}"

if not debug:
    ms, *_ = triton.testing.do_bench(lambda: torch.matmul(A, B))
    print("PyTorch", ms)

    # for g in range(0, 82):
    ms, *_ = triton.testing.do_bench(lambda: matmul(A, B, g, False, 64, 64, 64))
    print("Triton grid", g, ms)

    ms, *_ = triton.testing.do_bench(lambda: matmul(A, B, 0, False, 64, 64, 64))
    print("Triton grid 0", ms)

    ms, *_ = triton.testing.do_bench(lambda: matmul_ref(A, B, 1))
    print("Triton ref", ms)

# for experiments on 3090 RTX
# sudo nvidia-smi -pm 1 -i 0
# sudo nvidia-smi -i 0 -pl 350  # 400 for A100
# sudo nvidia-smi -i 0 -lgc 1005


# TODO test on A100


# shapes = [(int(m), int(n), int(k)) for m in range(64, 512, 64) for n in range(64, 512, 64) for k in range(7104, 8192, 128)]
# random.shuffle(shapes)
# shapes = shapes[:1000]
# print("nb shapes", len(shapes))
# g = 82
# for m, n, k in shapes:
#     A = torch.randn(m, k, device="cuda", dtype=torch.float16)
#     B = torch.randn(k, n, device="cuda", dtype=torch.float16)
#     # for g in range(1, 82):
#     ms, *_ = triton.testing.do_bench(lambda: matmul(A, B, g, False, 64, 64, 64))
#     ms_baseline, *_ = triton.testing.do_bench(lambda: matmul_ref(A, B, 2))
#
#     if ms < ms_baseline:
#         print(f"problem size: {m} {n} {k}")
#         print(f"Triton {ms_baseline} < {ms} with g={g}")

# 0.9 -> 8.6 : zero -> empty to create C
# 8.6 -> 7.8 : remove the if / else statements


# list of problems working well
# problem size: 256 128 3968
# Triton 0.07168000191450119 < 0.03481600061058998 with g=82
# problem size: 128 512 3584
# Triton 0.0655359998345375 < 0.05119999870657921 with g=82
# problem size: 128 512 3584
# Triton 0.0655359998345375 < 0.05119999870657921 with g=82
# problem size: 256 128 5120
# Triton 0.09113600105047226 < 0.03891199827194214 with g=82
# problem size: 384 128 2816
# Triton 0.053247999399900436 < 0.03686400130391121 with g=82
# problem size: 256 128 3712
# Triton 0.06758400052785873 < 0.03379200026392937 with g=82
# problem size: 512 128 4608
# Triton 0.08191999793052673 < 0.06143999844789505 with g=82
# problem size: 128 384 3328
# Triton 0.06143999844789505 < 0.03993599861860275 with g=82
# problem size: 512 128 4992
# Triton 0.08908800035715103 < 0.06758400052785873 with g=82

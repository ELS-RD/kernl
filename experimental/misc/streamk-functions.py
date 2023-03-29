# sudo apt-get install zlib1g-dev
# for reproductible experiments
# sudo nvidia-smi -pm 1 -i 0
# sudo nvidia-smi -i 0 -pl 350  # 400 for A100
# sudo nvidia-smi -i 0 -lgc 1005

import torch
import triton
import triton.language as tl
import random
from triton.compiler import init_cuda_utils

torch.manual_seed(123)
random.seed(123)

# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

# iterate, multiply and accumulate over K axis
@triton.jit()
def mac_loop(
        A, B, C,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        total_programs_streamk, total_iters_streamk, total_tiles_streamk, iters_per_tile,
        start_iter, end_iter, locks,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr,
):
    # If no work to do, return early
    if end_iter == start_iter:
        return
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    # where are we in the grid
    tile_id = start_iter // iters_per_tile
    pid_m = tile_id // tl.cdiv(N, BLOCK_N)
    pid_n = tile_id % tl.cdiv(N, BLOCK_N)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A_ = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak) + BLOCK_K * stride_ak * (start_iter % iters_per_tile)
    B_ = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn) + BLOCK_K * stride_bk * (start_iter % iters_per_tile)

    for current_iter in range(start_iter, end_iter):
        a = tl.load(A_)
        b = tl.load(B_)
        acc += tl.dot(a, b)
        A_ += BLOCK_K * stride_ak
        B_ += BLOCK_K * stride_bk

    C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    if end_iter - start_iter == iters_per_tile:
        tl.store(C_, acc)
    elif tl.atomic_cas(locks + tile_id, 0, 1) == 0:
            tl.store(C_, acc)
            tl.atomic_xchg(locks + tile_id, 2)
    else:
        while tl.atomic_cas(locks + tile_id, 2, 3) != 3:
            pass
        tl.atomic_add(C_, acc)


@triton.jit()
def first_wave(
        A, B, C,
        M, N, K,
        locks,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        total_programs_streamk, total_iters_streamk, total_tiles_streamk, iters_per_tile,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    full = total_iters_streamk // total_programs_streamk  # iterations related to full wave
    remaining = total_iters_streamk % total_programs_streamk  # iterations related to last (partial) wave
    start_iter_1 = pid * full + tl.minimum(pid, remaining)
    end_iter = (pid + 1) * full + tl.minimum(pid + 1, remaining)
    start_iter_2 = start_iter_1 + (iters_per_tile - start_iter_1 % iters_per_tile)
    start_iter_2 = tl.minimum(start_iter_2, end_iter)
    start_iter_3 = start_iter_2 + (iters_per_tile - start_iter_2 % iters_per_tile)
    start_iter_3 = tl.minimum(start_iter_3, end_iter)

    # finish tile another SM was working on (may be a complete tile)
    mac_loop(
        A, B, C,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        total_programs_streamk, total_iters_streamk, total_tiles_streamk, iters_per_tile,
        start_iter_1, start_iter_2, locks,
        BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
    )

    # do full tile (if there is enough work for that)
    mac_loop(
        A, B, C,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        total_programs_streamk, total_iters_streamk, total_tiles_streamk, iters_per_tile,
        start_iter_2, start_iter_3, locks,
        BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
    )

    # start a new tile (may be incomplete)
    mac_loop(
        A, B, C,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        total_programs_streamk, total_iters_streamk, total_tiles_streamk, iters_per_tile,
        start_iter_3, end_iter, locks,
        BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
    )

# similar to the reference matmul kernel
@triton.jit()
def full_tiles(
        A, B, C,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        total_programs_streamk, total_iters_streamk, total_tiles_streamk, iters_per_tile,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr,
):
    pid = tl.program_id(0) + total_programs_streamk
    pid = pid + (total_tiles_streamk - total_programs_streamk) # first wave has done more tiles than there are SMs, we adjust pid
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

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class _matmul(torch.autograd.Function):

    @staticmethod
    def _call(a: torch.Tensor, b: torch.Tensor, total_programs_streamk: int, debug: bool, BLK_M: int, BLK_N: int, BLK_K: int, num_stages: int, num_warps: int):
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
        total_blocking_tiles = (total_tiles // total_programs_streamk) * total_programs_streamk if total_programs_streamk > 0 else total_tiles
        if total_tiles >= total_programs_streamk:
            # for two-tile Stream-K + data-parallel in the paper
            total_blocking_tiles -= total_programs_streamk
        total_tiles_streamk = total_tiles - total_blocking_tiles
        total_iters_streamk = total_tiles_streamk * iters_per_tile

        total_programs = total_programs_streamk + (total_tiles - total_tiles_streamk)  # grid
        total_programs_classic = total_programs - total_programs_streamk
        if debug:
            print(f"m,n,k={M},{N},{K} ; BLK_M,BLK_N,BLK_K={BLK_M},{BLK_N},{BLK_K}")
            print(f"{total_blocks_M=} x {total_blocks_N=} = {total_tiles=}")
            print(f"{total_tiles_streamk=} + {total_blocking_tiles=} = {total_tiles=}")
            print(f"{total_tiles=} * {iters_per_tile=} = {total_iters=}")
            print(f"{total_iters_streamk=}")
            print(f"{total_programs_streamk=} + {total_programs_classic=} = {total_programs=}")

        # allocates output
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        locks = torch.zeros((total_tiles_streamk,), device=device, dtype=torch.int32)
        assert c.dtype == torch.float16
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
            total_iters_streamk=total_iters_streamk,
            total_tiles_streamk=total_tiles_streamk,
            iters_per_tile=iters_per_tile,
            total_programs_streamk=total_programs_streamk,
            BLOCK_M=BLK_M,
            BLOCK_N=BLK_N,
            BLOCK_K=BLK_K,
            ACC_TYPE=ACC_TYPE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        if debug:
            print(f"{k1.n_regs} registers used")
        # assert k1.n_spills == 0, f"register spilling detected: {k1.n_spills}"
        k2 = full_tiles[(total_programs_classic,)](
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
            total_programs_streamk=total_programs_streamk,
            BLOCK_M=BLK_M,
            BLOCK_N=BLK_N,
            BLOCK_K=BLK_K,
            ACC_TYPE=ACC_TYPE,
            num_stages=3,
            num_warps=4,
        )
        # assert k2.n_spills == 0, f"register spilling detected: {k2.n_spills}"
        return c

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, grid: int, debug: bool = False, BLK_M=128, BLK_N=128, BLK_K=32, num_stages=3, num_warps=4):
        return _matmul._call(a=a, b=b, total_programs_streamk=grid, debug=debug, BLK_M=BLK_M, BLK_N=BLK_N, BLK_K=BLK_K, num_warps=num_warps, num_stages=num_stages)


matmul = _matmul.apply

# ---------------------------------------------------------------------------
# Example and Benchmark
# ---------------------------------------------------------------------------

device = torch.cuda.current_device()
init_cuda_utils()
total_sm = triton.compiler.cuda_utils.get_device_properties(device)["multiprocessor_count"]
print(f"total SMs: {total_sm}")
total_programs_streamk = total_sm  # number of tiles to use in Stream-K first wave
m, n, k = 1536, 1792, 6016  # some problem size to test
A = torch.randn(m, k, device="cuda", dtype=torch.float16)
B = torch.randn(k, n, device="cuda", dtype=torch.float16)

debug = False
C = matmul(A, B, total_programs_streamk, debug, 64, 64, 64)
expected = A @ B

assert torch.allclose(C, expected, atol=1), f"max: {(C - expected).abs().max().item()}\n{C}\n{expected}"

if not debug:
    triton_ms, *_ = triton.testing.do_bench(lambda: torch.matmul(A, B))
    print("PyTorch", triton_ms)

    triton_ms, *_ = triton.testing.do_bench(lambda: matmul(A, B, total_programs_streamk, debug, 128, 128, 32, 3, 4))
    print(f"hybrid stream-k (grid={total_programs_streamk})", triton_ms)

    total_programs_streamk *= 2
    triton_ms, *_ = triton.testing.do_bench(lambda: matmul(A, B, total_programs_streamk, debug, 128, 128, 32, 3, 4))
    print(f"hybrid stream-k (grid={total_programs_streamk})", triton_ms)

    triton_ms, *_ = triton.testing.do_bench(lambda: matmul(A, B, 0, debug))
    print("tile matmul (grid=0)", triton_ms)

# exit(0)
# ---------------------------------------------------------------------------
# Log-sampled benchmark
# ---------------------------------------------------------------------------

# tried to reproduce the tests described in the paper
num_samples = 100  # 32768
values = ((torch.logspace(torch.tensor(128).log2(), torch.tensor(8192).log2(), num_samples,
                          base=2) / 128).round() * 128).unique().tolist()
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
    triton_ms_1sm, *_ = triton.testing.do_bench(lambda: matmul(A, B, total_sm, False, 128, 128, 32, 3, 4))
    triton_2sm_ms, *_ = triton.testing.do_bench(lambda: matmul(A, B, total_sm*2, False, 128, 128, 32, 3, 4))
    pytorch_ms, *_ = triton.testing.do_bench(lambda: A @ B)

    expected = A @ B
    C = matmul(A, B, total_sm, False, 64, 64, 64)
    max_disc = (C - expected).abs().max().item()
    # for very large K, rounding due to half precision requires a large tolerance. We set it to 1.
    assert max_disc <= 1., f"max: {max_disc}\n{C}\n{expected}"

    results.append((m, n, k, max_disc, pytorch_ms, triton_ms_1sm, triton_2sm_ms, triton_ms_1sm < triton_2sm_ms, pytorch_ms / triton_ms_1sm))


results.sort(key=lambda x: x[-1], reverse=False)

# ---------------------------------------------------------------------------
# Benchmark export
# ---------------------------------------------------------------------------

import json
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)


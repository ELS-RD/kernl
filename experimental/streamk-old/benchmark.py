from typing import Optional

import torch
import triton
import random

import json

from triton.runtime import driver

from experimental.streamk.kernel import matmul


torch.manual_seed(123)
random.seed(123)

device = torch.cuda.current_device()
total_sm = driver.utils.get_device_properties(device)["multiprocessor_count"]
print(f"total SMs: {total_sm}")

# TODO restore disable two-tile so can support very little # SMs
# TODO dead lock with:
# m, n, k = 2304, 5632, 3328
# matmul.set_debug(True)
# A = torch.randn(m, k, device="cuda", dtype=torch.float16)
# B = torch.randn(k, n, device="cuda", dtype=torch.float16)
# C = matmul.apply(A, B, 158, 128, 128, 32, False, 4, 4)
# exit(0)

m, n, k = 1536, 1792, 6016  # some problem size to test
A = torch.randn(m, k, device="cuda", dtype=torch.float16)
B = torch.randn(k, n, device="cuda", dtype=torch.float16)

matmul.set_debug(True)
C = matmul.apply(A, B, 158, 128, 128, 32, True, 4, 4)
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
values = ((torch.logspace(torch.tensor(step).log2(), torch.tensor(8192).log2(), num_samples, base=2) / step).round() * step).unique().tolist()
shapes = [(int(m), int(n), int(k)) for m in values for n in values for k in values]
shapes = random.sample(shapes, num_samples)
assert len(shapes) == num_samples
output: Optional[torch.Tensor] = None


def wrapper_matmul(*args, **kwargs):
    global output
    output = matmul.apply(*args, **kwargs)
    return output


results = []
for idx, (m, n, k) in enumerate(shapes):
    # print progress bar
    if idx % 10 == 0 and idx > 0:
        speedups = [r["speedup"] for r in results]
        print(f"{idx}/{num_samples} - average speedup: {sum(speedups) / len(speedups):.3f}")

    A = torch.randn(m, k, device="cuda", dtype=torch.float16)
    B = torch.randn(k, n, device="cuda", dtype=torch.float16)

    expected = A @ B
    pytorch_ms = triton.testing.do_bench(lambda: A @ B)
    measures = list()
    for two_tiles in [True]:  # TODO reenable False when dead lock is fixed
        nb_sm = [total_sm, total_sm * 2]
        total_tile = (m // 128) * (n // 128)
        if total_tile < total_sm * 2:
            nb_sm.append(total_tile)
        nb_sm += random.sample(range(2, total_sm * 2, 2), 10)
        for sm in nb_sm:
            triton_ms = triton.testing.do_bench(lambda: wrapper_matmul(A, B, sm, 128, 128, 32, two_tiles, 4, 4))
            max_disc = (output - expected).abs().max().item()
            # large tolerance to accomodate for large K (rounding due to half precision), we just want to catch bugs.
            assert max_disc <= 5., f"pb size: {m}x{n}x{k} - max discrepancy: {max_disc} - sm: {sm}, 2 tiles: {two_tiles}\n{output}\n{expected}"
            info = {
                "2 tiles": two_tiles,
                "sm": sm,
                "disc": max_disc,
                "triton_ms": triton_ms,
            }
            measures.append(info)
    best_triton_ms = min([m["triton_ms"] for m in measures])
    d = {
        "m": m,
        "n": n,
        "k": k,
        "triton": measures,
        "pytorch_ms": pytorch_ms,
        "speedup": pytorch_ms / best_triton_ms,
    }
    results.append(d)
    measures = list()

results.sort(key=lambda x: x["speedup"], reverse=False)

# ---------------------------------------------------------------------------
# Benchmark export
# ---------------------------------------------------------------------------

with open("./experimental/streamk/results.json", "w") as f:
    json.dump(results, f, indent=4)

# python -m experimental.streamk.benchmark

# 32760/32768 - average speedup: 0.962 (A100)
# 990/1000 - average speedup: 1.060 (3090 RTX no while loop)
# 990/1000 - average speedup: 1.053 (3090 RTX with while loop)
# 990/1000 - average speedup: 1.063 (3090 RTX with while loop and 2 tiles disabled / enabled)

# for profiling:
# (echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"') | sudo tee -a /etc/modprobe.d/RestrictedProfiling.conf >/dev/null
# sudo update-initramfs -u -k all
# cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
# sudo apt-get install zlib1g-dev
# for reproductible experiments
# sudo nvidia-smi -pm 1 -i 0
# sudo nvidia-smi -i 0 -pl 350  # 400 for A100
# sudo nvidia-smi -i 0 -lgc 1005
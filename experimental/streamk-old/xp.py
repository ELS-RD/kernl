import torch
import triton
import random

from triton.runtime import driver

from experimental.streamk.kernel import matmul


torch.manual_seed(123)
random.seed(123)

device = torch.cuda.current_device()
total_sm = driver.utils.get_device_properties(device)["multiprocessor_count"]
print(f"total SMs: {total_sm}")


m, n, k = 1024, 256, 768
A = torch.randn(m, k, device="cuda", dtype=torch.float16)
B = torch.randn(k, n, device="cuda", dtype=torch.float16)


triton_ms = triton.testing.do_bench(lambda: torch.matmul(A, B))
print("PyTorch", triton_ms)

triton_ms = triton.testing.do_bench(lambda: matmul.apply(A, B, total_sm, 128, 128, 32, True, 4, 4))
print(f"hybrid stream-k (grid={total_sm})", triton_ms)

triton_ms = triton.testing.do_bench(lambda: matmul.apply(A, B, total_sm * 2, 128, 128, 32, True, 4, 4))
print(f"hybrid stream-k (grid={total_sm * 2})", triton_ms)

triton_ms = triton.testing.do_bench(lambda: matmul.apply(A, B, 0, 128, 128, 32, True, 4, 4))
print("tile matmul (grid=0)", triton_ms)

for i in range(1, 82):
    triton_ms = triton.testing.do_bench(lambda: matmul.apply(A, B, i, 128, 128, 32, True, 4, 4))
    print(f"hybrid stream-k (grid={i})", triton_ms)

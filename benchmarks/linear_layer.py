import logging
from time import perf_counter_ns
import torch

from benchmarks.linear_layer_utils import fused_matmul, get_mean, CudaGraph

logging.basicConfig(level=logging.INFO)

nb_repeat = 10
batch = 1
M = 768
N = 256
K = 512

print("batch M N K", batch, M, N, K)

torch.manual_seed(0)
a = torch.randn((batch, M, K), device='cuda', dtype=torch.float16, requires_grad=False)
b = torch.randn_like(a)
large_a = torch.randn((batch*8, M*2, K), device='cuda', dtype=torch.float16, requires_grad=False)
large_b = torch.randn_like(large_a)
layer_weight = torch.randn((N, K), device='cuda', dtype=torch.float16, requires_grad=False)

linear_layer = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
linear_layer.weight.data = layer_weight
cg = CudaGraph(weights=layer_weight)

torch_output = linear_layer(a)
triton_output, _ = fused_matmul(x=a, weight=layer_weight, bias=None)

assert torch.allclose(torch_output, triton_output, atol=1e-1)
cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')

timings = list()
end_event = list()
for _ in range(nb_repeat):
    fused_matmul(x=a, weight=layer_weight, bias=None)
torch.cuda.synchronize()

for _ in range(nb_repeat):
    cache.zero_()
    s = perf_counter_ns()
    fused_matmul(x=a, weight=layer_weight, bias=None)
    timings.append((perf_counter_ns() - s) * 1e-6)
torch.cuda.synchronize()

triton_time = get_mean(timings)
print(f"Triton time: {triton_time:.2f} ms")
timings.clear()

for _ in range(nb_repeat):
    linear_layer(a)
torch.cuda.synchronize()

for _ in range(nb_repeat):
    cache.zero_()
    s = perf_counter_ns()
    linear_layer(a)
    timings.append((perf_counter_ns() - s) * 1e-6)
torch.cuda.synchronize()

torch_time = get_mean(timings)
print(f"Torch time: {torch_time:.4f} ms")
print(f"speedup: {torch_time / triton_time:.4f}")
timings.clear()


cg_output_a = cg.run(a)
print(cg_output_a.shape)
print(linear_layer(a).shape)
assert torch.allclose(cg_output_a, linear_layer(a))

cg_output_b = cg.run(b)
assert torch.allclose(cg_output_b, linear_layer(b), atol=1e-1)

torch.cuda.synchronize()

for _ in range(nb_repeat):
    cache.zero_()
    s = perf_counter_ns()
    cg.run(b)
    timings.append((perf_counter_ns() - s) * 1e-6)
torch.cuda.synchronize()

cuda_graph_time = get_mean(timings)
print(f"Triton + cuda graphs time: {cuda_graph_time:.4f} ms")
print(f"speedup: {torch_time / cuda_graph_time:.4f}")
timings.clear()

# output_large_a = cg.run(large_a)
# print(output_large_a.shape)
# print(linear_layer(large_a).shape)
# assert torch.allclose(output_large_a, linear_layer(large_a), atol=1e-1)

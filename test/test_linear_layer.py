import torch
import pytest

from implementations.cuda_graph import CudaGraph
from implementations.linear_layer import linear_layer
from test import set_seed


@pytest.mark.parametrize("size", [128 * i for i in range(2, 10)])
@pytest.mark.parametrize("batch", [8])
@pytest.mark.parametrize("implementation", ["cublas", "triton", "triton_cuda_graph", "pytorch"])
@set_seed()
def test_benchmark(benchmark, size, batch, implementation):
    M = size
    N = size
    K = size

    a = torch.randn((batch, M, K), device='cuda', dtype=torch.float16, requires_grad=False)
    layer_weight = torch.randn((K * 4, K), device='cuda', dtype=torch.float16, requires_grad=False)

    torch_linear_layer = torch.nn.Linear(K, K*4, bias=False, device="cuda", dtype=torch.float16)
    torch_linear_layer.weight.data = layer_weight
    expected = torch_linear_layer(a)

    if implementation == "cublas":
        value = benchmark(torch_linear_layer, a)
    elif implementation == "triton":
        value, _ = benchmark(linear_layer, x=a, weight=layer_weight, bias=None)
    elif implementation == "triton_cuda_graph":
        cg = CudaGraph(weights=layer_weight)
        value = benchmark(cg.run, inputs=a)
    elif implementation == "pytorch":
        value = benchmark(torch_linear_layer, input=a)
    else:
        raise ValueError(f"Unknown implementation {implementation}")
    assert torch.allclose(value, expected, atol=1e-2)

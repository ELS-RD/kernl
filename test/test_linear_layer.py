import torch
import pytest
from implementations.linear_layer import linear_layer

@pytest.mark.parametrize("size", [128 * i for i in range(2, 10)])
@pytest.mark.parametrize("batch", [8])
@pytest.mark.parametrize("implementation", ["cublas", "triton"])
def test_benchmark(benchmark, size, batch, implementation):
    torch.manual_seed(0)

    M = size
    N = size
    K = size

    a = torch.randn((batch, M, K), device='cuda', dtype=torch.float16, requires_grad=False)
    layer_weight = torch.randn((K * 4, K), device='cuda', dtype=torch.float16, requires_grad=False)

    torch_linear_layer = torch.nn.Linear(K, K*4, bias=False, device="cuda", dtype=torch.float16)
    torch_linear_layer.weight.data = layer_weight

    if implementation == "cublas":
        benchmark(torch_linear_layer, a)
    if implementation == "triton":
        benchmark(linear_layer, x=a, weight=layer_weight, bias=None)

    value, _ = linear_layer(x=a, weight=layer_weight, bias=None)
    expected = torch_linear_layer(a)
    assert torch.allclose(value, expected)
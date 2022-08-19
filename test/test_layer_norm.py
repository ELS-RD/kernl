import torch
import pytest

from implementations.layer_norm import layer_norm_forward


@pytest.mark.parametrize("size", [128])
@pytest.mark.parametrize("implementation", ["cublas", "triton"])
def test_benchmark(benchmark, size, implementation):
    torch.manual_seed(0)

    M = size
    N = size
    dtype = torch.float16
    eps = 1e-5
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')

    value= None
    expected = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)
    if implementation == "cublas":
         value = benchmark(torch.nn.functional.layer_norm, x, w_shape, weight, bias, eps)
    elif implementation == "triton":
         value = benchmark(layer_norm_forward, x, w_shape, weight, bias, eps)
    assert torch.allclose(value, expected, atol=1e-2)

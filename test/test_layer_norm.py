import torch
import pytest

from implementations.cuda_graph import cuda_graphs_wrapper
from implementations.layer_norm import layer_norm_forward


@pytest.mark.parametrize("size", [128, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("implementation", ["pytorch", "triton", "triton_cuda_graph"])
def test_benchmark_layer_norm(benchmark, size, implementation):
    torch.manual_seed(0)
    M = N = size
    eps = 1e-6

    weight = torch.rand((N,), dtype=torch.float16, device='cuda', requires_grad=False)
    bias = torch.randn_like(weight, dtype=torch.float16, device='cuda', requires_grad=False)
    x = -20 + 0.5 * torch.randn((M, N), dtype=torch.float16, device='cuda', requires_grad=False)
    expected = torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps)

    if implementation == "pytorch":
        value = benchmark(torch.nn.functional.layer_norm, x, weight.shape, weight, bias, eps)
    elif implementation == "triton":
        value = benchmark(layer_norm_forward, x, weight, bias, eps)
    elif implementation == "triton_cuda_graph":
        def wrapper(tensor):
            return layer_norm_forward(tensor, weight, bias, eps)

        run = cuda_graphs_wrapper(model=wrapper, inputs=[x], copy_outputs=False)
        value, = benchmark(run, x)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    assert torch.allclose(value, expected, atol=1e-2)

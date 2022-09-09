from typing import Callable

import torch
import pytest

from implementations.cuda_graph import cuda_graphs_wrapper
from implementations.layer_norm import layer_norm_forward

implementations: dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]] = {
    "pytorch": lambda x, weight, bias, eps: torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps),
    "triton": lambda x, weight, bias, eps: layer_norm_forward(x, weight, bias, eps),
}


@pytest.mark.parametrize("size", [128, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("cuda_graphs", [True, False])
@pytest.mark.parametrize("implementation", ["pytorch", "triton"])
def test_benchmark_layer_norm(benchmark, size: int, cuda_graphs: bool, implementation: str):
    assert implementation in implementations, f"Unknown implementation: {implementation}"

    torch.manual_seed(0)
    M = N = size
    eps = 1e-6

    weight = torch.rand((N,), dtype=torch.float16, device='cuda', requires_grad=False)
    bias = torch.randn_like(weight, dtype=torch.float16, device='cuda', requires_grad=False)
    x = -20 + 0.5 * torch.randn((M, N), dtype=torch.float16, device='cuda', requires_grad=False)
    expected = torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps)

    inference = implementations[implementation]
    if cuda_graphs:
        def func_wrapper(tensor):
            return inference(tensor, weight, bias, eps)

        run = cuda_graphs_wrapper(model=func_wrapper, inputs=[x], copy_outputs=False)

        def inference(x, *args, **kwargs):
            return run(x)[0]

    value = benchmark(inference, x, weight, bias, eps)

    assert torch.allclose(value, expected, atol=1e-2)

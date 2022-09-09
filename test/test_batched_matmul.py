import torch
import pytest
from implementations.batched_matmul import batched_matmul


@pytest.mark.parametrize("m", [24, 32], ids=lambda x: f"m={x}")
@pytest.mark.parametrize("n", [24, 32], ids=lambda x: f"n={x}")
@pytest.mark.parametrize("k", [24], ids=lambda x: f"k={x}")
@pytest.mark.parametrize("batch", [1, 16], ids=lambda x: f"batch={x}")
@pytest.mark.parametrize("implementation", ["pytorch", "triton"])
def test_benchmark(benchmark, m, n, k, batch, implementation):
    torch.manual_seed(0)

    a = torch.randn((batch, m, k), device='cuda', dtype=torch.float16, requires_grad=False)
    b = torch.randn((batch, k, n), device='cuda', dtype=torch.float16, requires_grad=False)
    expected = torch.matmul(a, b)
    if implementation == "pytorch":
        value = benchmark(torch.matmul, a, b)
    elif implementation == "triton":
        value = benchmark(batched_matmul, a, b)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    assert torch.allclose(value, expected, atol=1e-2)

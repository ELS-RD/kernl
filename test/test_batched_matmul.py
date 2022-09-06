import torch
import pytest
from implementations.batched_matmul import batched_matmul
from data_utils import generate_random_data


@pytest.mark.parametrize("m", [24, 32])
@pytest.mark.parametrize("n", [24, 32])
@pytest.mark.parametrize("k", [24])
@pytest.mark.parametrize("batch", [1, 16])
@pytest.mark.parametrize("implementation", ["cublas", "triton"])
def test_benchmark(benchmark, m, n, k, batch, implementation):
    a = generate_random_data((batch, m, k), 'cuda', torch.float16)
    b = generate_random_data((batch, k, n), 'cuda', torch.float16)
    expected = torch.matmul(a, b)
    if implementation == "cublas":
        value = benchmark(torch.matmul, a, b)
    elif implementation == "triton":
        value = benchmark(batched_matmul, a, b)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    torch.allclose(value, expected, atol=1e-2)

import pytest
import torch
from implementations.vector_addition import add

from conftest import assert_all_close, set_seed


@set_seed()
@pytest.mark.parametrize("n_elements", [2 ** i for i in range(5, 25, 5)], ids=lambda x: f"n_elements={x}")
@pytest.mark.parametrize("implementation", ["triton"])
def test_add_kernel(benchmark, n_elements, implementation):
    torch.manual_seed(0)
    x = torch.rand(n_elements, device="cuda", dtype=torch.float32)
    y = torch.rand(n_elements, device="cuda", dtype=torch.float32)
    expected = torch.add(x, y)
    if implementation == "pytorch":
        value = benchmark(torch.add, x, y)
    elif implementation == "triton":
        value = benchmark(add, x, y)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    assert_all_close(value, expected, atol=1e-2)

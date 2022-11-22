from collections import namedtuple

import pytest
import torch
import triton
from torch._inductor.triton_ops.autotune import get_cuda_stream

from conftest import assert_all_close, set_seed
from implementations.vector_addition import add, add_kernel


@set_seed()
@pytest.mark.parametrize("n_elements", [64, 128, 256], ids=lambda x: f"n_elements={x}")
def test_compile(n_elements):
    torch.manual_seed(0)
    x = torch.rand(n_elements, device='cuda', dtype=torch.float32)
    y = torch.rand(n_elements, device='cuda', dtype=torch.float32)
    output = torch.zeros(x.size(), device="cuda")
    expected = torch.add(x, y)

    configs = [namedtuple("instance_descriptor", [
            "divisible_by_16", "equal_to_1"])(
            tuple(range(4)),
            ())]
    constants = {"BLOCK_SIZE": n_elements}
    binary = triton.compile(
        fn=add_kernel,
        signature={0: "*i64", 1: "*i64", 2: "*i64", 3: "i32"},
        constants=constants,
        configs=configs
    )

    grid = lambda meta: (triton.cdiv(n_elements, constants["BLOCK_SIZE"]),)
    stream = get_cuda_stream(torch.cuda.current_device())
    grid_0 = grid(constants["BLOCK_SIZE"])[0]
    binary.c_wrapper(
        grid_0, 1, 1, binary.num_warps, binary.shared, stream, binary.cu_function, x, y, output, n_elements)

    assert_all_close(output, expected, atol=1e-2)


@set_seed()
@pytest.mark.parametrize("n_elements", [2 ** i for i in range(1, 4, 1)], ids=lambda x: f"n_elements={x}")
@pytest.mark.parametrize("implementation", ["triton"])
def test_add_kernel(benchmark, n_elements, implementation):
    torch.manual_seed(0)
    x = torch.rand(n_elements, device='cuda', dtype=torch.float32)
    y = torch.rand(n_elements, device='cuda', dtype=torch.float32)
    expected = torch.add(x, y)
    if implementation == "pytorch":
        value = benchmark(torch.add, x, y)
    elif implementation == "triton":
        value = benchmark(add, x, y)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    assert_all_close(value, expected, atol=1e-2)

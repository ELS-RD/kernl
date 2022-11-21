from collections import namedtuple

import pytest
import torch
import triton
from torch._inductor.triton_ops.autotune import get_cuda_stream

from conftest import assert_all_close, set_seed
from implementations.vector_addition import add, add_kernel


@set_seed()
@pytest.mark.parametrize("n_elements", [2 ** i for i in range(12, 26, 4)], ids=lambda x: f"n_elements={x}")
def test_compile(n_elements):
    torch.manual_seed(0)
    x = torch.rand(n_elements, device='cuda', dtype=torch.float32)
    y = torch.rand(n_elements, device='cuda', dtype=torch.float32)
    output = torch.zeros(x.size(), device="cuda")
    expected = torch.add(x, y)

    config = namedtuple("instance_descriptor", [
            "divisible_by_16", "equal_to_1"])(
            tuple(range(4)),
            ())
    configs = [config]
    binary = triton.compile(fn=add_kernel, signature={0: "*i32", 1: "*i32", 2: "*i32", 3: "i32"}, device=0, configs=configs)
    scope = {
        "grid_meta": {"BLOCK_SIZE": 1024},
        "bin": binary,
        "torch": torch,
        "set_device": torch.cuda.set_device,
        "current_device": torch.cuda.current_device,
    }
    exec(
        f"""
        def launcher(x, y, output, n_elements, grid, stream):
            set_device(current_device())
            grid_0 = grid(grid_meta["BLOCK_SIZE"])[0]
            bin.c_wrapper(grid_0, 1, 1, bin.num_warps, bin.shared,
                          stream, bin.cu_function, x, y, output, n_elements)
        """.lstrip(),
        scope,
    )

    launcher = scope["launcher"]
    launcher.config = triton.Config({"BLOCK_SIZE": 1024})
    stream = get_cuda_stream(torch.cuda.current_device())
    grid = lambda meta: (triton.cdiv(n_elements, 1024),)
    value = launcher(x, y, output, n_elements, grid, stream=stream)
    assert_all_close(value, expected, atol=1e-2)


@set_seed()
@pytest.mark.parametrize("n_elements", [2 ** i for i in range(12, 26, 4)], ids=lambda x: f"n_elements={x}")
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

import pytest
import torch

from kernl.debugger.memory_map import MemoryMap


# TODO: better test
def test_store():
    memory = MemoryMap()
    t = torch.rand((10,), device="cuda")
    ptr = memory.add_tensor(t)

    original = t.clone()

    expected = t.clone()
    expected[1] = 10.0

    memory.store(
        torch.tensor([ptr + 1], device="cuda"),
        torch.tensor([10.0], device="cuda"),
        mask=torch.tensor([False], device="cuda"),
    )
    assert torch.equal(original, t)

    memory.store(
        torch.tensor([ptr + 1], device="cuda"),
        torch.tensor([10.0], device="cuda"),
        mask=torch.tensor([True], device="cuda"),
    )
    assert torch.equal(expected, t)


def test_load_spanning_error():
    pytest.skip()


def test_load_square_mask():
    pytest.skip()


def test_load_broadcast_mask():
    pytest.skip()


# Ensure we can't accidentally edit memory from the loaded tensor
def test_load_is_in_different_memory():
    memory = MemoryMap()
    a = torch.rand(
        (
            10,
            100,
        ),
        device="cuda",
    )
    ptr_a = memory.add_tensor(a)

    t = memory.load(ptr_a + torch.arange(a.numel(), device="cuda"))
    assert t.storage().data_ptr() != a.storage().data_ptr()


def test_load_not_contiguous():
    original_a = torch.rand(
        (
            10,
            100,
        ),
        device="cuda",
    )
    expected_a = original_a.flatten()

    transposed_a = original_a.transpose(-1, -2)
    memory = MemoryMap()
    ptr_a = memory.add_tensor(transposed_a)

    assert not torch.equal(original_a, transposed_a)
    assert original_a.stride() != transposed_a.stride()

    # When we load transposed matrix, it loads original matrix, because memory not changed
    assert torch.equal(expected_a, memory.load(ptr_a + torch.arange(transposed_a.numel(), device="cuda")))


def test_load_after_store():
    memory = MemoryMap()
    a = torch.rand(
        (
            10,
            100,
        ),
        device="cuda",
    )
    ptr_a = memory.add_tensor(a)

    t = memory.load(ptr_a + torch.arange(a.numel(), device="cuda"))
    mult_t = t * 5
    memory.store(ptr_a + torch.arange(a.numel(), device="cuda"), mult_t)

    value = memory.load(ptr_a + torch.arange(a.numel(), device="cuda"))
    assert torch.equal(value, mult_t)


def test_load_multiple_tensor():
    memory = MemoryMap()
    a = torch.rand(
        (
            10,
            100,
        ),
        device="cuda",
    )
    ptr_a = memory.add_tensor(a)

    b = torch.rand(
        (
            10,
            100,
        ),
        device="cuda",
    )
    ptr_b = memory.add_tensor(b)

    loaded_a = memory.load(ptr_a + torch.arange(a.numel(), device="cuda"))
    loaded_b = memory.load(ptr_b + torch.arange(a.numel(), device="cuda"))
    assert torch.equal(loaded_a, a.flatten())
    assert torch.equal(loaded_b, b.flatten())


def test_load_vect_mask():
    t = torch.rand((100,), device="cuda")
    mask = torch.arange(100, device="cuda") > 22
    other = 10.0
    expected = torch.where(mask, t, other)

    memory = MemoryMap()
    ptr = memory.add_tensor(t)

    without_mask = memory.load(ptr + torch.arange(100, device="cuda"))
    assert torch.equal(without_mask, t)

    with_mask = memory.load(ptr + torch.arange(100, device="cuda"), mask=mask, other=other)
    assert torch.equal(with_mask, expected)


@pytest.mark.parametrize("shape", [(10, 100), (1, 12), (20, 10)])
@pytest.mark.parametrize("transpose", [False, True])
def test_load_square(shape, transpose):
    memory = MemoryMap()
    t = torch.rand(shape, device="cuda")
    if transpose:
        t = t.transpose(-1, -2)
    ptr = memory.add_tensor(t)

    # Load by line
    for i in range(0, t.size(0)):
        assert torch.equal(
            memory.load(ptr + i * t.stride(0) + torch.arange(t.size(1), device="cuda") * t.stride(1)), t[i]
        )

    # Load by column
    for i in range(0, t.size(1)):
        assert torch.equal(
            memory.load(ptr + t.stride(0) * torch.arange(t.size(0), device="cuda") + i * t.stride(1)), t[:, i]
        )


@pytest.mark.parametrize("config", [(10, 4, 2), (12, 2, 10), (100, 50, 1), (1, 0, 1)])
def test_load_partial_vect(config):
    shape, start_index, size = config

    memory = MemoryMap()
    t = torch.rand(shape, device="cuda")
    ptr = memory.add_tensor(t)

    index_to_load = start_index + torch.arange(size, device="cuda")
    assert torch.equal(memory.load(ptr + index_to_load), t[start_index : start_index + size])


@pytest.mark.parametrize("torch_type", [torch.float16, torch.float32, torch.int8, torch.int32])
def test_load_type_matches_tensor_type(torch_type):
    memory = MemoryMap()
    t = torch.empty((10, 100), dtype=torch_type, device="cuda")
    ptr = memory.add_tensor(t)

    assert memory.load(ptr + torch.arange(1, device="cuda")).dtype == t.dtype

import torch

from kernl.optimizer.cuda_graph import argsort, prepare_inputs
from kernl.optimizer.pool_cuda_graphs import CudaGraphPool, get_aligned_size


def test_argsort():
    assert argsort(iterable=[1, 5, 3, 4, 5], key=lambda x: x) == [0, 2, 3, 1, 4]


def test_aligned_size():
    t = torch.ones((33,))
    assert get_aligned_size(t, alignment=16) == 144
    assert get_aligned_size(t, alignment=32) == 160
    t = torch.ones((1,))
    assert get_aligned_size(t, alignment=8) == 8


def test_cuda_graph_pool_e2e():
    all_inputs = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6, 7], dtype=torch.bfloat16),
        torch.tensor([8]),
        torch.tensor([9, 10, 11, 12], dtype=torch.double),
        torch.tensor([13, 14], dtype=torch.int8),
        torch.tensor([15], dtype=torch.float32),
        torch.tensor([16], dtype=torch.int64),
    ]
    all_pools: list[CudaGraphPool] = [CudaGraphPool(32, device="cpu"), CudaGraphPool(48, device="cpu")]

    new_tensors: list[torch.Tensor] = prepare_inputs(inputs=all_inputs, pools=all_pools)
    assert len(new_tensors) == len(all_inputs)
    assert len(all_pools) == 3
    for original, copy in zip(all_inputs, new_tensors):
        assert original.shape == copy.shape
        assert original.stride() == copy.stride()
        assert original.dtype == copy.dtype
        assert original.device == copy.device
        assert torch.all(original == copy)
        assert original.data_ptr() != copy.data_ptr()


def test_cuda_graph_pool():
    c = CudaGraphPool(8, device="cpu")
    t = torch.ones((16,), dtype=torch.int64)
    assert not c.can_store(t)
    t = torch.ones((1,), dtype=torch.int8)
    assert c.can_store(t)
    assert c.offset == 0
    c.copy_to_pool(t)
    assert c.offset > 0
    t = t.cuda()
    assert not c.can_store(t)
    c.reset()
    assert c.offset == 0


def test_exact_cg_copy():
    pool = CudaGraphPool(16)
    tensor = torch.tensor([1, 1, 1, 1], dtype=torch.float32, device="cuda")
    assert len(tensor.untyped_storage()) == 16
    pool.copy_to_pool(tensor)

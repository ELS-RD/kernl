import torch

from kernl.optimizer.cuda_graph import prepare_inputs
from kernl.optimizer.pool_cuda_graphs import CudaGraphPool, get_aligned_size


def test_aligned_size():
    t = torch.ones((33,))
    assert get_aligned_size(t, alignment=16) == 144
    assert get_aligned_size(t, alignment=32) == 160
    t = torch.ones((1,))
    assert get_aligned_size(t, alignment=8) == 8


def test_cuda_graph_pool():
    all_inputs = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6, 7], dtype=torch.bfloat16),
        torch.tensor([8]),
        torch.tensor([9, 10, 11, 12], dtype=torch.double),
        torch.tensor([13, 14], dtype=torch.int8),
        torch.tensor([15], dtype=torch.float32),
        torch.tensor([16], dtype=torch.int64),
    ]
    all_pools = [CudaGraphPool(30), CudaGraphPool(50)]

    new_tensors = prepare_inputs(inputs=all_inputs, pools=all_pools)
    assert len(new_tensors) == len(all_inputs)
    assert len(all_pools) == 3
    for original, copy in zip(all_inputs, new_tensors):
        assert torch.all(original == copy)
        assert original.data_ptr() != copy.data_ptr()
        assert original.dtype == copy.dtype
        assert original.device == copy.device
        assert original.size() == copy.size()
        assert original.stride() == copy.stride()

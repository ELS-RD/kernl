import torch


class CudaGraphPool:
    """
    Memory pool for CUDA graphs.
    """

    def __init__(self, size, dtype=torch.int8, device="cuda"):  # TODO change device to cuda
        """
        :param size: size of the pool in bytes. Must be a multiple of the dtype size.
        :param dtype: dtype of the pool
        """
        assert size > 8, "Size must be at least 8 bytes"
        element_size = torch.tensor([], dtype=dtype).element_size()
        assert size % element_size == 0, "Size must be a multiple of the dtype size"
        self.pool: torch.Tensor = torch.empty(size // element_size, dtype=dtype, device=device)
        self.size = len(self.pool.untyped_storage())
        self.offset = 0

    def copy_to_pool(self, t: torch.Tensor) -> torch.Tensor:
        """
        Copy the tensor t in the pool and return a tensor that is a view of the pool.
        :param t: tensor to copy in the pool
        :return: tensor copy (that is a view of the pool)
        """
        assert t.device == self.pool.device
        # 64 bits alignment
        tensor_aligned_size = get_aligned_size(t)
        new_offset = self.offset + tensor_aligned_size
        assert new_offset <= self.size, "Memory pool is full"
        # offset is expressed in t.dtype number of elements
        new_t = torch.as_strided(
            self.pool.view(t.dtype), size=t.size(), stride=t.stride(), storage_offset=self.offset // t.element_size()
        )
        new_t.copy_(t)
        self.offset = new_offset
        return new_t

    def can_store(self, t: torch.Tensor) -> bool:
        """
        Check if the tensor t can be stored in the pool.
        :param t: tensor to check
        :return: True if the tensor can be stored in the pool
        """

        return self.size - self.offset >= get_aligned_size(t)

    def reset(self):
        """
        Reset the pool offset to 0.
        """
        self.offset = 0


def get_aligned_size(t: torch.Tensor, alignment=64) -> int:
    """
    Get the aligned size of the tensor t.
    :param t: tensor to get the aligned size of
    :param alignment: alignment size
    :return: aligned size
    """
    storage_len = len(t.untyped_storage())
    alined_storage_len = (storage_len + alignment - 1) // alignment * alignment
    return alined_storage_len

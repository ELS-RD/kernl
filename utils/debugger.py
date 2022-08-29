import random
from itertools import product
from typing import Optional, Union

import torch
from torch import Tensor

from utils.range_dict import RangeKeyDict


class TritonDebugger:
    float32 = torch.float32
    float16 = torch.float32  # to run on torch cpu which has a low support for fp16

    def __init__(self, grid: list[int], inputs: list[torch.Tensor], shuffle: bool = False):
        """
        Initialize a serialized triton code runner.
        :param grid: execution grid, should match what you would provide to Cuda
        :param inputs: a list of all tensors you plan to use in triton code. Will generate fake pointers.
        :param shuffle: suffle execution order like in parallel execution. Helps to avoid mistakes like relying on
            some order which is not possible in parallel execution.
        """
        self.grid_positions = list(product(*(range(axis) for axis in grid)))
        if shuffle:
            random.shuffle(self.grid_positions)

        self.current_grid_position = None
        self.constexpr = int | str
        previous_boundary = 0

        range_tensor_dict = dict()
        for t in inputs:
            range_tensor_dict[(previous_boundary, previous_boundary + t.nelement())] = t
            previous_boundary = previous_boundary + t.nelement()

        self.range_tensor_dict = RangeKeyDict(range_tensor_dict)
        self.tensor_ptr: dict[torch.Tensor, int] = {tensor: range_ptrs[0] for range_ptrs, tensor in range_tensor_dict.items()}
        self.total_gm_read = 0
        self.total_gm_write = 0

    def new_program(self):
        """
        Update program id from the grid.
        """
        self.current_grid_position = self.grid_positions.pop(0)

    def program_id(self, axis: int) -> torch.Tensor:
        """
        Get program id for the given axis.
        :param axis: axis to get program id for. Should be 0, 1, 2.
        :return: program ID as a tensor
        """
        assert axis in [0, 1, 2]
        return torch.tensor(self.current_grid_position[axis])

    def has_next(self) -> bool:
        """
        Check if there is another program to run.
        :return: True if there is another program to run, False otherwise.
        """
        return len(self.grid_positions) > 0

    @staticmethod
    def offset_to_indexes(tensor: torch.Tensor, offsets: torch.Tensor) -> list[torch.Tensor]:
        """
        Convert the offsets to indexes of a given tensor using each axis stride.
        :param tensor: tensor to get indexes for.
        :param offsets: offsets to convert to indexes.
        :return: list of indexes for each axis.
        """
        coordinates: list[torch.Tensor] = list()
        for dim in range(0, tensor.ndim):
            dim_stride = tensor.stride(dim)
            dim_index = torch.div(offsets, dim_stride, rounding_mode='floor')
            coordinates.append(dim_index.long())
            offsets -= dim_index * dim_stride

        return coordinates

    def _get_tensor(self, ptr: torch.Tensor) -> torch.Tensor:
        """
        Get tensor from the input dictionary.
        :param ptr: pointer to the tensor.
        :return:
        """
        first_elem_ptr = ptr.flatten()[0].item()
        tensor = self.range_tensor_dict[first_elem_ptr]
        return tensor

    def _get_indexes(self, tensor: torch.Tensor, ptr: torch.Tensor, mask: torch.Tensor) -> list[Tensor]:
        """
        Convert the pointers to indexes of a given tensor.
        :param tensor: tensor to get indexes for.
        :param ptr: pointer to the tensor.
        :param mask: mask to apply to the pointers.
        :return: list of indexes.
        """
        first_ptr = self.tensor_ptr[tensor]
        offsets = ptr - first_ptr
        offsets = offsets[mask]
        indexes: list[Tensor] = self.offset_to_indexes(tensor=tensor, offsets=offsets)
        return indexes

    def load(self, ptr: torch.Tensor, mask: Optional[torch.Tensor] = None, other: float = 0., eviction_policy: str = "") -> torch.Tensor:
        """
        Load data from the provided pointers / mask.
        :param ptr: pointers to the data.
        :param mask: mask to apply to the pointers.
        :param other: value to return if the position is masked.
        :param eviction_policy: not used, just for compatibility
        :return: loaded data.
        """
        if mask is None:
            mask = torch.ones_like(ptr).bool()
        tensor = self._get_tensor(ptr=ptr)
        indexes = self._get_indexes(tensor=tensor, ptr=ptr, mask=mask)
        block = torch.full_like(ptr, fill_value=other, dtype=tensor.dtype)
        block[mask] = tensor[indexes]
        self.total_gm_read += mask.sum().item()
        return block

    def store(self, ptr: torch.Tensor, data: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """
        Store tensor to the provided pointers / mask.
        :param ptr: pointers where to store the provided data.
        :param data: data to store.
        :param mask: mask to apply to the pointers/data.
        """
        if mask is None:
            mask = torch.ones_like(ptr).bool()
        tensor = self._get_tensor(ptr)
        indexes = self._get_indexes(tensor=tensor, ptr=ptr, mask=mask)
        tensor[indexes] = data[mask]
        self.total_gm_write += mask.sum().item()

    def get_ptr(self, tensor: torch.Tensor) -> int:
        """
        Get pointer to the beginning of the given tensor.
        :param tensor: input tensor
        :return: pointer as an integer
        """
        return self.tensor_ptr[tensor]

    @staticmethod
    def arange(start: int, end: int) -> torch.Tensor:
        """
        Returns contiguous values within the open interval [start, end).
        @param start: start of the interval
        @param end: end of the interval. Must be a power of two >= start.
        @return: a tensor of size (end - start) with values in [start, end).
        """
        assert (end & (end-1) == 0) and end != 0, f"end must be a power of 2: {end}"
        assert start < end, f"start must be less than end: {start} > {end}"
        return torch.arange(start=start, end=end)

    @staticmethod
    def cdiv(x: int, y: int) -> int:
        """
        Ceiling division returns the closest integer greater than or equal to the quotient.
        """
        return (x + y - 1) // y

    @staticmethod
    def max(x: torch.Tensor, axis=0) -> torch.Tensor:
        return torch.max(x, dim=axis).values

    @staticmethod
    def exp(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    @staticmethod
    def sum(x: torch.Tensor, axis=0) -> torch.Tensor:
        return torch.sum(x, dim=axis)

    @staticmethod
    def zeros(shape: (int, int), dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(size=shape, dtype=dtype)

    @staticmethod
    def dot(input: torch.Tensor, other: torch.Tensor, trans_a: bool = False, trans_b: bool = False, allow_tf32=True) -> torch.Tensor:
        if trans_a:
            input = input.T
        if trans_b:
            other = other.T
        return torch.matmul(input=input, other=other)

    @staticmethod
    def where(condition: torch.BoolTensor, x: torch.Tensor, y: Union[torch.Tensor, float]) -> torch.Tensor:
        return torch.where(condition, x, y)

    @staticmethod
    def rand(seed: int, offset: torch.Tensor, n_rounds: int = 10):
        """
        Generate random data. Seed is not used as it would produce always the same pattern.
        :param seed: not used
        :param offset: random tensor will have offset shape.
        :param n_rounds: not used
        :return:
        """
        # seed not used as it would produce always the same pattern
        return torch.rand(offset.shape)

    @staticmethod
    def sqrt(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x)

    @staticmethod
    def multiple_of(input: int, values: int) -> int:
        """
        Used to type the compiler, we just return input argument.
        :param input: some input
        :param values: input guaranted multiple of.
        :return: input argument.
        """
        return input

    @staticmethod
    def maximum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.max(x, y)

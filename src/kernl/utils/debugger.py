#  Copyright 2022 Lefebvre Sarrut
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import random
from itertools import product
from typing import Optional, Union

import torch
from torch import Tensor

from kernl.utils.range_dict import RangeKeyDict


class TritonDebugger:
    float32 = torch.float32
    float16 = torch.float16

    def __init__(self, grid: list[int], shuffle: bool = False):
        """Initialize a serialized triton code runner.

        Args:
            grid: execution grid, should match what you would provide to Cuda
            shuffle: suffle execution order like in parallel execution. Helps to avoid mistakes like relying on some order which is not possible in parallel execution.
        """
        self.grid_positions = list(product(*(range(axis) for axis in grid)))
        if shuffle:
            random.shuffle(self.grid_positions)

        self.current_grid_position = None
        self.constexpr = Union[int, str]
        self.previous_boundary = 0
        self.tensor_dict: dict[tuple(int, int), torch.Tensor] = dict()
        self.range_tensor_dict: RangeKeyDict = dict()
        self.tensor_ptr: dict[torch.Tensor, int] = dict()
        self.total_gm_read = 0
        self.total_gm_write = 0

    def _add_input_if_not_exists(self, tensor: torch.Tensor):
        """Add inputs for triton debugger.

        Args:
            tensor: torch.Tensor to be added as input
        """
        if any([x is tensor for x in self.tensor_dict.values()]):
            return
        assert tensor.device.type == "cuda", f"Tensor {tensor} is not on cuda"
        self.tensor_dict[(self.previous_boundary, self.previous_boundary + tensor.nelement())] = tensor
        self.previous_boundary = self.previous_boundary + tensor.nelement()
        self.tensor_ptr = {tensor: range_ptrs[0] for range_ptrs, tensor in self.tensor_dict.items()}
        self.range_tensor_dict = RangeKeyDict(self.tensor_dict)

    def new_program(self):
        """Update program id from the grid."""
        self.current_grid_position = self.grid_positions.pop(0)

    def program_id(self, axis: int) -> torch.Tensor:
        """Get program id for the given axis.

        Args:
            axis: axis to get program id for. Should be 0, 1, 2.

        Returns:
            program ID as a tensor
        """
        assert axis in [0, 1, 2]
        return torch.tensor(self.current_grid_position[axis])

    def has_next(self) -> bool:
        """Check if there is another program to run.

        Returns:
            True if there is another program to run, False otherwise.
        """
        return len(self.grid_positions) > 0

    @staticmethod
    def offset_to_indexes(tensor: torch.Tensor, offsets: torch.Tensor) -> list[torch.Tensor]:
        """Convert the offsets to indexes of a given tensor using each axis stride.

        Args:
            tensor: tensor to get indexes for.
            offsets: offsets to convert to indexes.

        Returns:

            list of indexes for each axis.
        """
        coordinates: list[torch.Tensor] = list()
        for dim in range(0, tensor.ndim):
            dim_stride = tensor.stride(dim)
            dim_index = torch.div(offsets, dim_stride, rounding_mode="floor")
            coordinates.append(dim_index.long())
            offsets -= dim_index * dim_stride

        return coordinates

    def _get_tensor(self, ptr: torch.Tensor) -> torch.Tensor:
        """Get tensor from the input dictionary.

        Args:
            ptr: pointer to the tensor.
        """
        first_elem_ptr = ptr.flatten()[0].item()
        tensor = self.range_tensor_dict[first_elem_ptr]
        return tensor

    def _get_indexes(self, tensor: torch.Tensor, ptr: torch.Tensor, mask: torch.Tensor) -> list[Tensor]:
        """Convert the pointers to indexes of a given tensor.

        Args:
            tensor: tensor to get indexes for.
            ptr: pointer to the tensor.
            mask: mask to apply to the pointers.

        Returns:
            list of indexes.
        """
        first_ptr = self.tensor_ptr[tensor]
        offsets = ptr - first_ptr
        offsets = offsets[mask]
        indexes: list[Tensor] = self.offset_to_indexes(tensor=tensor, offsets=offsets)
        return indexes

    def load(
        self, ptr: torch.Tensor, mask: Optional[torch.Tensor] = None, other: float = 0.0, eviction_policy: str = ""
    ) -> torch.Tensor:
        """Load data from the provided pointers / mask.

        Args:
            ptr: pointers to the data.
            mask: mask to apply to the pointers.
            other: value to return if the position is masked.
            eviction_policy: not used, just for compatibility

        Returns:
            loaded data.
        """
        if mask is None:
            mask = torch.ones_like(ptr).bool()
        tensor = self._get_tensor(ptr=ptr)
        indexes = self._get_indexes(tensor=tensor, ptr=ptr, mask=mask)
        block = torch.full_like(ptr, fill_value=other, dtype=tensor.dtype, device="cuda")
        block[mask] = tensor[indexes]
        self.total_gm_read += mask.sum().item()
        return block

    def store(self, ptr: torch.Tensor, data: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """Store tensor to the provided pointers / mask.

        Args:
            ptr: pointers where to store the provided data.
            data: data to store.
            mask: mask to apply to the pointers/data.
        """
        if mask is None:
            mask = torch.ones_like(ptr).bool()
        tensor = self._get_tensor(ptr)
        indexes = self._get_indexes(tensor=tensor, ptr=ptr, mask=mask)
        tensor[indexes] = data[mask]
        self.total_gm_write += mask.sum().item()

    def get_ptr(self, tensor: torch.Tensor) -> int:
        """Get pointer to the beginning of the given tensor.

        Args:
            tensor: input tensor

        Returns:
            pointer as an integer
        """
        self._add_input_if_not_exists(tensor)
        return self.tensor_ptr[tensor]

    @staticmethod
    def arange(start: int, end: int) -> torch.Tensor:
        """Returns contiguous values within the open interval [start, end).

        Args:
            start: start of the interval
            end: end of the interval. Must be a power of two >= start.

        Returns:
            a tensor of size (end - start) with values in [start, end).
        """
        assert (end & (end - 1) == 0) and end != 0, f"end must be a power of 2: {end}"
        assert start < end, f"start must be less than end: {start} > {end}"
        return torch.arange(start=start, end=end, device="cuda")

    @staticmethod
    def cdiv(x: int, y: int) -> int:
        """Ceiling division returns the closest integer greater than or equal to the quotient."""
        return (x + y - 1) // y

    @staticmethod
    def max(x: torch.Tensor, axis=0) -> torch.Tensor:
        return torch.max(x, dim=axis).values

    @staticmethod
    def min(x: torch.Tensor, axis=0) -> torch.Tensor:
        return torch.min(x, dim=axis).values

    @staticmethod
    def exp(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    @staticmethod
    def sum(x: torch.Tensor, axis=0) -> torch.Tensor:
        return torch.sum(x, dim=axis)

    @staticmethod
    def zeros(shape: (int, int), dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(size=shape, dtype=dtype, device="cuda")

    @staticmethod
    def dot(
        input: torch.Tensor, other: torch.Tensor, trans_a: bool = False, trans_b: bool = False, allow_tf32=True
    ) -> torch.Tensor:
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
        """Generate random data. Seed is not used as it would produce always the same pattern.

        Args:
            seed: not used
            offset: random tensor will have offset shape.
            n_rounds: not used
        """
        # seed not used as it would produce always the same pattern
        return torch.rand(offset.shape, device="cuda")

    @staticmethod
    def sqrt(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x)

    @staticmethod
    def multiple_of(input: int, values: int) -> int:
        """Used to type the compiler, we just return input argument.

        Args:
            input: some input
            values: input guaranted multiple of.

        Returns:
            input argument.
        """
        return input

    @staticmethod
    def maximum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.max(x, y)

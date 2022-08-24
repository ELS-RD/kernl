import random
from itertools import product
from typing import Optional, List

import torch
from torch import Tensor


class RangeKeyDict:
    def __init__(self, my_dict):
        assert not any(map(lambda x: not isinstance(x, tuple) or len(x) != 2 or x[0] > x[1], my_dict))

        def lte(bound):
            return lambda x: bound <= x

        def gt(bound):
            return lambda x: x < bound

        # generate the inner dict with tuple key like (lambda x: 0 <= x, lambda x: x < 100)
        self._my_dict = {(lte(k[0]), gt(k[1])): v for k, v in my_dict.items()}

    def __getitem__(self, number):
        from functools import reduce
        _my_dict = self._my_dict
        try:
            result = next((_my_dict[key] for key in _my_dict if list(reduce(lambda s, f: filter(f, s), key, [number]))))
        except StopIteration:
            raise KeyError(number)
        return result

    def get(self, number, default=None):
        try:
            return self.__getitem__(number)
        except KeyError:
            return default


class TritonDebugger:

    def __init__(self, grid: list[int], inputs: list[torch.Tensor], shuffle: bool = True):
        self.grid_positions = list(product(*(range(axis) for axis in grid)))
        if shuffle:
            random.shuffle(self.grid_positions)

        self.current_grid_position = None
        self.constexpr = int
        previous_boundary = 0

        d = dict()
        for t in inputs:
            d[(previous_boundary, previous_boundary + t.nelement())] = t
            previous_boundary = previous_boundary + t.nelement()

        self.inputs = RangeKeyDict(d)
        self.tensor_ptr: dict[torch.Tensor, int] = {v: k[0] for k, v in d.items()}

    def increment(self):
        self.current_grid_position = self.grid_positions.pop(0)

    def program_id(self, axis: int) -> torch.Tensor:
        return torch.tensor(self.current_grid_position[axis])

    def has_next(self) -> bool:
        return len(self.grid_positions) > 0

    @staticmethod
    def arange(start: int, end: int) -> torch.Tensor:
        """
        Returns contiguous values within the open interval [start, end).
        @param start: start of the interval
        @param end: end of the interval. Must be a power of two >= start.
        @return: a tensor of size (end - start) with values in [start, end).
        """
        assert (end & (end-1) == 0) and end != 0, "end must be a power of 2"
        assert start < end, "start must be less than end"
        return torch.arange(start=start, end=end)

    def get_tensor(self, ptr: torch.Tensor) -> torch.Tensor:
        first_elem_ptr = ptr.flatten()[0].item()
        tensor = self.inputs[first_elem_ptr]
        return tensor

    def get_indexes(self, tensor: torch.Tensor, ptr: torch.Tensor, mask: torch.Tensor) -> list[Tensor]:
        first_ptr = self.tensor_ptr[tensor]
        offsets = ptr - first_ptr
        offsets = offsets[mask]
        indexes: list[Tensor] = self.offset_to_indexes(offsets=offsets, tensor=tensor)
        return indexes

    def load(self, ptr: torch.Tensor, mask: Optional[torch.Tensor] = None, other: float = 0.) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(ptr).bool()
        tensor = self.get_tensor(ptr=ptr)
        indexes = self.get_indexes(tensor=tensor, ptr=ptr, mask=mask)
        block = torch.full_like(ptr, fill_value=other, dtype=tensor.dtype)
        block[mask] = tensor[indexes]
        return block

    def store(self, ptr: torch.Tensor, data: torch.Tensor, mask: torch.Tensor) -> None:
        tensor = self.get_tensor(ptr)
        indexes = self.get_indexes(tensor=tensor, ptr=ptr, mask=mask)
        tensor[indexes] = data[mask]

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
    def offset_to_indexes(offsets: torch.Tensor, tensor: torch.Tensor) -> list[torch.Tensor]:
        coordinates: list[torch.Tensor] = list()
        for dim in range(0, tensor.ndim):
            dim_stride = tensor.stride(dim)
            dim_index = torch.div(offsets, dim_stride, rounding_mode='floor')
            coordinates.append(dim_index.long())
            offsets -= dim_index * dim_stride

        return coordinates

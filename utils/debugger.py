from typing import Optional

import torch


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

    def __init__(self, grid: list[int], inputs: list[torch.Tensor]):
        self.grid = grid
        self.current_grid_position = [0] * len(grid)
        self.constexpr = int
        previous_boundary = 0

        d = dict()
        for t in inputs:
            d[(previous_boundary, previous_boundary + t.nelement())] = t
            previous_boundary = previous_boundary + t.nelement()

        self.inputs = RangeKeyDict(d)
        self.tensor_ptr: dict[torch.Tensor, int] = {v: k[0] for k, v in d.items()}

    def program_id(self, axis: int) -> torch.Tensor:
        if self.current_grid_position[axis] > self.grid[axis]:
            raise Exception("Program id out of bounds")

        current_position = self.current_grid_position[axis]
        self.current_grid_position[axis] += 1
        return torch.tensor(current_position)

    def has_next(self) -> bool:
        return self.current_grid_position[0] < self.grid[0]

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

    def load(self, ptr: torch.Tensor, mask: Optional[torch.Tensor] = None, other: Optional[float] = None) -> torch.Tensor:
        first_elem_ptr = ptr.flatten()[0].item()
        tensor = self.inputs[first_elem_ptr]
        first_ptr = self.tensor_ptr[tensor]
        tensor_flat = tensor.flatten()
        assert tensor_flat.data_ptr() == tensor.data_ptr()
        indexes = ptr - first_ptr
        if mask is not None:
            if other is None:
                indexes = indexes[mask]
                return tensor_flat[indexes]
            else:
                block = torch.full_like(ptr, fill_value=other, dtype=tensor_flat.dtype)
                indexes = indexes[mask]
                data = tensor_flat[indexes]
                selection = [slice(0, s) for s in indexes.size()]
                block[selection] = data
                return block

    def store(self, ptr: torch.Tensor, data: torch.Tensor, mask: torch.Tensor) -> None:
        first_elem_ptr = ptr.flatten()[0].item()
        t = self.inputs[first_elem_ptr]
        first_ptr = self.tensor_ptr[t]
        ptr = ptr - first_ptr
        ptr = ptr[mask]
        t_flat = t.flatten()
        assert t_flat.data_ptr() == t.data_ptr()
        t_flat[ptr] = data
        t_flat.resize_(t.shape)

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


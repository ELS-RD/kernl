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
            d[(previous_boundary, previous_boundary+t.nelement())] = t
            previous_boundary = previous_boundary+t.nelement()

        self.inputs = RangeKeyDict(d)
        self.tensor_ptr: dict[torch.Tensor, int] = {v: k[0] for k, v in d.items()}

    def program_id(self, axis: int) -> int:
        if self.current_grid_position[axis] > self.grid[axis]:
            raise Exception("Program id out of bounds")

        current_position = self.current_grid_position[axis]
        self.current_grid_position[axis] += 1
        return current_position

    def has_next(self) -> bool:
        return self.current_grid_position[0] < self.grid[0]

    def arange(self, start: int, end: int) -> torch.Tensor:
        return torch.arange(start=start, end=end)

    def load(self, ptr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        first_elem_ptr = ptr.data[0].item()
        t = self.inputs[first_elem_ptr]
        first_ptr = self.tensor_ptr[t]
        ptr = ptr - first_ptr
        mask_index, = (mask == True).nonzero(as_tuple=True)
        ptr = ptr[mask_index]
        return t[ptr]

    def store(self, ptr: torch.Tensor, data: torch.Tensor, mask: torch.Tensor) -> None:
        first_elem_ptr = ptr.data[0].item()
        t = self.inputs[first_elem_ptr]
        first_ptr = self.tensor_ptr[t]
        ptr = ptr - first_ptr
        mask_index, = (mask == True).nonzero(as_tuple=True)
        ptr = ptr[mask_index]
        t[ptr] = data

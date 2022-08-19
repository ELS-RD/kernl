from typing import Dict, Tuple

import torch


Size = Tuple[int, ...]


class CudaGraphStaticTensor:
    default_size: int = 30000

    def __init__(self, name: str, device: torch.device, dtype: torch.dtype):
        self.name: str = name
        self.device = device
        self.dtype = dtype
        self.static_size = self.default_size
        self.tensor = torch.empty((self.static_size,), device=self.device, dtype=self.dtype)
        self.needs_graph_recompilation = False

    def update_content(self, tensor: torch.Tensor):
        assert self.tensor.device == tensor.device
        assert self.tensor.dtype == tensor.dtype
        self.resize(tensor.size())
        self.update_size(tensor.size())
        self.tensor.copy_(tensor)

    def update_size(self, size: Size):
        self.tensor.resize_(size)

    def needs_resize(self, size: torch.Size):
        return size.numel() > self.static_size

    def resize(self, size: torch.Size):
        needs_resize = self.needs_resize(size)
        if needs_resize:
            while self.needs_resize(size):
                self.static_size *= 2
            self.tensor = torch.empty((self.static_size,), device=self.device, dtype=self.dtype)
            self.needs_graph_recompilation = True

    @property
    def size(self):
        return self.tensor.size()


CudaGraphStaticTensors = Dict[str, CudaGraphStaticTensor]

from abc import ABC, abstractmethod
from typing import Dict

import torch

from .static_tensor import CudaGraphStaticTensors


Tensors = Dict[str, torch.Tensor]


class CudaGraphKernelWrapper(ABC):
    @abstractmethod
    def declare_tensors(self, *args, **kwargs) -> CudaGraphStaticTensors:
        return dict()

    @abstractmethod
    def run_kernel(self, tensors: CudaGraphStaticTensors, *args, **kwargs):
        pass

    @abstractmethod
    def prepare_tensors(self, tensors: CudaGraphStaticTensors, input: Tensors):
        pass

    @abstractmethod
    def prepare_output(self, tensors: CudaGraphStaticTensors) -> Tensors:
        pass

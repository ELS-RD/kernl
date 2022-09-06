import torch
from contextlib import contextmanager


@contextmanager
def set_seed():
    torch.manual_seed(0)
    yield


@set_seed()
def generate_random_data(size: tuple, device: str, torch_dtype: torch.dtype):
    return torch.randn(size, device=device, dtype=torch_dtype, requires_grad=False)

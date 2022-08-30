import torch
from contextlib import contextmanager


@contextmanager
def set_seed():
    torch.manual_seed(0)
    yield

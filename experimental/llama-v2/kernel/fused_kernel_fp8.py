from typing import List, Optional

import triton
import triton.language as tl
import torch

torch.manual_seed(123)


def find_last_one_index(lst: List[int]) -> Optional[int]:
    index = len(lst) - 1
    while index >= 0:
        if lst[index] == 1:
            return index
        else:
            index -= 1
    return None


def f8_to_f16(x, dtypes=tl.float8e5) -> torch.Tensor:
    assert x.dtype == torch.int8, f"torch.int8 expected but got {x.dtype}"
    assert "cuda" in str(x.device), f"CUDA tensors only but got {x.device}"

    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    ret = torch.empty_like(x, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    numel = ret.untyped_storage().size() // ret.element_size()  # manage cases where tensor is not contiguous, like ::2
    kernel[grid](ret, triton.reinterpret(x, dtypes), numel, BLOCK_SIZE=1024)
    return ret


def f16_to_f8(x: torch.Tensor, dtypes=tl.float8e5) -> torch.Tensor:
    assert x.dtype in [torch.float16, torch.float32]
    assert "cuda" in str(x.device), f"CUDA tensors only but got {x.device}"

    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    ret = torch.empty_like(x, dtype=torch.int8)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    numel = x.untyped_storage().size() // x.element_size()  # manage cases where tensor is not contiguous, like ::2
    kernel[grid](triton.reinterpret(ret, dtypes), x, numel, BLOCK_SIZE=1024)
    return ret


class FakeLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, *args, **kwargs):
        raise Exception("should not be used")


def get_model_fp8(parent_module: torch.nn.Module) -> None:
    from llama.model import Attention, RMSNorm

    for name, module in parent_module.named_children():
        if isinstance(module, Attention):
            module.cache_v = f16_to_f8(module.cache_v)
            module.cache_k = f16_to_f8(module.cache_k)

        if isinstance(module, RMSNorm) or isinstance(module, torch.nn.Linear):
            if name not in ["norm", "wo", "w2", "output", "lm_head"]:
                assert module.weight.abs().max() < 431
                weight_fp8 = f16_to_f8(module.weight.data)
                m = FakeLinear(weight=weight_fp8)
                setattr(parent_module, name, m)

        get_model_fp8(module)  # Recursion for nested modules


for _ in range(100):
    a = torch.randn((16, 128), dtype=torch.float16, device="cuda")
    b = f16_to_f8(a, dtypes=tl.float8e4)
    c = f8_to_f16(b, dtypes=tl.float8e4) + 1e-4

    assert (a/c).abs().mean().item()-1 < 1e-1, f"{(a/c).abs().mean()}"

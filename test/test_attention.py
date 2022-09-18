from typing import Callable

import torch
import pytest
from implementations.attention_masked_original import attention_reference, masked_attention_forward_original
from implementations.attention_original import attention_forward_original
from implementations.attention import attention_forward


def original_triton_flash_attention(is_causal: bool, *args, **kwargs):
    if is_causal:
        return masked_attention_forward_original(*args, **kwargs)
    else:
        return attention_forward_original(*args, **kwargs)


implementations = {
    "original": lambda q, k, v, output, sm_scale, is_causal: original_triton_flash_attention(is_causal, q, k, v, output, sm_scale),
    "triton": lambda q, k, v, output, sm_scale, is_causal: attention_forward(q, k, v, output, sm_scale, is_causal),
    "torch": lambda q, k, v, output, sm_scale, is_causal: attention_reference(q, k, v, output, sm_scale, is_causal),
}


@pytest.mark.parametrize("shape", [(bs, seq_l) for bs in [1, 8, 32, 64] for seq_l in [16, 64, 128, 256, 384, 512]],
                         ids=lambda x: f"{x[0]}x{x[1]}")
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "non-causal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("implementation", implementations.keys())
def test_benchmark_masked(benchmark, shape: (int, int), implementation: Callable, dtype: torch.dtype, is_causal: bool):
    batch, seq_length = shape
    if implementation == "original" and (dtype == torch.float32 or seq_length != 512):
        pytest.skip("Original Triton implementation only supports fp16 and seq_length=512")

    torch.manual_seed(0)
    # batch, heads, seq_length, dhead
    mat_shape = (batch, 48, seq_length, 64)
    args = {
        "q": torch.rand(mat_shape, device="cuda") * 2,
        "k": torch.rand(mat_shape, device="cuda") * 2,
        "v": torch.rand(mat_shape, device="cuda") * 2,
        "output": torch.empty(mat_shape, device="cuda"),
        "sm_scale": 0.3,  # Scaling applied before softmax (sqrt(dhead) in Vaswani et al.)
        "is_causal": is_causal,
    }

    expected = attention_reference(**args)
    args = {k: v.to(dtype).clone() if isinstance(v, torch.Tensor) else v for k, v in args.items()}

    func = implementations[implementation]
    value = benchmark(func, **args)

    assert torch.allclose(value.float(), expected, atol=1e-1)


def test_mixed_stride():
    torch.manual_seed(0)
    # Column major
    q = torch.transpose(torch.rand((4, 48, 64, 512), dtype=torch.float16, device="cuda"), -1, -2)
    # Interlaced batch
    k = torch.transpose(torch.rand((48, 4, 512, 64), dtype=torch.float16, device="cuda"), 0, 1)
    v = torch.rand_like(q)
    sm_scale = 0.3

    expected = attention_reference(q=q, k=k, v=v, output=torch.empty_like(q), sm_scale=sm_scale, is_causal=False)
    output = torch.empty_like(q)
    attention_forward(q, k, v, output, sm_scale)
    assert torch.allclose(output, expected, atol=1e-2)

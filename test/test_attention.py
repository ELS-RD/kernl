import gc

import torch
import pytest
from implementations.attention_masked_original import masked_attention_reference, masked_attention_forward_original
from implementations.attention_original import attention_reference, attention_forward_original
from implementations.attention import attention_forward


@pytest.mark.parametrize("batch", [1, 8, 64, 128])
@pytest.mark.parametrize("implementation", ["torch", "triton", "triton_original"])
def test_benchmark_masked(benchmark, batch, implementation):
    torch.manual_seed(0)
    # batch, heads, seqlength, dhead
    q = torch.rand((batch, 48, 512, 64), dtype=torch.float16, device="cuda")
    k = torch.rand((batch, 48, 512, 64), dtype=torch.float16, device="cuda")
    v = torch.rand((batch, 48, 512, 64), dtype=torch.float16, device="cuda")
    # Scaling applied before softmax (sqrt(dhead) in Vaswani et al.)
    sm_scale = 0.3

    expected = masked_attention_reference(q, k, v, sm_scale)
    value = None
    if implementation == "triton_original":
        value = benchmark(masked_attention_forward_original, q, k, v, sm_scale)
    if implementation == "triton":
        output = torch.empty_like(q)
        value = benchmark(attention_forward, q, k, v, output, sm_scale, is_causal=True)
    if implementation == "torch":
        value = benchmark(masked_attention_reference, q, k, v, sm_scale)

    assert torch.allclose(value, expected, atol=1e-2)


@pytest.mark.parametrize("batch", [1, 8, 32, 64, 128])
@pytest.mark.parametrize("implementation", ["torch", "triton_original", "triton"])
def test_benchmark(benchmark, batch, implementation):
    torch.manual_seed(0)
    # batch, heads, seqlength, dhead
    q = torch.rand((batch, 48, 512, 64), dtype=torch.float16, device="cuda")
    k = torch.rand((batch, 48, 512, 64), dtype=torch.float16, device="cuda")
    v = torch.rand((batch, 48, 512, 64), dtype=torch.float16, device="cuda")
    # Scaling applied before softmax (sqrt(dhead) in Vaswani et al.)
    sm_scale = 0.3

    expected = attention_reference(q, k, v, sm_scale)
    value = None
    if implementation == "triton_original":
        value = benchmark(attention_forward_original, q, k, v, sm_scale)
    if implementation == "triton":
        output = torch.empty_like(q)
        value = benchmark(attention_forward, q, k, v, output, sm_scale)
    if implementation == "torch":
        value = benchmark(attention_reference, q, k, v, sm_scale)

    assert torch.allclose(value, expected, atol=1e-2)


def test_mixed_stride(benchmark):
    torch.manual_seed(0)
    # Column major
    q = torch.transpose(torch.rand((4, 48, 64, 512), dtype=torch.float16, device="cuda"), -1, -2)
    # Interlaced batch
    k = torch.transpose(torch.rand((48, 4, 512, 64), dtype=torch.float16, device="cuda"), 0, 1)
    v = torch.rand_like(q)
    sm_scale = 0.3

    expected = attention_reference(q, k, v, sm_scale)
    output = torch.empty_like(q)
    attention_forward(q, k, v, output, sm_scale)
    assert torch.allclose(output, expected, atol=1e-2)

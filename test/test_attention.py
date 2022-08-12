import torch
import pytest
from implementations.attention_masked_original import masked_attention_reference, masked_attention_forward_original
from implementations.attention_original import attention_reference, attention_forward_original
from implementations.attention import attention_forward


@pytest.mark.parametrize("implementation", ["torch", "triton", "triton_original"])
def test_benchmark_masked_causal(benchmark, implementation):
    torch.manual_seed(0)
    q = torch.rand((4, 48, 512, 64), dtype=torch.float16, device="cuda")
    k = torch.rand((4, 48, 512, 64), dtype=torch.float16, device="cuda")
    v = torch.rand((4, 48, 512, 64), dtype=torch.float16, device="cuda")
    sm_scale = 0.3

    expected = masked_attention_reference(q, k, v, sm_scale)
    if implementation == "triton_original":
        value = benchmark(masked_attention_forward_original, q, k, v, sm_scale)
        assert torch.allclose(value, expected, atol=1e-2)
    if implementation == "triton":
        value = benchmark(attention_forward, q, k, v, sm_scale, is_causal=True)
        assert torch.allclose(value, expected, atol=1e-2)
    if implementation == "torch":
        benchmark(masked_attention_reference, q, k, v, sm_scale)

@pytest.mark.parametrize("implementation", ["torch", "triton_original", "triton"])
def test_benchmark(benchmark, implementation):
    torch.manual_seed(0)
    q = torch.rand((4, 48, 512, 64), dtype=torch.float16, device="cuda")
    k = torch.rand((4, 48, 512, 64), dtype=torch.float16, device="cuda")
    v = torch.rand((4, 48, 512, 64), dtype=torch.float16, device="cuda")
    sm_scale = 0.3

    expected = attention_reference(q, k, v, sm_scale)
    if implementation == "triton_original":
        value = benchmark(attention_forward_original, q, k, v, sm_scale)
        assert torch.allclose(value, expected, atol=1e-2)
    if implementation == "triton":
        value = benchmark(attention_forward, q, k, v, sm_scale)
        assert torch.allclose(value, expected, atol=1e-2)
    if implementation == "torch":
        benchmark(attention_reference, q, k, v, sm_scale)
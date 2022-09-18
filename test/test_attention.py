import torch
import pytest
from implementations.attention_masked_original import attention_reference, masked_attention_forward_original
from implementations.attention_original import attention_forward_original
from implementations.attention import attention_forward


@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.parametrize("implementation", ["torch", "triton", "triton_original"])
def test_benchmark_masked(benchmark, batch, implementation):
    torch.manual_seed(0)
    # batch, heads, seqlength, dhead
    q = torch.rand((batch, 48, 512, 64), device="cuda") * 2
    k = torch.rand((batch, 48, 512, 64), device="cuda") * 2
    v = torch.rand((batch, 48, 512, 64), device="cuda") * 2

    q_half = q.half()
    k_half = k.half()
    v_half = v.half()

    # Scaling applied before softmax (sqrt(dhead) in Vaswani et al.)
    sm_scale = 0.3

    expected = attention_reference(q=q, k=k, v=v, output=torch.empty_like(q), sm_scale=sm_scale, is_causal=True)
    output = torch.empty_like(q_half)
    expected_fp16 = attention_reference(q=q_half, k=k_half, v=v_half, output=output, sm_scale=sm_scale, is_causal=True)
    output.zero_()
    value = None
    if implementation == "triton_original":
        value = benchmark(masked_attention_forward_original, q_half, k_half, v_half, output, sm_scale)
    if implementation == "triton":
        value = benchmark(attention_forward, q_half, k_half, v_half, output, sm_scale, is_causal=True)
    if implementation == "torch":
        value = benchmark(attention_reference, q_half, k_half, v_half, output, sm_scale, is_causal=True)

    diff_reference = torch.abs(expected-expected_fp16.float()).max()
    diff_tested = torch.abs(expected-value.float()).max()
    assert diff_reference >= diff_tested, f"{diff_reference=}, {diff_tested=}"


@pytest.mark.parametrize("batch", [1, 8, 32, 64])
@pytest.mark.parametrize("implementation", ["torch", "triton_original", "triton"])
def test_benchmark(benchmark, batch, implementation):
    torch.manual_seed(0)
    # batch, heads, seqlength, dhead
    q = torch.rand((batch, 48, 512, 64), dtype=torch.float16, device="cuda")
    k = torch.rand((batch, 48, 512, 64), dtype=torch.float16, device="cuda")
    v = torch.rand((batch, 48, 512, 64), dtype=torch.float16, device="cuda")
    # Scaling applied before softmax (sqrt(dhead) in Vaswani et al.)
    sm_scale = 0.3

    expected = attention_reference(q=q, k=k, v=v, output=torch.empty_like(q), sm_scale=sm_scale, is_causal=False)
    value = None
    output = torch.empty_like(q)
    if implementation == "triton_original":
        value = benchmark(attention_forward_original, q, k, v, output, sm_scale)
    if implementation == "triton":
        value = benchmark(attention_forward, q, k, v, output, sm_scale)
    if implementation == "torch":
        value = benchmark(attention_reference, q, k, v, output, sm_scale, False)

    assert torch.allclose(value, expected, atol=1e-2)


@pytest.mark.parametrize("seq_length", [16, 64, 128, 256, 512], ids=lambda x: f"seq_length={x}")
@pytest.mark.parametrize("batch", [1, 8, 16, 32, 64], ids=lambda x: f"batch={x}")
def test_optimized(batch, seq_length):
    torch.manual_seed(0)
    # batch, heads, seqlength, dhead
    q = torch.rand((batch, 48, seq_length, 64), dtype=torch.float16, device="cuda")
    k = torch.rand((batch, 48, seq_length, 64), dtype=torch.float16, device="cuda")
    v = torch.rand((batch, 48, seq_length, 64), dtype=torch.float16, device="cuda")
    sm_scale = 0.3

    output = torch.empty_like(q)
    expected = attention_reference(q=q, k=k, v=v, output=torch.empty_like(q), sm_scale=sm_scale, is_causal=False)
    attention_forward(q, k, v, output, sm_scale)
    assert torch.allclose(output, expected, atol=1e-2)


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

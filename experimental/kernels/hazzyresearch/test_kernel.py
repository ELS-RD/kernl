import math

import pytest
import torch

from experimental.kernels.hazzyresearch.flash_attention import flash_attn_func
from experimental.kernels.hazzyresearch.reference import attention_ref as attention_ref_hazyresearch

from kernl.implementations.attention import attention_reference


@pytest.mark.parametrize("dtype", ([torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [40, 48, 64, 128, 80, 88, 96])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
@pytest.mark.parametrize("bias_shape", ([None, "1h1k", "1hqk", "b11k", "b1qk"]))
def test_forward(seqlen_q, seqlen_k, d, causal, dtype, bias_shape):
    if seqlen_q >= 2048 and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    nheads = 4
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    if bias_shape == "1h1k":
        bias = torch.randn(1, nheads, 1, seqlen_k, dtype=torch.float, device=device)
    elif bias_shape == "1hqk":
        bias = torch.randn(1, nheads, seqlen_q, seqlen_k, dtype=torch.float, device=device)
    elif bias_shape == "b11k":
        bias = torch.randn(batch_size, 1, 1, seqlen_k, dtype=torch.float, device=device)
    elif bias_shape == "b1qk":
        bias = torch.randn(batch_size, 1, seqlen_q, seqlen_k, dtype=torch.float, device=device)
    else:
        bias = None

    # q, k, v = [x.detach().requires_grad_() for x in [q, k, v]]
    output = flash_attn_func(q, k, v, bias, causal)

    output_ref, attn_ref = attention_ref_hazyresearch(q, k, v, bias=bias, causal=causal)
    output_pt, attn_pt = attention_ref_hazyresearch(q, k, v, bias=bias, causal=causal, upcast=False, reorder_ops=True)
    print(f"Output max diff: {(output - output_ref).abs().max().item()}")
    print(f"Output mean diff: {(output - output_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(output_pt - output_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(output_pt - output_ref).abs().mean().item()}")

    assert (output - output_ref).abs().max().item() <= 2 * (output_pt - output_ref).abs().max().item()


def test_compare_ref():
    torch.manual_seed(123)
    device = "cuda"
    dtype = torch.float32
    batch_size, seqlen, nheads, d = 32, 128, 4, 64
    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    q2 = q.transpose(1, 2)
    k2 = k.transpose(1, 2)
    v2 = v.transpose(1, 2)

    output_ref = torch.zeros_like(q)
    sm_scale = 1 / math.sqrt(d)
    attention_reference(
        q=q2,
        k=k2,
        v=v2,
        output=output_ref,
        sm_scale=sm_scale,
        is_causal=False,
        attention_mask=None,
    )
    output_hazy, _ = attention_ref_hazyresearch(
        q=q,
        k=k,
        v=v,
        bias=None,
        causal=False,
        query_padding_mask=None,
        key_padding_mask=None,
        dropout_p=0.0,
        dropout_mask=None,
        upcast=False,
        reorder_ops=False,
    )
    output_hazy = output_hazy.transpose(1, 2)
    assert torch.allclose(output_ref, output_hazy, atol=1e-1)

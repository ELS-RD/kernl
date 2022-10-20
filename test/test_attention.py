#  Copyright 2022 Lefebvre Sarrut
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from typing import Callable

import pytest
import torch

from conftest import check_all_close, set_seed

from kernl.implementations.attention import attention_forward, attention_reference
from kernl.implementations.attention_masked_original import masked_attention_forward_original
from kernl.implementations.attention_original import attention_forward_original


def original_triton_flash_attention(is_causal: bool, *args, **kwargs):
    if is_causal:
        return masked_attention_forward_original(*args, **kwargs)
    else:
        return attention_forward_original(*args, **kwargs)


implementations = {
    "original": lambda q, k, v, output, sm_scale, is_causal, attention_mask: original_triton_flash_attention(
        is_causal, q, k, v, output, sm_scale
    ),
    "triton": lambda q, k, v, output, sm_scale, is_causal, attention_mask: attention_forward(
        q, k, v, output, sm_scale, is_causal, attention_mask
    ),
    "torch": lambda q, k, v, output, sm_scale, is_causal, attention_mask: attention_reference(
        q, k, v, output, sm_scale, is_causal, attention_mask
    ),
}


def generate_broadcast_mask(batch, seq_length, dtype=torch.float32):
    attention_mask = (
        torch.randint(1, seq_length, (batch,), device="cuda")[:, None]
        > torch.arange(0, seq_length, device="cuda")[None, :]
    )
    attention_mask = attention_mask.to(dtype)
    attention_mask = torch.reshape(attention_mask, (batch, 1, 1, seq_length))
    attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
    return attention_mask


def generate_bias_mask(batch, seq_length, dtype=torch.float32):
    return torch.rand((batch, 48, seq_length, seq_length), dtype=dtype, device="cuda")


def generate_none_mask(batch, seq_length, dtype=torch.float32):
    return None


@set_seed()
@pytest.mark.parametrize(
    "shape",
    [(bs, seq_l) for bs in [1, 8, 32, 64] for seq_l in [16, 33, 64, 128, 256, 384, 512]] + [(32, 32)],
    ids=lambda x: f"{x[0]}x{x[1]}",
)
# fp32 not yet possible because of a bug in triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "non-causal"])
@pytest.mark.parametrize(
    "mask_fn",
    [generate_bias_mask, generate_broadcast_mask, generate_none_mask],
    ids=["bias-mask", "broadcast-mask", "no-mask"],
)
@pytest.mark.parametrize("implementation", implementations.keys())
def test_benchmark_masked(
    benchmark, shape: (int, int), implementation: Callable, mask_fn: Callable, dtype: torch.dtype, is_causal: bool
):
    batch, seq_length = shape
    if implementation == "original" and (dtype == torch.bfloat16 or seq_length != 512):
        pytest.skip("Original Triton implementation only supports fp16 and seq_length=512")
    elif implementation == "original" and mask_fn != generate_none_mask:
        pytest.skip("Original Triton implementation doesn't support masks")

    # batch, heads, seq_length, dhead
    mat_shape = (batch, 48, seq_length, 64)
    args = {
        "q": torch.rand(mat_shape, device="cuda"),
        "k": torch.rand(mat_shape, device="cuda"),
        "v": torch.rand(mat_shape, device="cuda"),
        "output": torch.empty(mat_shape, device="cuda"),
        "sm_scale": 0.3,  # Scaling applied before softmax (sqrt(dhead) in Vaswani et al.)
        "is_causal": is_causal,
        "attention_mask": mask_fn(batch, seq_length),
    }

    expected = attention_reference(**args)
    cast_args = {k: v.to(dtype).clone() if isinstance(v, torch.Tensor) else v for k, v in args.items()}

    func = implementations[implementation]
    value = benchmark(func, **cast_args)

    check_all_close(a=value.float(), b=expected, atol=1e-1)


@set_seed()
def test_mixed_stride():
    # Column major
    q = torch.rand((4, 48, 64, 512), dtype=torch.float16, device="cuda").transpose(-1, -2)
    # Interlaced batch
    k = torch.rand((48, 4, 512, 64), dtype=torch.float16, device="cuda").transpose(0, 1)
    v = torch.rand_like(q)
    mask = torch.rand((48, 4, 512, 512), dtype=torch.float16, device="cuda").transpose(0, 1).transpose(-1, -2)
    sm_scale = 0.3

    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not mask.is_contiguous()

    expected = attention_reference(
        q=q, k=k, v=v, output=torch.empty_like(q), sm_scale=sm_scale, is_causal=False, attention_mask=mask
    )
    output = torch.empty_like(q)
    attention_forward(q, k, v, output, sm_scale, attention_mask=mask)
    check_all_close(a=output, b=expected, atol=1e-2)


@set_seed()
def test_complex():
    q = torch.rand((1, 8, 1, 64), dtype=torch.float16, device="cuda")
    k = torch.rand((1, 8, 24, 64), dtype=torch.float16, device="cuda")
    v = torch.rand_like(k)
    mask = torch.rand((1, 8, 1, 24), dtype=torch.float16, device="cuda")
    sm_scale = 1.0

    expected = attention_reference(
        q=q, k=k, v=v, output=torch.empty_like(q), sm_scale=sm_scale, is_causal=False, attention_mask=mask
    )
    output = torch.empty_like(q)
    attention_forward(q, k, v, output, sm_scale, attention_mask=mask)
    check_all_close(a=output, b=expected, atol=1e-2)

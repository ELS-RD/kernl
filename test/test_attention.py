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

from conftest import assert_all_close, set_seed
from src.kernl.implementations.attention_skinny import skinny_attention_forward
from src.kernl.implementations.cuda_graph import cuda_graphs_wrapper

from kernl.implementations.attention import attention_forward, attention_reference, closest_power_of_2
from kernl.implementations.attention_vec_mat import attention_vec_mat_mul_forward


implementations = {
    "triton": lambda q, k, v, output, sm_scale, is_causal, attention_mask: attention_forward(
        q, k, v, output, sm_scale, is_causal, attention_mask
    ),
    "torch": lambda q, k, v, output, sm_scale, is_causal, attention_mask: attention_reference(
        q, k, v, output, sm_scale, is_causal, attention_mask
    ),
}


def generate_broadcast_mask(
    batch: int, heads: int, seq_length: int, dhead: int, dtype: torch.Tensor = torch.float32
) -> torch.Tensor:
    attention_mask = (
        torch.randint(1, seq_length, (batch,), device="cuda")[:, None]
        > torch.arange(0, seq_length, device="cuda")[None, :]
    )
    attention_mask = attention_mask.to(dtype)
    attention_mask = torch.reshape(attention_mask, (batch, 1, 1, seq_length))
    attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
    return attention_mask


def generate_bias_mask(
    batch: int, heads: int, seq_length: int, dhead: int, dtype: torch.Tensor = torch.float32
) -> torch.Tensor:
    return torch.rand((batch, heads, seq_length, seq_length), dtype=dtype, device="cuda")


def generate_none_mask(*_) -> None:
    return None


@set_seed()
@pytest.mark.parametrize(
    "shape",
    [(bs, 48, seq_l, 64) for bs in [1, 8, 32, 64] for seq_l in [8, 16, 32, 33, 64, 128, 256, 257, 384, 512]]
    + [(8, 1, 1500, 64)],
    ids=lambda x: f"shape(batch,heads,seq_len,dhead)={x[0]}x{x[1]}x{x[2]}x{x[3]}",
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
    benchmark,
    shape: (int, int, int, int),
    implementation: Callable,
    mask_fn: Callable,
    dtype: torch.dtype,
    is_causal: bool,
):
    batch, heads, seq_length, dhead = shape

    # bf16 reduced precision is sensitive to this value
    tensor_max_val = 1.0 if dtype == torch.bfloat16 else 2.0

    args = {
        "q": torch.rand(shape, device="cuda") * tensor_max_val,
        "k": torch.rand(shape, device="cuda") * tensor_max_val,
        "v": torch.rand(shape, device="cuda") * tensor_max_val,
        "output": torch.empty(shape, device="cuda"),
        "sm_scale": 0.3,  # Scaling applied before softmax (sqrt(dhead) in Vaswani et al.)
        "is_causal": is_causal,
        "attention_mask": mask_fn(batch, heads, seq_length, dhead),
    }

    expected = attention_reference(**args)
    cast_args = {k: v.to(dtype).clone() if isinstance(v, torch.Tensor) else v for k, v in args.items()}

    func = implementations[implementation]
    value = benchmark(func, **cast_args)

    assert_all_close(a=value.float(), b=expected, atol=1e-1)

    for _ in range(10):
        o = func(**cast_args)
        assert_all_close(value, o, atol=1e-2)


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
    assert_all_close(a=output, b=expected, atol=1e-2)


@set_seed()
def test_cross_attention():
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
    assert_all_close(a=output, b=expected, atol=1e-2)


def test_closest_power_of_2():
    min_range = 16
    max_range = 128
    assert closest_power_of_2(1, min_range=min_range, max_range=max_range) == [8, 16, 32]
    assert closest_power_of_2(20, min_range=min_range, max_range=max_range) == [16, 32]
    assert closest_power_of_2(257, min_range=min_range, max_range=max_range) == [64, 128, 256]
    assert closest_power_of_2(20, min_range=min_range, max_range=max_range) == [16, 32]
    min_range = 4
    max_range = 128
    assert closest_power_of_2(1, min_range=min_range, max_range=max_range) == [2, 4, 8]


@set_seed()
@pytest.mark.parametrize("implementation", ["torch", "split-k", "vec-mat-mul", "classic"])
def test_benchmark_cross_attention_split(benchmark, implementation):
    q = torch.rand((5, 20, 1, 64), dtype=torch.float16, device="cuda")
    k = torch.rand((5, 20, 1500, 64), dtype=torch.float16, device="cuda")
    v = torch.rand_like(k)
    mask = None
    sm_scale = 1.0
    is_causal = False
    p = torch.cuda.graph_pool_handle()
    expected = attention_reference(
        q=q, k=k, v=v, output=torch.empty_like(q), sm_scale=sm_scale, is_causal=is_causal, attention_mask=mask
    )
    output = torch.empty_like(q)
    if implementation == "torch":

        def fn(q, k, v):
            return attention_reference(
                q=q, k=k, v=v, output=output, sm_scale=sm_scale, is_causal=is_causal, attention_mask=mask
            )

        r = cuda_graphs_wrapper(fn, [q, k, v], pool=p)
        _ = r(q, k, v)[0]
        result = benchmark(r)[0]
    elif implementation == "split-k":

        def fn(q, k, v):
            return skinny_attention_forward(
                q, k, v, output, sm_scale=sm_scale, attention_mask=mask, is_causal=is_causal
            )

        r = cuda_graphs_wrapper(fn, [q, k, v], pool=p)
        _ = r(q, k, v)[0]
        result = benchmark(r)[0]
    elif implementation == "vec-mat-mul":
        if is_causal or mask is not None:
            pytest.skip("causal or masked attention not supported")

        if v.stride()[-1] == 1:  # is row major
            # change layout to col major
            v.set_(source=v.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2))

        def fn(q, k, v):
            return attention_vec_mat_mul_forward(
                q=q, k=k, v=v, output=output, sm_scale=sm_scale, attention_mask=mask, is_causal=is_causal
            )

        r = cuda_graphs_wrapper(fn, [q, k, v], pool=p)
        _ = r(q, k, v)[0]
        result = benchmark(r)[0]

    elif implementation == "classic":

        def fn(q, k, v):
            return attention_forward(
                q, k, v, output=output, sm_scale=sm_scale, attention_mask=mask, is_causal=is_causal
            )

        r = cuda_graphs_wrapper(fn, [q, k, v], pool=p)
        _ = r(q, k, v)[0]
        result = benchmark(r)[0]
    else:
        raise ValueError(f"Unknown implementation {implementation}")

    assert_all_close(a=expected, b=result, atol=1e-1)

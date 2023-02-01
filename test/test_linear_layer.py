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

from typing import Callable, Tuple

import pytest
import torch

from conftest import assert_all_close, set_seed

from kernl.implementations.linear_layer import linear_layer
from kernl.optimizer.cuda_graph import cuda_graphs_wrapper


def get_pytorch_activation(activation: str) -> Callable:
    if activation == "gelu":
        return torch.nn.functional.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "relu":
        return torch.relu
    elif activation == "":
        return lambda x: x
    else:
        raise ValueError(f"Unknown activation: {activation}")


implementations = {
    "pytorch": lambda weight, bias, activation: lambda x: get_pytorch_activation(activation)(
        torch.nn.functional.linear(x, weight, bias)
    ),
    "triton": lambda weight, bias, activation: lambda x: linear_layer(x, weight, bias, activation),
}


@set_seed()
@pytest.mark.parametrize("contiguous", [True, False], ids=["contiguous", "non-contiguous"])
@pytest.mark.parametrize("bias", [True, False], ids=["with_bias", "no_bias"])
@pytest.mark.parametrize("activation", ["", "tanh", "gelu", "relu"], ids=["no_activation", "tanh", "gelu", "relu"])
@pytest.mark.parametrize(
    "shape",
    [(1, 8, 8, 8)] + [(bs, M, 768, 768) for bs in [1, 16] for M in [8, 16, 128, 256, 512]],
    ids=lambda s: "x".join(map(str, s)),
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("cuda_graphs", [False, True], ids=["no_cuda_graphs", "cuda_graphs"])
@pytest.mark.parametrize("implementation", ["triton", "pytorch"])
def test_benchmark(
    benchmark,
    implementation: str,
    cuda_graphs: bool,
    shape: Tuple[int, int, int, int],
    dtype: torch.dtype,
    bias: bool,
    activation: str,
    contiguous: bool,
):
    batch, M, N, K = shape

    # order of dimensions is wrong so we force contiguous call
    x = torch.randn((batch, K, M), device="cuda", dtype=torch.float32, requires_grad=False)
    x = x.mT
    if contiguous:
        x = x.contiguous()
    else:
        assert not x.is_contiguous()
    factory_kwargs = {"device": "cuda", "dtype": torch.float32, "requires_grad": False}
    layer_weight = torch.randn((N, K), **factory_kwargs)
    layer_bias = torch.randn((K,), **factory_kwargs) if bias else None
    pytorch_layer_activation = get_pytorch_activation(activation)
    expected = pytorch_layer_activation(torch.nn.functional.linear(x, layer_weight, layer_bias))

    # tensors casting
    layer_weight = layer_weight.to(dtype=dtype)
    if layer_bias is not None:
        layer_bias = layer_bias.to(dtype=dtype)
    x = x.to(dtype=dtype)

    fn = implementations[implementation](layer_weight, layer_bias, activation)
    if cuda_graphs:
        run = cuda_graphs_wrapper(model=fn, inputs=[x])
        # CUDA graphs wraps output in a tuple
        fn = lambda tensor: run(tensor)[0]  # noqa: E731

    value = benchmark(fn, x)

    assert_all_close(expected, value.float(), rtol=1e-1, atol=1e-1)

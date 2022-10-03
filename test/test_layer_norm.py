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

import pytest
import torch

from conftest import set_seed

from nucle.implementations.cuda_graph import cuda_graphs_wrapper
from nucle.implementations.layer_norm import (
    _layer_norm_fwd_fused_multi_pass,
    _layer_norm_fwd_fused_single_pass,
    layer_norm,
    layer_norm_xformers,
    pytorch_naive_layernorm,
)


implementations = {
    "pytorch": lambda weight, bias, eps: lambda x: torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps),
    "triton_original": lambda weight, bias, eps: lambda x: layer_norm(
        x, weight, bias, eps, _layer_norm_fwd_fused_multi_pass
    ),
    "triton_improved": lambda weight, bias, eps: lambda x: layer_norm(
        x, weight, bias, eps, _layer_norm_fwd_fused_single_pass
    ),
    "triton_xformer": lambda weight, bias, eps: lambda x: layer_norm(x, weight, bias, eps, layer_norm_xformers),
    "pytorch_naive": lambda weight, bias, eps: lambda x: pytorch_naive_layernorm(x, weight, bias, eps),
}


@set_seed()
@pytest.mark.parametrize("shape", [128, 512, 1024, 2048, 4096], ids=lambda x: f"shape={x}x{x}")
@pytest.mark.parametrize("cuda_graphs", [True, False], ids=["cuda_graphs", "no_cuda_graphs"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("implementation", implementations.keys())
def test_benchmark_layer_norm(benchmark, shape: int, dtype, cuda_graphs: bool, implementation: str):
    M = N = shape
    eps = 1e-5
    factory_kwargs = {"device": "cuda", "dtype": torch.float32, "requires_grad": False}
    layer_weight = torch.rand((N,), **factory_kwargs)
    layer_bias = torch.randn_like(layer_weight)
    x = -20 + 0.5 * torch.randn((M, N), **factory_kwargs)
    expected = torch.nn.functional.layer_norm(x, layer_weight.shape, layer_weight, layer_bias, eps)

    # tensors casting
    layer_weight = layer_weight.to(dtype)
    layer_bias = layer_bias.to(dtype)
    x = x.to(dtype)

    fn = implementations[implementation](layer_weight, layer_bias, eps)
    if cuda_graphs:
        run = cuda_graphs_wrapper(model=fn, inputs=[x], copy_outputs=False)
        # cuda graphs wraps output in a tuple
        fn = lambda tensor: run(tensor)[0]  # noqa: E731

    value = benchmark(fn, x)

    assert torch.allclose(value.float(), expected, atol=1e-1)

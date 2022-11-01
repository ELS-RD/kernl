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

from conftest import check_all_close, set_seed

from kernl.implementations.cuda_graph import cuda_graphs_wrapper
from kernl.implementations.layer_norm import (
    _layer_norm_fwd_fused_multi_pass,
    _layer_norm_fwd_fused_single_pass,
    layer_norm,
    layer_norm_xformers,
    pytorch_naive_layernorm,
    pytorch_naive_rmsnorm,
)


implementations_layer_norm = {
    "pytorch": lambda weight, bias, eps: lambda x: torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps),
    # "triton_original": lambda weight, bias, eps: lambda x: layer_norm(
    #     x, weight, bias, eps, _layer_norm_fwd_fused_multi_pass, use_rms_norm=False
    # ),
    # "triton_improved": lambda weight, bias, eps: lambda x: layer_norm(
    #     x, weight, bias, eps, _layer_norm_fwd_fused_single_pass, use_rms_norm=False
    # ),
    # "triton_xformer": lambda weight, bias, eps: lambda x: layer_norm(
    #     x, weight, bias, eps, layer_norm_xformers, use_rms_norm=False
    # ),
    # "pytorch_naive": lambda weight, bias, eps: lambda x: pytorch_naive_layernorm(x, weight, bias, eps),
}
torch.autograd.set_detect_anomaly(True)

@set_seed()
@pytest.mark.parametrize("shape", [128, 256, 512, 1024], ids=lambda x: f"shape={x}x{x}")
@pytest.mark.parametrize("cuda_graphs", [True])
@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
@pytest.mark.parametrize("implementation", implementations_layer_norm.keys())
def test_benchmark_layer_norm(benchmark, shape: int, dtype, cuda_graphs: bool, implementation: str):
    M = N = shape
    eps = 1e-5
    layer_weight = torch.rand((N,), requires_grad=True, device="cuda", dtype=torch.float32)
    layer_bias = torch.randn_like(layer_weight, requires_grad=True)

    # not marked as requires_grad to avoid the gradient computation
    x = -20 + 0.5 * torch.randn((M, N), device="cuda", dtype=torch.float32)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    expected = torch.nn.functional.layer_norm(x, layer_weight.shape, layer_weight, layer_bias, eps)

    expected.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, layer_weight, layer_bias]]
    x.grad, layer_weight.grad, layer_bias.grad = None, None, None

    # tensors casting
    layer_weight = layer_weight.to(dtype)
    layer_bias = layer_bias.to(dtype)
    x = x.to(dtype)
    dy = dy.to(dtype)
    x.retain_grad()
    layer_weight.retain_grad()
    layer_bias.retain_grad()

    fn = implementations_layer_norm[implementation](layer_weight, layer_bias, eps)
    if cuda_graphs:
        run = cuda_graphs_wrapper(model=fn, inputs=[x], copy_outputs=False)
        # CUDA graphs wraps output in a tuple
        fn = lambda tensor: run(tensor)[0]  # noqa: E731

    value = benchmark(fn, x)
    check_all_close(value.float(), expected, atol=1e-1)

    value.backward(dy, retain_graph=True)

    dx_fn, dw_fn, db_fn = [_.grad.clone() for _ in [x, layer_weight, layer_bias]]
    check_all_close(dx_ref.float(), dx_fn.float(), atol=1e-1)
    check_all_close(dw_ref.float(), dw_fn.float(), atol=1e-1)
    check_all_close(db_ref.float(), db_fn.float(), atol=1e-1)


implementations_rms_norm = {
    "pytorch": lambda weight, eps: lambda x: pytorch_naive_rmsnorm(x, weight, eps),
    "triton_improved": lambda weight, eps: lambda x: layer_norm(
        x, weight, None, eps, _layer_norm_fwd_fused_single_pass, use_rms_norm=True
    ),
}


# @pytest.mark.parametrize("cuda_graphs", [True, False], ids=["cuda_graphs", "no_cuda_graphs"])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
# @pytest.mark.parametrize("shape", [128, 512, 1024, 2048, 4096], ids=lambda x: f"shape={x}x{x}")
# @pytest.mark.parametrize("implementation", implementations_rms_norm.keys())
# def test_benchmark_rms_norm(benchmark, shape: int, dtype, cuda_graphs: bool, implementation: str):
#     M = N = shape
#     eps = 1e-5
#     factory_kwargs = {"device": "cuda", "dtype": torch.float32, "requires_grad": False}
#     layer_weight = torch.rand((N,), **factory_kwargs)
#     x = -20 + 0.5 * torch.randn((M, N), **factory_kwargs)
#     expected = pytorch_naive_rmsnorm(x, layer_weight, eps)
#
#     # tensors casting
#     layer_weight = layer_weight.to(dtype)
#     x = x.to(dtype)
#
#     fn = implementations_rms_norm[implementation](layer_weight, eps)
#     if cuda_graphs:
#         run = cuda_graphs_wrapper(model=fn, inputs=[x], copy_outputs=False)
#         # CUDA graphs wraps output in a tuple
#         fn = lambda tensor: run(tensor)[0]  # noqa: E731
#
#     value = benchmark(fn, x)
#     check_all_close(value.float(), expected, atol=1e-1)

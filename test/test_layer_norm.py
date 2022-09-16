from typing import Callable

import torch
import pytest

from implementations.cuda_graph import cuda_graphs_wrapper
from implementations.layer_norm import layer_norm_forward, _layer_norm_fwd_fused_single_pass, \
    _layer_norm_fwd_fused_multi_pass, layer_norm_xformers, pytorch_naive_layernorm

implementations: dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]] = {
    "pytorch": lambda x, weight, bias, eps: torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps),
    "triton_original": lambda x, weight, bias, eps: layer_norm_forward(x, weight, bias, eps, _layer_norm_fwd_fused_multi_pass),
    "triton_improved": lambda x, weight, bias, eps: layer_norm_forward(x, weight, bias, eps, _layer_norm_fwd_fused_single_pass),
    "triton_xformer": lambda x, weight, bias, eps: layer_norm_forward(x, weight, bias, eps, layer_norm_xformers),
    "pytorch_naive": lambda x, weight, bias, eps: pytorch_naive_layernorm(x, weight, bias, eps),
}


@pytest.mark.parametrize("shape", [128, 512, 1024, 2048, 4096, 8192], ids=lambda x: f"shape={x}x{x}")
@pytest.mark.parametrize("cuda_graphs", [True, False], ids=["cuda_graphs", "no_cuda_graphs"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@pytest.mark.parametrize("implementation", implementations.keys())
def test_benchmark_layer_norm(benchmark, shape: int, dtype, cuda_graphs: bool, implementation: str):
    torch.manual_seed(0)
    M = N = shape
    eps = 1e-6

    weight = torch.rand((N,), dtype=dtype, device='cuda', requires_grad=False)
    bias = torch.randn_like(weight, dtype=dtype, device='cuda', requires_grad=False)
    x = -20 + 0.5 * torch.randn((M, N), dtype=dtype, device='cuda', requires_grad=False)
    expected = torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps)

    inference = implementations[implementation]
    if cuda_graphs:
        def func_wrapper(tensor):
            return inference(tensor, weight, bias, eps)

        run = cuda_graphs_wrapper(model=func_wrapper, inputs=[x], copy_outputs=False)

        def inference(x, *args, **kwargs):
            return run(x)[0]

    value = benchmark(inference, x, weight, bias, eps)

    assert torch.allclose(value, expected, atol=1e-1)

import torch
import pytest

from implementations.cuda_graph import cuda_graphs_wrapper
from implementations.layer_norm import layer_norm, _layer_norm_fwd_fused_single_pass, \
    _layer_norm_fwd_fused_multi_pass, layer_norm_xformers, pytorch_naive_layernorm
from conftest import set_seed

implementations = {
    "pytorch": lambda weight, bias, eps: lambda x: torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps),
    "triton_original": lambda weight, bias, eps: lambda x: layer_norm(x, weight, bias, eps, _layer_norm_fwd_fused_multi_pass),
    "triton_improved": lambda weight, bias, eps: lambda x: layer_norm(x, weight, bias, eps, _layer_norm_fwd_fused_single_pass),
    "triton_xformer": lambda weight, bias, eps: lambda x: layer_norm(x, weight, bias, eps, layer_norm_xformers),
    "pytorch_naive": lambda weight, bias, eps: lambda x: pytorch_naive_layernorm(x, weight, bias, eps),
}


@set_seed()
@pytest.mark.parametrize("shape", [128, 512, 1024, 2048, 4096, 8192], ids=lambda x: f"shape={x}x{x}")
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
        fn = lambda tensor: run(tensor)[0]  # cuda graphs wraps output in a tuple

    value = benchmark(fn, x)

    assert torch.allclose(value.float(), expected, atol=1e-1)

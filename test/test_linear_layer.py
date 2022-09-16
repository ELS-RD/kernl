import dataclasses

import torch
import pytest

from implementations.cuda_graph import cuda_graphs_wrapper
from implementations.linear_layer import linear_layer


@dataclasses.dataclass
class Shape:
    bs: int
    M: int
    N: int
    K: int

    @property
    def __dict__(self):
        return dataclasses.asdict(self)


@pytest.mark.parametrize("contiguous", [True, False], ids=["contiguous", "non-contiguous"])
@pytest.mark.parametrize("bias", [True, False], ids=["with_bias", "no_bias"])
@pytest.mark.parametrize("activation", ["", "tanh", "gelu"], ids=["no_activation", "tanh", "gelu"])
@pytest.mark.parametrize("shape", [Shape(bs=1, M=8, N=8, K=8)] +
                         [Shape(bs=bs, M=M, N=768, K=768) for bs in [1, 16] for M in [8, 16, 128, 256, 512]],
                         ids=lambda x: f"{x.bs}x{x.M}x{x.N}x{x.K}")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("implementation", ["triton", "triton_cuda_graph", "pytorch", "pytorch_cuda_graph"])
def test_benchmark(benchmark, shape: Shape, dtype: torch.dtype, bias: bool, activation: str, contiguous: bool, implementation: str):
    torch.manual_seed(0)
    batch, M, N, K = dataclasses.astuple(shape)

    # order of dimensions is wrong so we force contiguous call

    a = torch.randn((batch, K, M), device='cuda', dtype=dtype, requires_grad=False)
    a = a.mT
    if contiguous:
        a = a.contiguous()
    else:
        assert not a.is_contiguous()

    layer_weight = torch.randn((N, K), device='cuda', dtype=dtype, requires_grad=False)

    if activation == "gelu":
        activation_fn = torch.nn.functional.gelu
    elif activation == "tanh":
        activation_fn = torch.tanh
    elif activation == "":
        activation_fn = lambda x: x
    else:
        raise ValueError(f"Unknown activation: {activation}")

    torch_linear_layer = torch.nn.Linear(K, N, bias=bias, device="cuda", dtype=dtype)
    torch_linear_layer.weight.data = layer_weight

    def torch_linear_activation(x):
        return activation_fn(torch_linear_layer(x))

    expected = torch_linear_activation(a)

    cuda_graph_pool = torch.cuda.graph_pool_handle()

    if implementation == "pytorch":
        value = benchmark(torch_linear_activation, a)
    elif implementation == "pytorch_cuda_graph":
        run = cuda_graphs_wrapper(model=torch_linear_activation, inputs=[a], pool=cuda_graph_pool)
        (value,) = benchmark(run, a)
    elif implementation == "triton":
        value, _ = benchmark(linear_layer, x=a, weight=layer_weight, bias=torch_linear_layer.bias,
                             activation=activation)
    elif implementation == "triton_cuda_graph":
        def wrapper(x):
            return linear_layer(x=x, weight=layer_weight, bias=torch_linear_layer.bias, activation=activation)

        run = cuda_graphs_wrapper(model=wrapper, inputs=[a], pool=cuda_graph_pool, copy_outputs=False)
        value, _ = benchmark(run, a)
    else:
        raise ValueError(f"Unknown implementation {implementation}")

    assert torch.allclose(value, expected, rtol=1e-1, atol=1e-1), f"max diff: {torch.abs(value - expected).max()}"

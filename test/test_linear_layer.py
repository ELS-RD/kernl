import dataclasses
from functools import lru_cache
from typing import Callable

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


@lru_cache
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
    "pytorch": lambda weight, bias, activation: lambda x: get_pytorch_activation(activation)(torch.nn.functional.linear(x, weight, bias)),
    "triton": lambda weight, bias, activation: lambda x: linear_layer(x, weight, bias, activation),
}


@pytest.mark.parametrize("contiguous", [True, False], ids=["contiguous", "non-contiguous"])
@pytest.mark.parametrize("bias", [True, False], ids=["with_bias", "no_bias"])
@pytest.mark.parametrize("activation", ["", "tanh", "gelu", "relu"], ids=["no_activation", "tanh", "gelu", "relu"])
@pytest.mark.parametrize("shape", [Shape(bs=1, M=8, N=8, K=8)] +
                         [Shape(bs=bs, M=M, N=768, K=768) for bs in [1, 16] for M in [8, 16, 128, 256, 512]],
                         ids=lambda x: f"{x.bs}x{x.M}x{x.N}x{x.K}")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("cuda_graphs", [False, True], ids=["no_cuda_graphs", "cuda_graphs"])
@pytest.mark.parametrize("implementation", ["triton", "pytorch"])
def test_benchmark(benchmark, implementation: str, cuda_graphs: bool, shape: Shape, dtype: torch.dtype, bias: bool,
                   activation: str, contiguous: bool):
    torch.manual_seed(0)
    batch, M, N, K = dataclasses.astuple(shape)

    # order of dimensions is wrong so we force contiguous call
    x = torch.randn((batch, K, M), device='cuda', dtype=torch.float32, requires_grad=False)
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

    cuda_graph_pool = torch.cuda.graph_pool_handle()

    if cuda_graphs:
        run = cuda_graphs_wrapper(model=fn, inputs=[x], pool=cuda_graph_pool)
        (value,) = benchmark(run, x)
    else:
        value = benchmark(fn, x)

    assert torch.allclose(expected, value.float(), rtol=1e-1,
                          atol=1e-1), f"max diff: {torch.abs(value.float() - expected).max()}"

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
@pytest.mark.parametrize("activation", ["", "tanh", "gelu", "relu"], ids=["no_activation", "tanh", "gelu", "relu"])
@pytest.mark.parametrize("shape", [Shape(bs=1, M=8, N=8, K=8)] +
                         [Shape(bs=bs, M=M, N=768, K=768) for bs in [1, 16] for M in [8, 16, 128, 256, 512]],
                         ids=lambda x: f"{x.bs}x{x.M}x{x.N}x{x.K}")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("cuda_graphs", [False, True], ids=["no_cuda_graphs", "cuda_graphs"])
@pytest.mark.parametrize("implementation", ["triton", "pytorch"])
def test_benchmark(benchmark, implementation: str, cuda_graphs: bool, shape: Shape, dtype: torch.dtype, bias: bool, activation: str, contiguous: bool):
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
    elif activation == "relu":
        activation_fn = torch.relu
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
        fn = torch_linear_activation
    elif implementation == "triton":
        def fn(x):
            return linear_layer(x, layer_weight, torch_linear_layer.bias, activation)
    else:
        raise ValueError(f"Unknown implementation {implementation}")

    if cuda_graphs:
        run = cuda_graphs_wrapper(model=fn, inputs=[a], pool=cuda_graph_pool)
        (value,) = benchmark(run, a)
    else:
        value = benchmark(fn, a)

    assert torch.allclose(value, expected, rtol=1e-1, atol=1e-1), f"max diff: {torch.abs(value - expected).max()}"

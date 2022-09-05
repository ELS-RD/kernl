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

    def x_shape(self):
        return self.bs, self.K, self.M

    def weight_shape(self):
        return self.K, self.N


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("shape", [Shape(bs=1, M=8, N=768, K=768),
                                   Shape(bs=1, M=16, N=768, K=768),
                                   Shape(bs=1, M=128, N=768, K=768),
                                   Shape(bs=1, M=256, N=768, K=768),
                                   Shape(bs=1, M=512, N=768, K=768),
                                   Shape(bs=16, M=8, N=768, K=768),
                                   Shape(bs=16, M=16, N=768, K=768),
                                   Shape(bs=16, M=128, N=768, K=768),
                                   Shape(bs=16, M=256, N=768, K=768),
                                   Shape(bs=16, M=512, N=768, K=768),
                                   ])
@pytest.mark.parametrize("implementation", ["triton", "triton_cuda_graph", "pytorch", "pytorch_cuda_graph"])
def test_benchmark(benchmark, shape: Shape, contiguous, implementation):
    torch.manual_seed(0)
    batch, M, N, K = dataclasses.astuple(shape)

    # order of dimensions is wrong so we force contiguous call
    a = torch.randn((batch, K, M), device='cuda', dtype=torch.float16, requires_grad=False)
    a = a.mT
    if contiguous:
        a = a.contiguous()
    else:
        assert not a.is_contiguous()
    assert not torch.any(a.isnan()).item()
    layer_weight = torch.randn((N, K), device='cuda', dtype=torch.float16, requires_grad=False)

    torch_linear_layer = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
    torch_linear_layer.weight.data = layer_weight
    expected = torch_linear_layer(a)
    assert not torch.any(expected.isnan()).item()
    cuda_graph_pool = torch.cuda.graph_pool_handle()

    if implementation == "pytorch":
        value = benchmark(torch_linear_layer, a)
    elif implementation == "pytorch_cuda_graph":
        run = cuda_graphs_wrapper(model=torch_linear_layer, inputs=[a], pool=cuda_graph_pool)
        (value,) = benchmark(run, a)
    elif implementation == "triton":
        value, _ = benchmark(linear_layer, x=a, weight=layer_weight, bias=None)
    elif implementation == "triton_cuda_graph":
        def wrapper(x):
            # we fix weights, so we have not to provide (and copy) it during inference
            return linear_layer(x=x, weight=layer_weight, bias=None)

        run = cuda_graphs_wrapper(model=wrapper, inputs=[a], pool=cuda_graph_pool, copy_outputs=False)
        value, _ = benchmark(run, a)
    elif implementation == "pytorch":
        value = benchmark(torch_linear_layer, input=a)
    else:
        raise ValueError(f"Unknown implementation {implementation}")

    assert torch.allclose(value, expected, rtol=1e-1, atol=1e-1), f"max diff: {torch.abs(value - expected).max()}"

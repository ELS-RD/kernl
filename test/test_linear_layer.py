import torch
import pytest

from implementations.cuda_graph import cuda_graphs_wrapper
from implementations.linear_layer import linear_layer


@pytest.mark.parametrize("size", [8, 32, 128, 256, 384, 512])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("implementation", ["triton", "triton_cuda_graph", "pytorch", "pytorch_cuda_graph"])
def test_benchmark(benchmark, size, contiguous, batch, implementation):
    torch.manual_seed(0)

    M = size
    N = size * 4
    K = size

    # order of dimensions is wrong so we force contiguous call
    a = torch.randn((batch, K, M), device='cuda', dtype=torch.float16, requires_grad=False)
    a = a.mT
    if contiguous:
        a = a.contiguous()
    else:
        assert not a.is_contiguous()

    layer_weight = torch.randn((N, K), device='cuda', dtype=torch.float16, requires_grad=False)

    torch_linear_layer = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
    torch_linear_layer.weight.data = layer_weight
    expected = torch_linear_layer(a)
    cuda_graph_pool = torch.cuda.graph_pool_handle()

    # TODO switch to dict implementation
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
    assert torch.allclose(value, expected, atol=1e-1)

import torch
import pytest
from implementations.batched_matmul import batched_matmul
from implementations.linear_layer import linear_layer


def generate_random_data(size: tuple, device: str, torch_dtype: torch.dtype):
    return torch.randn(size, device=device, dtype=torch_dtype, requires_grad=False)


@pytest.mark.parametrize("m", [24, 32])
@pytest.mark.parametrize("n", [24, 32])
@pytest.mark.parametrize("k", [24])
@pytest.mark.parametrize("batch", [1, 16])
def test_batched_matmul_precision(m, n, k, batch):
    torch.manual_seed(0)

    a_float32 = generate_random_data((batch, m, k), 'cuda', torch.float32)
    b_float32 = generate_random_data((batch, k, n), 'cuda', torch.float32)
    expected_float32 = torch.matmul(a_float32, b_float32)
    value_float32 = batched_matmul(a_float32, b_float32)
    assert torch.allclose(value_float32, expected_float32, atol=1e-2)

    a_float16 = generate_random_data((batch, m, k), 'cuda', torch.float16)
    b_float16 = generate_random_data((batch, k, n), 'cuda', torch.float16)
    expected_float16 = torch.matmul(a_float16, b_float16)
    value_float16 = batched_matmul(a_float16, b_float16)
    assert torch.allclose(value_float16, expected_float16, atol=1e-2)

    # The tolerated difference is: 2x (PT FP32 - PT FP16)
    difference_torch = torch.sub(expected_float32, expected_float16)
    difference_triton = torch.sub(expected_float32, value_float16)
    assert torch.le(difference_triton, 2 * difference_torch)


@pytest.mark.parametrize("size", [128 * i for i in range(2, 10)])
@pytest.mark.parametrize("batch", [8])
def test_linear_layer_precision(size, batch):
    torch.manual_seed(0)

    M = size
    K = size

    a_float16 = generate_random_data((batch, M, K), 'cuda', torch.float32)
    layer_weight_float16 = generate_random_data((K * 4, K), 'cuda', torch.float16)
    torch_linear_layer_float16 = torch.nn.Linear(K, K * 4, bias=False, device="cuda", dtype=torch.float16)
    torch_linear_layer_float16.weight.data = layer_weight_float16
    expected_float16 = torch_linear_layer_float16(a_float16)
    value_float16, _ = linear_layer(x=a_float16, weight=layer_weight_float16, bias=None)
    assert torch.allclose(value_float16, expected_float16, atol=1e-2)

    layer_weight_float32 = generate_random_data((K * 4, K), 'cuda', torch.float32)
    a_float32 = generate_random_data((batch, M, K), 'cuda', torch.float32)
    torch_linear_layer_float32 = torch.nn.Linear(K, K * 4, bias=False, device="cuda", dtype=torch.float32)
    torch_linear_layer_float32.weight.data = layer_weight_float32
    expected_float32 = torch_linear_layer_float32(a_float32)
    value_float32, _ = linear_layer(x=a_float32, weight=layer_weight_float32, bias=None)
    assert torch.allclose(value_float32, expected_float32, atol=1e-2)

    # The tolerated difference is: 2x (PT FP32 - PT FP16)
    difference_torch = torch.sub(expected_float32, expected_float16)
    difference_triton = torch.sub(expected_float32, value_float16)
    assert torch.le(difference_triton, 2 * difference_torch)

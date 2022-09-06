import torch

from implementations.batched_matmul import batched_matmul
from implementations.linear_layer import linear_layer
from test.data_utils import generate_random_data


def test_batched_matmul_precision():
    for m in [m_idx * 2 for m_idx in range(8, 256, 32)]:
        for n in [n_idx * 2 for n_idx in range(8, 256, 32)]:
            for k in [k_idx * 2 for k_idx in range(8, 256, 32)]:
                for batch in [8, 16, 32]:
                    a_float32 = generate_random_data((batch, m, k), 'cuda', torch.float32)
                    b_float32 = generate_random_data((batch, k, n), 'cuda', torch.float32)
                    expected_float32 = torch.matmul(a_float32, b_float32)
                    value_float32 = batched_matmul(a_float32, b_float32)
                    torch.allclose(value_float32, expected_float32, atol=1e-2)

                    a_float16 = a_float32.to(torch.float16)
                    b_float16 = b_float32.to(torch.float16)
                    expected_float16 = torch.matmul(a_float16, b_float16)
                    value_float16 = batched_matmul(a_float16, b_float16)
                    torch.allclose(value_float16, expected_float16, atol=1e-2)

                    # The tolerated difference is: 2x (PT FP32 - PT FP16)
                    difference_torch = torch.sub(expected_float32, expected_float16)
                    difference_triton = torch.sub(expected_float32, value_float16)
                    compare_results = torch.le(torch.abs(difference_triton), torch.abs(2 * difference_torch))
                    torch.all(compare_results).cpu().numpy()


def test_linear_layer_precision():
    for size in [64 * i for i in range(1, 20)]:
        for batch in [8, 16]:
            M = size
            K = size
            a_float32 = generate_random_data((batch, M, K), 'cuda', torch.float32)
            layer_weight_float32 = generate_random_data((K * 4, K), 'cuda', torch.float32)
            torch_linear_layer_float32 = torch.nn.Linear(K, K * 4, bias=False, device="cuda", dtype=torch.float32)
            torch_linear_layer_float32.weight.data = layer_weight_float32
            expected_float32 = torch_linear_layer_float32(a_float32)
            value_float32, _ = linear_layer(x=a_float32, weight=layer_weight_float32, bias=None)
            torch.allclose(value_float32, expected_float32, atol=1e-2)

            a_float16 = a_float32.to(torch.float16)
            layer_weight_float16 = layer_weight_float32.to(torch.float16)
            torch_linear_layer_float16 = torch_linear_layer_float32.to(torch.float16)
            torch_linear_layer_float16.weight.data = layer_weight_float16
            expected_float16 = torch_linear_layer_float16(a_float16)
            value_float16, _ = linear_layer(x=a_float16, weight=layer_weight_float16, bias=None)
            torch.allclose(value_float16, expected_float16, atol=1e-2)

            # The tolerated difference is: 2x (PT FP32 - PT FP16)
            difference_torch = torch.sub(expected_float32, expected_float16)
            difference_triton = torch.sub(expected_float32, value_float16)
            compare_results = torch.le(torch.abs(difference_triton), torch.abs(2 * difference_torch))
            torch.all(compare_results).cpu().numpy()

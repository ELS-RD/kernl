import dataclasses
from typing import Dict, Callable

import torch
import pytest
from test.models.bert import get_model_baseline, get_model_dynamo, get_model_dynamo_nvfuser_ofi, \
    get_model_dynamo_dropout_removed, get_model_optimized_cuda_graphs, get_model_dynamo_cuda_graphs, \
    get_model_optimized, get_model_optimized_causal_cuda_graphs
import torchdynamo


@pytest.fixture
def model_baseline_fp32():
    return get_model_baseline(float_16=False)


@dataclasses.dataclass
class Implementation:
    model: Callable
    causal: bool

    def get_input(self, size: (int, int)) -> Dict[str, torch.Tensor]:
        if self.causal:
            batch, seq_length = size
            mask = torch.tril(torch.ones((batch, seq_length, seq_length), device="cuda"))
            return {
                "input_ids": torch.randint(2, 1000, size=size, dtype=torch.int32, device="cuda"),
                "attention_mask": mask,
            }
        else:
            return {
                "input_ids": torch.randint(2, 1000, size=size, dtype=torch.int32, device="cuda"),
                "attention_mask": None,  # TODO None is not a correct value, no key at all would be better
            }


implementations: dict[str, Implementation] = {
    "baseline": Implementation(get_model_baseline, causal=False),
    "dynamo": Implementation(get_model_dynamo, causal=False),
    "dynamo_nvfuser_ofi": Implementation(get_model_dynamo_nvfuser_ofi, causal=False),
    "dynamo_no_dropout": Implementation(get_model_dynamo_dropout_removed, causal=False),
    "dynamo_cuda_graphs": Implementation(get_model_dynamo_cuda_graphs, causal=False),
    "dynamo_optimized": Implementation(get_model_optimized, causal=False),
    "dynamo_optimized_cuda_graphs": Implementation(get_model_optimized_cuda_graphs, causal=False),
    "dynamo_optimizer_cuda_graphs_causal": Implementation(get_model_optimized_causal_cuda_graphs, causal=True),
}


@pytest.mark.parametrize("input_shape", [(1, 16), (1, 128), (1, 256), (1, 384), (1, 512),
                                         (8, 16), (8, 128), (8, 256), (8, 384), (8, 512),
                                         (32, 16), (32, 128), (32, 256),
                                         ])
@pytest.mark.parametrize("implementation", implementations.keys())
def test_benchmark_implementations(benchmark, model_baseline_fp32, input_shape: (int, int), implementation: str):
    torch.manual_seed(0)
    assert implementation in implementations, f"unknown implementation: {implementation}"
    model_tested = implementations[implementation]

    with torch.inference_mode():
        inputs = model_tested.get_input(input_shape)
        expected = model_baseline_fp32(**inputs)
        model = model_tested.model()
        value = benchmark(model, **inputs)

    torchdynamo.reset()

    assert torch.allclose(input=value["last_hidden_state"].float(), other=expected["last_hidden_state"], rtol=1e-1, atol=1e-1)
    assert torch.allclose(input=value["pooler_output"].float(), other=expected["pooler_output"], rtol=1e-1, atol=1e-1)


def test_support_shape_change(model_baseline_fp32):
    for name, implementation in implementations.items():
        model_tested = implementation.model()
        for shape in [(1, 64), (8, 256), (16, 256), (16, 64)]:
            pytorch_input = implementation.get_input(shape)
            expected = model_baseline_fp32(**pytorch_input)
            result = model_tested(**pytorch_input)
            assert torch.allclose(result["last_hidden_state"].float(), expected["last_hidden_state"], atol=1e-1), f"failed on {name} with shape {shape}"

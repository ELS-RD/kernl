from typing import Tuple, Dict, Callable

import torch
import pytest
from test.models.bert import get_model_baseline, get_model_dynamo, get_model_dynamo_nvfuser_ofi, \
    get_model_dynamo_dropout_removed, get_model_optimized_cuda_graphs, get_model_dynamo_cuda_graphs, \
    get_model_optimized, get_model_optimized_causal_cuda_graphs
import torchdynamo


def get_pytorch_input(size: (int, int)) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(2, 1000, size=size, dtype=torch.int32, device="cuda"),
        "attention_mask": None,  # TODO None is not a correct value, no key at all would be better
    }


def get_pytorch_input_causal(size: (int, int)) -> Dict[str, torch.Tensor]:
    batch, seq_length = size
    mask = torch.tril(torch.ones((batch, seq_length, seq_length), device="cuda"))
    return {
        "input_ids": torch.randint(2, 1000, size=size, dtype=torch.int32, device="cuda"),
        "attention_mask": mask,
    }


implementations = {
    "baseline": get_model_baseline,
    "dynamo": get_model_dynamo,
    "dynamo_nvfuser_ofi": get_model_dynamo_nvfuser_ofi,
    "dynamo_no_dropout": get_model_dynamo_dropout_removed,
    "dynamo_cuda_graphs": get_model_dynamo_cuda_graphs,
    "dynamo_optimized": get_model_optimized,
    "dynamo_optimized_cuda_graphs": get_model_optimized_cuda_graphs,
    "dynamo_optimizer_cuda_graphs_causal": get_model_optimized_causal_cuda_graphs,
}


def base_benchmark_bert(benchmark, pytorch_input: Callable, input_shape: (int, int), implementation: str):
    torch.manual_seed(0)
    assert implementation in implementations, f"unknown implementation: {implementation}"
    create_model = implementations[implementation]

    with torch.inference_mode():
        inputs = pytorch_input(input_shape)
        model_baseline = get_model_baseline()
        expected = model_baseline(**inputs)
        model = create_model()
        value = benchmark(model, **inputs)

    torchdynamo.reset()
    assert torch.allclose(input=value["last_hidden_state"], other=expected["last_hidden_state"], rtol=1e-1, atol=1e-1)
    assert torch.allclose(input=value["pooler_output"], other=expected["pooler_output"], rtol=1e-1, atol=1e-1)


# TODO restore values
@pytest.mark.parametrize("input_shape", [(1, 16)  # , (1, 128), (1, 256), (1, 384), (1, 512),
                                         # (8, 16), (8, 128), (8, 256), (8, 384), (8, 512),
                                         # (32, 16), (32, 128), (32, 256),
                                         ])
@pytest.mark.parametrize("implementation", [
    "baseline",
    # "dynamo",
    # "dynamo_nvfuser_ofi",
    # "dynamo_no_dropout",
    # "dynamo_cuda_graphs",
    "dynamo_optimized",
    "dynamo_optimized_cuda_graphs",
])
def test_benchmark_bert(benchmark, input_shape: (int, int), implementation: str):
    base_benchmark_bert(benchmark=benchmark,
                        pytorch_input=get_pytorch_input,
                        input_shape=input_shape,
                        implementation=implementation)


# TODO restore values
@pytest.mark.parametrize("input_shape", [(1, 16)  # , (1, 128), (1, 256), (1, 384), (1, 512),
                                         # (8, 16), (8, 128), (8, 256), (8, 384), (8, 512),
                                         # (32, 16), (32, 128), (32, 256),
                                         ])
@pytest.mark.parametrize("implementation", [
    "baseline",
    "dynamo_optimizer_cuda_graphs_causal"
])
def test_benchmark_bert_causal_mask(benchmark, input_shape: (int, int), implementation):
    base_benchmark_bert(benchmark=benchmark,
                        pytorch_input=get_pytorch_input_causal,
                        input_shape=input_shape,
                        implementation=implementation)


def test_should_support_shape_change():
    model_baseline = get_model_baseline()
    model_optimized = get_model_optimized_cuda_graphs()

    for shape in [(1, 64), (8, 256), (16, 256), (16, 64)]:
        pytorch_input = get_pytorch_input(shape)
        expected = model_baseline(**pytorch_input)
        result = model_optimized(**pytorch_input)
        assert torch.allclose(result["last_hidden_state"], expected["last_hidden_state"], atol=1e-1)

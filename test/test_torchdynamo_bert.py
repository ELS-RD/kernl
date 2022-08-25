from typing import Tuple, Dict

import torch
import pytest
from test.models.bert import get_model_baseline, get_model_dynamo, get_model_dynamo_nvfuser_ofi, \
    get_model_dynamo_dropout_removed, get_model_optimized, get_model_dynamo_cudagraphs, \
    get_model_optimized_without_cudagraph, get_model_optimized_causal, get_model_dynamo_inductor, \
    get_model_dynamo_onnx2tensorrt
import torchdynamo


def get_pytorch_input(size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(2, 1000, size=size, dtype=torch.int32, device="cuda"),
        "attention_mask": None,
    }


def get_pytorch_input_causal(size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    batch, seq_length = size
    mask = torch.tril(torch.ones((batch, seq_length, seq_length), device="cuda"))
    return {
        "input_ids": torch.randint(2, 1000, size=size, dtype=torch.int32, device="cuda"),
        "attention_mask": mask,
    }


@pytest.mark.parametrize("batch", [1, 8, 16, 32])
@pytest.mark.parametrize("seq_length", [16, 64, 128, 256])
@pytest.mark.parametrize("implementation", [
    "baseline",
    "dynamo",
    "dynamo_nvfuser_ofi",
    "dynamo_no_dropout",
    # "dynamo_onnx2tensorrt",
    "dynamo_cudagraphs",
    # "dynamo_optimized_without_cudagraph",
    "dynamo_optimized",
    # "dynamo_inductor",
])
def test_benchmark_bert(benchmark, batch, seq_length, implementation):
    with torch.inference_mode():
        torch.manual_seed(0)
        input = get_pytorch_input((batch, seq_length))

        model_baseline = get_model_baseline()
        expected = model_baseline(**input)
        value = None
        if implementation == "baseline":
            model_baseline(**input)
            value = benchmark(model_baseline, **input)
        if implementation == "dynamo":
            model = get_model_dynamo()
            value = benchmark(model, **input)
        if implementation == "dynamo_nvfuser_ofi":
            model = get_model_dynamo_nvfuser_ofi()
            value = benchmark(model, **input)
        if implementation == "dynamo_no_dropout":
            model = get_model_dynamo_dropout_removed()
            value = benchmark(model, **input)
        if implementation == "dynamo_optimized_without_cudagraph":
            model = get_model_optimized_without_cudagraph()
            value = benchmark(model, **input)
        if implementation == "dynamo_cudagraphs":
            model = get_model_dynamo_cudagraphs()
            value = benchmark(model, **input)
        if implementation == "dynamo_inductor":
            model = get_model_dynamo_inductor()
            value = benchmark(model, **input)
        if implementation == "dynamo_onnx2tensorrt":
            model = get_model_dynamo_onnx2tensorrt()
            value = benchmark(model, **input)
        if implementation == "dynamo_optimized":
            model = get_model_optimized()
            value = benchmark(model, **input)

        torchdynamo.reset()
        assert torch.allclose(value["last_hidden_state"], expected["last_hidden_state"], atol=1e-1)
        assert torch.allclose(value["pooler_output"], expected["pooler_output"], atol=1e-1)


@pytest.mark.parametrize("batch", [1, 8, 16])
@pytest.mark.parametrize("seq_length", [512])
@pytest.mark.parametrize("implementation", [
    "baseline",
    "dynamo_optimizer"
])
def test_benchmark_bert_causal_mask(benchmark, batch, seq_length, implementation):
    with torch.inference_mode():
        torch.manual_seed(0)
        input = get_pytorch_input_causal((batch, seq_length))

        model_baseline = get_model_baseline()
        expected = model_baseline(**input)
        value = None
        if implementation == "baseline":
            model_baseline(**input)
            value = benchmark(model_baseline, **input)
        if implementation == "dynamo_optimizer":
            model = get_model_optimized_causal()
            value = benchmark(model, **input)
        torchdynamo.reset()
        assert torch.allclose(value["last_hidden_state"], expected["last_hidden_state"], atol=1e-1)
        assert torch.allclose(value["pooler_output"], expected["pooler_output"], atol=1e-1)

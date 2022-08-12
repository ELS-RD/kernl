from typing import Tuple, Dict

import torch
import pytest
from implementations.torchdynamo_bert import get_model_baseline, get_model_dynamo, get_model_dynamo_nvfuser_ofi, \
    get_model_dynamo_droput_removed, get_model_dynamo_fused_attention


def get_pytorch_input(size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(2, 1000, size=size, dtype=torch.int32, device="cuda"),
        "attention_mask": None,
    }


@pytest.mark.parametrize("batch", [1, 8, 16])
@pytest.mark.parametrize("implementation", [
    "baseline",
    "torchdynamo",
    "torchdynamo_nvfuser_ofi",
    "torchdynamo_no_dropout",
    "torchdynamo_fused_attention"
])
def test_benchmark(benchmark, batch, implementation):
    with torch.inference_mode():
        torch.manual_seed(0)
        input = get_pytorch_input((batch, 512))

        model_baseline = get_model_baseline()
        expected = model_baseline(**input)["last_hidden_state"]
        if implementation == "baseline":
            model_baseline(**input)
            value = benchmark(model_baseline, **input)
            assert torch.allclose(value["last_hidden_state"], expected, atol=1e-2)
        if implementation == "torchdynamo":
            model = get_model_dynamo()
            value = benchmark(model, **input)
            assert torch.allclose(value["last_hidden_state"], expected, atol=1e-2)
        if implementation == "torchdynamo_nvfuser_ofi":
            model = get_model_dynamo_nvfuser_ofi()
            value = benchmark(model, **input)
            # WARNING PRECISION CHANGED HERE !!!!
            assert torch.allclose(value["last_hidden_state"], expected, atol=1e-1)
        if implementation == "torchdynamo_no_dropout":
            model = get_model_dynamo_droput_removed()
            value = benchmark(model, **input)
            assert torch.allclose(value["last_hidden_state"], expected, atol=1e-2)
        if implementation == "torchdynamo_fused_attention":
            model = get_model_dynamo_fused_attention()
            value = benchmark(model, **input)
            # WARNING PRECISION CHANGED HERE !!!!
            assert torch.allclose(value["last_hidden_state"], expected, atol=1e-1)

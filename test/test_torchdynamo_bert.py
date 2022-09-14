import dataclasses
import logging
from typing import Callable

import pytest
import torch
import torchdynamo

from test.models.bert import get_model_baseline, get_model_dynamo, get_model_dynamo_nvfuser_ofi, \
    get_model_dynamo_dropout_removed, get_model_optimized_cuda_graphs, get_model_dynamo_cuda_graphs, \
    get_model_optimized, get_model_optimized_causal_cuda_graphs, get_bert_onnx, get_bert_tensorrt
from test.utils.benchmark_utils import get_input_non_causal, get_input_causal

logger = logging.getLogger(__name__)


@pytest.fixture
def model_baseline_fp32():
    return get_model_baseline(float_16=False)


@dataclasses.dataclass
class Implementation:
    model: Callable
    is_causal: bool


implementations: dict[str, Implementation] = {
    "baseline": Implementation(get_model_baseline, is_causal=False),
    "dynamo": Implementation(get_model_dynamo, is_causal=False),
    "dynamo_nvfuser_ofi": Implementation(get_model_dynamo_nvfuser_ofi, is_causal=False),
    "dynamo_no_dropout": Implementation(get_model_dynamo_dropout_removed, is_causal=False),
    "dynamo_cuda_graphs": Implementation(get_model_dynamo_cuda_graphs, is_causal=False),
    "dynamo_optimized": Implementation(get_model_optimized, is_causal=False),
    "dynamo_optimized_cuda_graphs": Implementation(get_model_optimized_cuda_graphs, is_causal=False),
    "dynamo_optimizer_cuda_graphs_causal": Implementation(get_model_optimized_causal_cuda_graphs, is_causal=True),
}

try:
    import onnxruntime as ort
    implementations["onnx"] = Implementation(get_bert_onnx, is_causal=True)
except ImportError:
    logger.warning(
        "It seems that onnx runtime is not yet installed. Onnx models will not be included in the benchmark."
    )

try:
    import tensorrt as trt
    implementations["tensorrt"] = Implementation(get_bert_tensorrt, is_causal=True)
except ImportError:
    logger.error(
        "It seems that TensorRT is not yet installed. It is required to include TensorRT in benchmark."
    )


@pytest.mark.parametrize("input_shape", [(1, 16)
                                         ], ids=lambda x: f"{x[0]}x{x[1]}")
@pytest.mark.parametrize("implementation", implementations.keys())
def test_benchmark_implementations(benchmark, model_baseline_fp32, input_shape: (int, int), implementation: str):
    torch.manual_seed(0)
    assert implementation in implementations, f"unknown implementation: {implementation}"
    model_tested = implementations[implementation]

    inputs = get_input_causal(input_shape) if model_tested.is_causal else get_input_non_causal(input_shape)

    with torch.inference_mode():
        expected = model_baseline_fp32(**inputs)
        model = model_tested.model(shape=input_shape) if implementation in ["onnx", "tensorrt"] else model_tested.model()
        value = benchmark(model, **inputs)

    torchdynamo.reset()

    if "last_hidden_state" in value:
        assert torch.allclose(input=value["last_hidden_state"].float(), other=expected["last_hidden_state"], rtol=1e-1, atol=1e-1)
    if "pooler_output" in value:
        assert torch.allclose(input=value["pooler_output"].float(), other=expected["pooler_output"], rtol=1e-1, atol=1e-1)


def test_support_shape_change(model_baseline_fp32):
    """Test that the model can handle shape changes without being reloaded/rebuilt."""
    for name, implementation in implementations.items():
        if name in ["onnx", "tensorrt"]:
            # onnx and tensorrt implementations are shape dependent for now,
            # so they don't support handling shape change without reload.
            continue
        model_tested = implementation.model()
        for shape in [(1, 64), (8, 256), (16, 256), (16, 64)]:
            pytorch_input = get_input_causal(shape) if implementation.is_causal else get_input_non_causal(shape)
            expected = model_baseline_fp32(**pytorch_input)
            result = model_tested(**pytorch_input)
            assert torch.allclose(result["last_hidden_state"].float(), expected["last_hidden_state"], atol=1e-1), f"failed on {name} with shape {shape}"

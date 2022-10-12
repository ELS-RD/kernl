#  Copyright 2022 Lefebvre Sarrut
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import dataclasses
import warnings
from test.models.bert import (
    get_model_baseline,
    get_model_dynamo,
    get_model_dynamo_cuda_graphs,
    get_model_dynamo_dropout_removed,
    get_model_dynamo_nvfuser_ofi,
    get_model_from_hf,
    get_model_optimized,
    get_model_optimized_causal_cuda_graphs,
    get_model_optimized_cuda_graphs,
)
from test.models.data_utils import get_input_causal, get_input_non_causal
from typing import Callable, Dict

import pytest
import torch
import torchdynamo

from conftest import check_all_close, set_seed


@dataclasses.dataclass
class Implementation:
    name: str
    model: Callable
    is_causal: bool


implementations: [Implementation] = [
    Implementation("baseline", get_model_baseline, is_causal=False),
    Implementation("dynamo", get_model_dynamo, is_causal=False),
    Implementation("dynamo_nvfuser_ofi", get_model_dynamo_nvfuser_ofi, is_causal=False),
    Implementation("dynamo_no_dropout", get_model_dynamo_dropout_removed, is_causal=False),
    Implementation("dynamo_cuda_graphs", get_model_dynamo_cuda_graphs, is_causal=False),
    Implementation("dynamo_optimized", get_model_optimized, is_causal=False),
    Implementation("dynamo_optimized_cuda_graphs", get_model_optimized_cuda_graphs, is_causal=False),
    # In this implementation both causal mask and the assume causal mask optimization will be applied, leads to slower
    # benchmark. It's not needed if we are sure the mask is causal, we can use the "assume causal mask optimization".
    Implementation("dynamo_optimizer_cuda_graphs_causal", get_model_optimized_causal_cuda_graphs, is_causal=True),
]


try:
    # check imports and initialize onnx model
    from test.models.bert import get_bert_onnx

    # _ = get_bert_onnx()
    implementations.append(Implementation("onnx", get_bert_onnx, is_causal=False))
except ImportError as e:
    error = f"It seems that you are missing some dependencies: onnx won't be included in benchmarks. \n {str(e)}"
    warnings.warn(UserWarning(error))

try:
    # check imports and initialize optimized fp32 onnx model
    from test.models.bert import get_bert_optim_fp32_onnx

    # _ = get_bert_optim_fp32_onnx()
    implementations.append(Implementation("onnx_optim_fp32", get_bert_optim_fp32_onnx, is_causal=False))
except ImportError as e:
    error = (
        f"It seems that you are missing some dependencies: onnx_optim_fp32 won't be included in benchmarks. \n {str(e)}"
    )
    warnings.warn(UserWarning(error))

try:
    # check imports and initialize optimized fp16 onnx model
    from test.models.bert import get_bert_optim_fp16_onnx

    # _ = get_bert_optim_fp16_onnx()
    implementations.append(Implementation("onnx_optim_fp16", get_bert_optim_fp16_onnx, is_causal=False))
except ImportError as e:
    error = (
        f"It seems that you are missing some dependencies: onnx_optim_fp16 won't be included in benchmarks. \n {str(e)}"
    )
    warnings.warn(UserWarning(error))


@pytest.fixture
def reference_fp32(request):
    return get_model_from_hf(request.param)


@set_seed()
@pytest.mark.parametrize(
    "reference_fp32",
    [
        "bert-base-uncased",
        "t5-small",
        "distilbert-base-uncased",
        "xlm-roberta-base",
        "camembert-base",
        "sentence-transformers/all-MiniLM-L6-v2",
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "shape",
    [(bs, seq_l) for bs in [1, 8, 32] for seq_l in [16, 128, 256, 384, 512] if bs * seq_l < 10000],
    ids=lambda x: f"{x[0]}x{x[1]}",
)
@pytest.mark.parametrize("implementation", implementations, ids=lambda v: v.name)
def test_benchmark_implementations(benchmark, reference_fp32, shape: (int, int), implementation: Implementation):
    inputs = (
        get_input_causal(reference_fp32, shape)
        if implementation.is_causal
        else get_input_non_causal(reference_fp32, shape)
    )
    with torch.inference_mode():
        expected = reference_fp32(**inputs)
        model = implementation.model(reference_fp32)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
            value = benchmark(model, **inputs)

    torchdynamo.reset()

    check_all_close(value["last_hidden_state"].float(), expected["last_hidden_state"].float(), rtol=1e-1, atol=1e-1)

    if "pooler_output" in expected:
        check_all_close(value["pooler_output"].float(), expected["pooler_output"].float(), rtol=1e-1, atol=1e-1)


@set_seed()
@pytest.mark.parametrize("implementation", implementations, ids=lambda v: v.name)
def test_support_shape_change(implementation):
    """Test that the model can handle shape changes without being reloaded/rebuilt."""
    model_reference_fp32 = get_model_from_hf("bert-base-uncased")
    model_tested = implementation.model(get_model_from_hf("bert-base-uncased"))
    for shape in [(1, 64), (8, 256), (16, 256), (16, 64)]:
        pytorch_input = (
            get_input_causal(model_reference_fp32, shape)
            if implementation.is_causal
            else get_input_non_causal(model_reference_fp32, shape)
        )
        with torch.inference_mode():
            expected = model_reference_fp32(**pytorch_input)
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                result = model_tested(**pytorch_input)
        max_diff = torch.max(torch.abs(result["last_hidden_state"].float() - expected["last_hidden_state"].float()))
        check_all_close(
            result["last_hidden_state"].float(), expected["last_hidden_state"].float(), atol=1e-1, rtol=1e-1
        ), f"failed with shape {shape}, max diff: {max_diff}"

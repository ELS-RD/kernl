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
from test.models.bert import (
    get_model_baseline,
    get_model_dynamo_cuda_graphs,
    get_model_dynamo_nvfuser_ofi,
    get_model_from_hf,
    get_model_optimized,
    get_model_optimized_causal_cuda_graphs,
    get_model_optimized_cuda_graphs,
)
from test.models.data_utils import get_input
from typing import Callable

import pytest
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from conftest import check_all_close, set_seed, setup_dynamo

from kernl.model_optimization import optimize_model


@dataclasses.dataclass
class Implementation:
    name: str
    model: Callable
    is_causal: bool


implementations: [Implementation] = [
    Implementation("baseline", get_model_baseline, is_causal=False),
    Implementation("dynamo_nvfuser_ofi", get_model_dynamo_nvfuser_ofi, is_causal=False),
    Implementation("dynamo_cuda_graphs", get_model_dynamo_cuda_graphs, is_causal=False),
    Implementation("dynamo_optimized", get_model_optimized, is_causal=False),
    Implementation("dynamo_optimized_cuda_graphs", get_model_optimized_cuda_graphs, is_causal=False),
    # In this implementation both causal mask and the assume causal mask optimization will be applied, leads to slower
    # benchmark. It's not needed if we are sure the mask is causal, we can use the "assume causal mask optimization".
    Implementation("dynamo_optimizer_cuda_graphs_causal", get_model_optimized_causal_cuda_graphs, is_causal=True),
]


@pytest.fixture
def reference_fp32(request):
    return get_model_from_hf(request.param)


@setup_dynamo()
@set_seed()
@pytest.mark.parametrize(
    "reference_fp32",
    ["bert-base-uncased", "t5-small"],
    indirect=True,
)
@pytest.mark.parametrize(
    "shape",
    # TODO add shape 32x32 which may be unstable with T5
    [(bs, seq_l) for bs in [1, 8, 32] for seq_l in [16, 33, 128, 256, 384, 512] if bs * seq_l < 10000],
    ids=lambda x: f"{x[0]}x{x[1]}",
)
@pytest.mark.parametrize("implementation", implementations, ids=lambda v: v.name)
def test_benchmark_implementations(benchmark, reference_fp32, shape: (int, int), implementation: Implementation):
    if "nvfuser" in implementation.name and reference_fp32.config.name_or_path != "bert-base-uncased":
        pytest.skip("Only supported for BERT")

    inputs = get_input(reference_fp32, shape, is_causal=implementation.is_causal)
    with torch.inference_mode():
        expected = reference_fp32(**inputs)
        model = implementation.model(reference_fp32)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
            value = benchmark(model, **inputs)

    check_all_close(value["last_hidden_state"].float(), expected["last_hidden_state"].float(), rtol=1e-1, atol=1e-1)

    if "pooler_output" in expected:
        check_all_close(value["pooler_output"].float(), expected["pooler_output"].float(), rtol=1e-1, atol=1e-1)


@setup_dynamo()
@set_seed()
@pytest.mark.parametrize("implementation", implementations, ids=lambda v: v.name)
def test_support_shape_change(implementation):
    """Test that the model can handle shape changes without being reloaded/rebuilt."""
    model_reference_fp32 = get_model_from_hf("bert-base-uncased")
    model_tested = implementation.model(get_model_from_hf("bert-base-uncased"))
    for shape in [(1, 64), (8, 256), (16, 256), (16, 64)]:
        pytorch_input = get_input(model_reference_fp32, shape, is_causal=implementation.is_causal)
        with torch.inference_mode():
            expected = model_reference_fp32(**pytorch_input)
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                result = model_tested(**pytorch_input)
        check_all_close(
            result["last_hidden_state"].float(), expected["last_hidden_state"].float(), atol=1e-1, rtol=1e-1
        )


@setup_dynamo()
def test_t5():
    tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=512)
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model = model.eval().cuda()
    task = "translate English to French: The house is wonderful."
    inputs = tokenizer(task, return_tensors="pt", padding=True).to("cuda")

    with torch.inference_mode(), torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            min_length=20,
            max_length=20,
            do_sample=False,
        )
        assert "La maison est merveilleuse." in tokenizer.batch_decode(output_sequences, skip_special_tokens=True)[0]

    optimize_model(model.encoder)
    optimize_model(model.decoder)

    with torch.inference_mode(), torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            min_length=20,
            max_length=20,
            do_sample=False,
        )
        assert "La maison est merveilleuse." in tokenizer.batch_decode(output_sequences, skip_special_tokens=True)[0]

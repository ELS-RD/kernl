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

import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch

# noinspection PyUnresolvedReferences
from onnxruntime import GraphOptimizationLevel, InferenceSession, IOBinding, SessionOptions
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.onnx import FeaturesManager, export


def build_onnx(
    model: PreTrainedModel,
    model_path: str,
    model_type: str = "bert",
    hidden_size: int = 0,
    num_heads: int = 0,
):
    from onnxruntime.transformers.optimizer import optimize_model

    optim_model_name = f"{model.config.name_or_path}_optim_fp16.onnx"
    if os.path.exists(os.path.join(model_path, optim_model_name)):
        optimized_model = create_model_for_provider(os.path.join(model_path, optim_model_name))
        return optimized_model

    onnx_path_model = os.path.join(model_path, f"{model.config.name_or_path}_base.onnx")
    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    onnx_path = Path(onnx_path_model)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model)
    onnx_config = model_onnx_config(model.config)
    _ = export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=onnx_config.default_onnx_opset,
        output=onnx_path,
        device="cuda",
    )

    assert onnx_path.exists()

    opt_level = 1 if model_type == "bert" else 0
    optimized_model = optimize_model(
        input=str(onnx_path),
        model_type=model_type,
        num_heads=num_heads,
        hidden_size=hidden_size,
        opt_level=opt_level,
        use_gpu=True,
        only_onnxruntime=True,
    )

    optimized_model.convert_float_to_float16(use_symbolic_shape_infer=False, keep_io_types=False)

    optimized_model.save_model_to_file(os.path.join(model_path, optim_model_name))
    return create_model_for_provider(os.path.join(model_path, optim_model_name))


def get_model_optim_fp16_onnx(model: PreTrainedModel, model_path: str) -> Callable:
    model_onnx = build_onnx(model, model_path)
    binding: IOBinding = model_onnx.io_binding()
    expected_input_names = [i.name for i in model_onnx.get_inputs()]

    def run(*args, **kwargs):
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
        inputs = {k: v for k, v in kwargs.items() if k in expected_input_names}
        outputs = inference_onnx_binding(model_onnx=model_onnx, binding=binding, inputs=inputs)
        return BaseModelOutputWithPooling(last_hidden_state=outputs["last_hidden_state"], pooler_output=outputs["1525"])

    return run


def create_model_for_provider(path: str) -> InferenceSession:
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.log_severity_level = 2
    return InferenceSession(path, options, providers=["CUDAExecutionProvider"])


def inference_onnx_binding(
    model_onnx: InferenceSession, binding: IOBinding, inputs: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    device = "cuda"
    device_id = 0
    bs, seq_len = inputs["input_ids"].shape
    for input_onnx in model_onnx.get_inputs():
        if input_onnx.name not in inputs:  # some inputs may be optional
            continue
        tensor: torch.Tensor = inputs[input_onnx.name]

        binding.bind_input(
            name=input_onnx.name,
            device_type=device,
            device_id=device_id,
            element_type=np.int64,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )
        inputs[input_onnx.name] = tensor

    outputs: dict[str, torch.Tensor] = dict()
    for out in model_onnx.get_outputs():
        if out.name == "last_hidden_state":
            output_tensor = torch.empty((bs, seq_len, 768), dtype=torch.float16, device="cuda")
        elif out.name == "1525":  # pooler_output
            output_tensor = torch.empty((bs, 768), dtype=torch.float16, device="cuda")
        else:
            raise ValueError(f"Unknown output name {out.name}. We only support Bert model.")
        outputs[out.name] = output_tensor
        binding.bind_output(
            name=out.name,
            device_type=device,
            device_id=device_id,
            element_type=np.float16,
            shape=tuple(output_tensor.shape),
            buffer_ptr=output_tensor.data_ptr(),
        )

    model_onnx.run_with_iobinding(binding)

    return outputs


models_dir = tempfile.TemporaryDirectory().name

bert_model = AutoModel.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
bert_model.eval().cuda()

apply = get_model_optim_fp16_onnx(model=bert_model, model_path=os.path.join(models_dir, "model.onnx"))


for shape in [(bs, seq_l) for bs in [1, 8, 32] for seq_l in [16, 32, 64, 128, 256, 384, 512] if bs * seq_l < 10000]:
    inputs = {
        "input_ids": torch.randint(10, 10000, shape, device="cuda", dtype=torch.long),
        "token_type_ids": torch.ones(shape, device="cuda", dtype=torch.long),
        "attention_mask": torch.ones(shape, device="cuda", dtype=torch.long),
    }

    timings = list()

    # warmup
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        for _ in range(10):
            output_ort = apply(**inputs)

        output_torch = bert_model(**inputs)
        assert torch.allclose(output_ort.last_hidden_state.float(), output_torch.last_hidden_state, 1e-1, 1e-1)

        torch.cuda.synchronize()
        for _ in range(10):
            start = time.perf_counter()
            apply(**inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

    print(f"{shape}: {torch.median(torch.tensor(timings)):.4f}")

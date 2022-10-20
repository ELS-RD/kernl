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
from pathlib import Path
from test.models.ort_utils import create_model_for_provider

from transformers import AutoTokenizer, PreTrainedModel
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


def get_model_optim_fp16_onnx(model: PreTrainedModel, model_path: str):
    from test.models.ort_utils import inference_onnx_binding

    from onnxruntime import IOBinding
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    model_onnx = build_onnx(model, model_path)
    binding: IOBinding = model_onnx.io_binding()
    expected_input_names = [i.name for i in model_onnx.get_inputs()]

    def run(*args, **kwargs):
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
        inputs = {k: v for k, v in kwargs.items() if k in expected_input_names}
        outputs = inference_onnx_binding(model_onnx=model_onnx, binding=binding, inputs=inputs)
        return BaseModelOutputWithPooling(last_hidden_state=outputs["last_hidden_state"], pooler_output=outputs["1607"])

    return run

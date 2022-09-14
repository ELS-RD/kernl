#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from pathlib import Path

import onnx
import onnxoptimizer
from onnx.shape_inference import infer_shapes_path


def save_onnx(proto: onnx.ModelProto, model_path: str, clean: bool = True) -> None:
    """
    Save Onnx file with external data support when required.

    :param proto: Onnx model
    :param clean: clean the model before saving
    :param model_path: output path
    """
    # protobuff doesn't support files > 2Gb, in this case, weights are stored in another binary file
    if clean:
        to_save = clean_graph(proto=proto)
    else:
        to_save = proto
    save_external_data: bool = to_save.ByteSize() > 2 * 1024**3
    filename = Path(model_path).name
    onnx.save_model(
        proto=to_save,
        f=model_path,
        save_as_external_data=save_external_data,
        all_tensors_to_one_file=True,
        location=filename + ".data",
    )
    infer_shapes_path(model_path=model_path, output_path=model_path)


def clean_graph(proto: onnx.ModelProto) -> onnx.ModelProto:
    """
    Remove unused nodes and unused initializers.
    May help TensorRT when it refuses to load a model.

    :param proto: Onnx model
    :return: Onnx model
    """
    # operations that are tested with transformers models
    all_optimizations = [
        "eliminate_deadend",
        "eliminate_duplicate_initializer",
        "eliminate_identity",
        "eliminate_nop_cast",
        "eliminate_nop_dropout",
        "eliminate_nop_flatten",
        "eliminate_nop_monotone_argmax",
        "eliminate_nop_pad",
        "eliminate_nop_transpose",
        "eliminate_unused_initializer",
    ]

    cleaned_model: onnx.ModelProto = onnxoptimizer.optimize(model=proto, passes=all_optimizations)
    return cleaned_model

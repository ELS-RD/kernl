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

from typing import Callable, Union

import torch
from torch._inductor.compile_fx import cudagraphify_impl
from torch._inductor.utils import dynamo_utils
from torch._subclasses import FakeTensor

static_inputs: list[torch.Tensor] = list()


def get_static_input_ids(model_inputs: list[torch.Tensor]) -> (list[int], list[torch.Tensor]):
    """
    Returns the indices of the inputs that are static (i.e. already seen).
    To know if a tensor is static, we check if it has a `is_static` attribute, if not we add it for the next time.
    @param model_inputs: list of inputs
    @return: list of indices of static inputs and list of inputs
    """
    static_input_indexes: list[int] = list()
    local_static_inputs = static_inputs.copy()
    for input_index in range(len(model_inputs)):
        original_tensor = model_inputs[input_index]
        if getattr(original_tensor, "seen", False):
            # cases:
            # - first prediction: static_inputs_exist is False
            # - new token: static_inputs_exist is True and static_index_set is True
            if len(local_static_inputs) == 0:  # first token to save of first prediction (likely warmup)
                static_tensor = original_tensor.clone().detach()
                static_inputs.append(static_tensor)
            else:  # new token
                static_tensor = local_static_inputs.pop(0)
                static_tensor.copy_(original_tensor)
            model_inputs[input_index] = static_tensor
            static_input_indexes.append(input_index)
        else:
            setattr(original_tensor, "seen", True)
    return static_input_indexes, model_inputs


def cuda_graphs_wrapper(model: Callable, inputs: Union[list[torch.Tensor], tuple[torch.Tensor]]) -> Callable:
    """
    Wrapper to run the model with cuda graphs.
    @param model: model to save as a CUDA graph
    @param inputs: inputs to the model
    @return: an inference function that runs the model with cuda graphs
    """

    assert isinstance(inputs, (list, tuple))
    # if using fake tensors, defer cudagraphs until we get real inputs at runtime
    if not any(isinstance(inp, FakeTensor) for inp in inputs):
        static_inputs, inputs = get_static_input_ids(inputs)

        model(*inputs)  # additional warmup needed when input is mutated by some kernel
        f = cudagraphify_impl(model=lambda args: model(*args), inputs=inputs, static_input_idxs=tuple(static_inputs))
        return lambda args: f(list(args))

    compiled_fn = None

    def run(*new_inputs):
        static_inputs, new_inputs = get_static_input_ids(list(new_inputs))
        nonlocal compiled_fn
        if compiled_fn is None:
            with dynamo_utils.preserve_rng_state():
                model(*new_inputs)  # additional warmup needed when input is mutated by some kernel
                f = cudagraphify_impl(
                    model=lambda args: model(*args), inputs=new_inputs, static_input_idxs=tuple(static_inputs)
                )

                def compiled_fn(args):
                    return f(list(args))

        return compiled_fn(new_inputs)

    return run

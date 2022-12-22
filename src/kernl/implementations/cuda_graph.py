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


def get_static_input_ids(inputs: list[torch.Tensor]) -> list[int]:
    """
    Returns the indices of the inputs that are static (i.e. already seen).
    To know if a tensor is static, we check if it has a `is_static` attribute, if not we add it for the next time.
    @param inputs: list of inputs
    @return: list of indices of static inputs
    """
    static_input_ids: list[int] = list()
    for idx, i in enumerate(inputs):
        if getattr(i, "is_static", False):
            static_input_ids.append(idx)
        else:
            setattr(i, "is_static", True)
    return static_input_ids


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
        static_inputs = get_static_input_ids(inputs)

        model(*inputs)  # additional warmup needed when input is mutated by some kernel
        f = cudagraphify_impl(model=lambda args: model(*args), inputs=inputs, static_input_idxs=tuple(static_inputs))
        return lambda args: f(list(args))

    compiled_fn = None

    def run(*new_inputs):
        static_inputs = get_static_input_ids(list(new_inputs))
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

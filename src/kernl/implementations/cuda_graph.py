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
from collections import defaultdict, deque
from typing import Callable, Union

import torch
import triton
from torch._inductor.compile_fx import cudagraphify_impl
from torch._inductor.utils import dynamo_utils
from torch._subclasses import FakeTensor


static_inputs: dict[str, deque[torch.Tensor]] = defaultdict(deque)


def get_static_inputs(model_inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Copy input tensors to a managed pool of tensors (to limit memory footprint).
    All tensors need to be declared as static inputs to CUDA graph.

    @param model_inputs: list of inputs
    @return: list of inputs to be used in CUDA graphs
    """
    local_static_inputs: dict[str, deque[torch.Tensor]] = defaultdict(deque)
    for k, v in static_inputs.items():
        local_static_inputs[k] = v.copy()
    cuda_graph_input = list()
    for index, original_tensor in enumerate(model_inputs):
        storage_size = triton.next_power_of_2(len(original_tensor._storage()))
        key = f"{storage_size}_{original_tensor.dtype}"
        if len(local_static_inputs[key]) > 0:
            static_tensor = local_static_inputs[key].popleft()
        else:
            static_tensor = torch.empty((storage_size,), dtype=original_tensor.dtype, device=original_tensor.device)
            static_inputs[key].append(static_tensor)

        # storage offset should not be used... otherwise it changes cuda address
        static_tensor = torch.as_strided(static_tensor, original_tensor.size(), original_tensor.stride())
        static_tensor.copy_(original_tensor)
        cuda_graph_input.append(static_tensor)

    return cuda_graph_input


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
        inputs = get_static_inputs(inputs)
        static_inputs = tuple(range(len(inputs)))

        model(*inputs)  # additional warmup needed when input is mutated by some kernel
        f = cudagraphify_impl(model=lambda args: model(*args), inputs=inputs, static_input_idxs=static_inputs)
        return lambda args: f(list(args))

    compiled_fn = None

    def run(*new_inputs):
        new_inputs = get_static_inputs(list(new_inputs))
        static_inputs = tuple(range(len(inputs)))
        nonlocal compiled_fn
        if compiled_fn is None:
            with dynamo_utils.preserve_rng_state():
                model(*new_inputs)  # additional warmup needed when input is mutated by some kernel
                f = cudagraphify_impl(
                    model=lambda args: model(*args), inputs=new_inputs, static_input_idxs=static_inputs
                )

                def compiled_fn(args):
                    return f(list(args))

        return compiled_fn(new_inputs)

    return run

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


def cuda_graphs_wrapper(model: Callable, inputs: Union[list[torch.Tensor], tuple[torch.Tensor]]):
    assert isinstance(inputs, (list, tuple))
    # if using fake tensors, defer cudagraphs until we get real inputs at runtime
    if not any(isinstance(inp, FakeTensor) for inp in inputs):
        f = cudagraphify_impl(lambda args: model(*args), inputs)
        return lambda args: f(list(args))

    compiled_fn = None

    def run(*new_inputs):
        nonlocal compiled_fn
        if compiled_fn is None:
            with dynamo_utils.preserve_rng_state():
                f = cudagraphify_impl(lambda args: model(*args), new_inputs)

                def compiled_fn(args):
                    return f(list(args))

        return compiled_fn(new_inputs)

    return run

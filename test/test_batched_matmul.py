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

import pytest
import torch

from conftest import assert_all_close, set_seed

from kernl.implementations.batched_matmul import batched_matmul


@set_seed()
@pytest.mark.parametrize("m", [24, 32], ids=lambda x: f"m={x}")
@pytest.mark.parametrize("n", [24, 32], ids=lambda x: f"n={x}")
@pytest.mark.parametrize("k", [24], ids=lambda x: f"k={x}")
@pytest.mark.parametrize("batch", [1, 16], ids=lambda x: f"batch={x}")
@pytest.mark.parametrize("implementation", ["pytorch", "triton"])
def test_benchmark(benchmark, m, n, k, batch, implementation):
    a = torch.randn((batch, m, k), device="cuda", dtype=torch.float16, requires_grad=False)
    b = torch.randn((batch, k, n), device="cuda", dtype=torch.float16, requires_grad=False)
    expected = torch.matmul(a, b)
    if implementation == "pytorch":
        value = benchmark(torch.matmul, a, b)
    elif implementation == "triton":
        value = benchmark(batched_matmul, a, b)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    assert_all_close(value, expected, atol=1e-2)

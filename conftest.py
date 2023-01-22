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
import gc
import random
from contextlib import contextmanager

import pytest
import torch
import torch._dynamo as dynamo

from kernl.benchmark.benchmark_fixture import BenchmarkFixture
from kernl.benchmark.benchmark_session import BenchmarkSession
from kernl.optimizer.cuda_graph import static_inputs_pool


@contextmanager
def set_seed(seed: int = 0):
    torch.manual_seed(seed=seed)
    random.seed(seed)
    yield


@pytest.fixture(autouse=True)
def reset_kernl_state():
    cache_limit = dynamo.config.cache_size_limit
    try:
        dynamo.config.cache_size_limit = 512
        dynamo.reset()
        static_inputs_pool.clear()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        yield {}
    except RuntimeError as err:
        raise err
    finally:
        dynamo.config.cache_size_limit = cache_limit
        static_inputs_pool.clear()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="function")
def benchmark(request):
    bs = request.config._benchmarksession
    node = request.node
    fixture = BenchmarkFixture(node, add_result=bs.benchmarks.append)
    return fixture


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config):
    config._benchmarksession = BenchmarkSession(config)


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--benchmark-group-by",
        action="store",
        default="fullname",
        help="Comma-separated list of categories by which to group tests. Can be one or more of: "
        "‘group’, ‘name’, ‘fullname’, ‘func’, ‘fullfunc’, ‘param’ or ‘param:NAME’, "
        "where NAME is the name passed to @pytest.parametrize. Default: ‘fullname’",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus):
    session.config._benchmarksession.finish()
    yield


def assert_all_close(a: torch.Tensor, b: torch.Tensor, rtol=0, atol=1e-1) -> None:
    """
    Check that all elements of tensors a and b are within provided thresholds.
    """
    assert a.shape == b.shape, f"Shapes don't match: {a.shape} != {b.shape}"
    assert a.dtype == b.dtype, f"Dtypes don't match: {a.dtype} != {b.dtype}"
    assert a.device == b.device, f"Devices don't match: {a.device} != {b.device}"
    max_abs_diff = torch.max(torch.abs(a - b))
    rel_diff = torch.abs(a / b)
    max_rel_diff = torch.max(rel_diff)
    mismatch_elements = torch.sum(torch.abs(a - b) > atol + rtol * torch.abs(b))
    nb_elements = torch.numel(a)
    msg = (
        f"Differences: "
        f"{max_abs_diff:.3f} (max abs), "
        f"{max_rel_diff:.3f} (max rel), "
        f"{mismatch_elements}/{nb_elements} (mismatch elements)"
    )
    assert torch.allclose(a, b, rtol=rtol, atol=atol), msg

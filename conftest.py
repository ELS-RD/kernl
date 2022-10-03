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

import random

import pytest
import torch
from decorator import contextmanager

from nucle.benchmark.benchmark_fixture import BenchmarkFixture
from nucle.benchmark.benchmark_session import BenchmarkSession


def pytest_generate_tests(metafunc):
    if "shape" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            batches = [1, 8, 16]
            seq_len = [16, 128, 384, 512]
        else:
            batches = [1, 3]
            seq_len = [5, 512]
        shapes = [(bs, sl) for bs in batches for sl in seq_len if bs * sl <= 10000]
        shapes += [(32, 32)]  # a shape that may be buggy on attention kernel
        shapes += [(1, 8)]  # super small shape
        shapes += [(3, 33)]  # non power of 2 shape
        metafunc.parametrize("shape", shapes, ids=lambda s: f"{s[0]}x{s[1]}")


@contextmanager
def set_seed(seed: int = 0):
    torch.manual_seed(seed=seed)
    random.seed(seed)
    yield


@pytest.fixture(scope="function")
def benchmark(request):
    bs = request.config._benchmarksession
    node = request.node
    fixture = BenchmarkFixture(node, add_result=bs.benchmarks.append)
    return fixture


@pytest.mark.trylast
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
    parser.addoption("--all", action="store_true", help="tests all shapes")


@pytest.hookimpl(hookwrapper=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus):
    session.config._benchmarksession.finish()
    yield

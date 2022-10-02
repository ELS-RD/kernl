import random

import pytest
import torch
from decorator import contextmanager

from nucle.benchmark.benchmark_fixture import BenchmarkFixture
from nucle.benchmark.benchmark_session import BenchmarkSession


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


@pytest.hookimpl(hookwrapper=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus):
    session.config._benchmarksession.finish()
    yield

import pytest
from benchmark import BenchmarkFixture, BenchmarkSession


@pytest.fixture(scope="function")
def benchmark(request):
    bs = request.config._benchmarksession
    node = request.node
    fixture = BenchmarkFixture(
        node,
        add_result=bs.benchmarks.append
    )
    return fixture


@pytest.mark.trylast
def pytest_configure(config):
    config._benchmarksession = BenchmarkSession(config)

def pytest_addoption(parser):
    parser.addoption(
        "--benchmark-group-by", action="store", default="fullname", help="Comma-separated list of categories by which to group tests. Can be one or more of: ‘group’, ‘name’, ‘fullname’, ‘func’, ‘fullfunc’, ‘param’ or ‘param:NAME’, where NAME is the name passed to @pytest.parametrize. Default: ‘fullname’"
    )

@pytest.hookimpl(hookwrapper=True)
def pytest_sessionfinish(session, exitstatus):
    session.config._benchmarksession.finish()
    yield

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.benchmark_fixture import BenchmarkFixture
    from benchmark.benchmark_data import BenchmarkData


class BenchmarkResult(object):
    def __init__(self, fixture: BenchmarkFixture, data_gpu: BenchmarkData, data_full: BenchmarkData):
        self.name = fixture.name
        self.group = ""  # Fix: support custom group
        self.fullname = fixture.fullname
        self.data_gpu = data_gpu
        self.data_full = data_full
        self.param = fixture.param
        self.params = fixture.params
        self.fixture = fixture

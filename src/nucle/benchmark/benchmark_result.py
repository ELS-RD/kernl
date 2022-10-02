from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from nucle.benchmark.benchmark_data import BenchmarkData
    from nucle.benchmark.benchmark_fixture import BenchmarkFixture


class BenchmarkResult(object):
    def __init__(self, fixture: BenchmarkFixture, data_gpu: BenchmarkData, data_full: BenchmarkData):
        self.name = fixture.name
        self.group = str({i: fixture.params[i] for i in fixture.params if i != "implementation"})
        self.fullname = fixture.fullname
        self.data_gpu = data_gpu
        self.data_full = data_full
        self.param = fixture.param
        self.params = fixture.params
        self.fixture = fixture

    @property
    def func(self) -> str:
        return self.name.split("[")[0]

    @property
    def fullfunc(self) -> str:
        return self.fullname.split("[")[0]

    def to_dict(self):
        return {
            "name": self.name,
            "group": self.group,
            "fullname": self.fullname,
            "func": self.func,
            "fullfunc": self.fullfunc,
            "params": self.params,
            "data_gpu": self.data_gpu.to_dict(),
            "data_full": self.data_full.to_dict(),
        }

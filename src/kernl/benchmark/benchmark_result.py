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

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from kernl.benchmark.benchmark_data import BenchmarkData
    from kernl.benchmark.benchmark_fixture import BenchmarkFixture


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

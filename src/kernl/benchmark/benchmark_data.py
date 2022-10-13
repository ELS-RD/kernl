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

from typing import Iterable

import torch


class BenchmarkData(object):
    def __init__(self, data):
        self.data: torch.FloatTensor = data

    def __bool__(self) -> bool:
        return bool(self.data)

    def __nonzero__(self) -> bool:
        return bool(self.data)

    @property
    def total(self) -> float:
        return torch.sum(self.data).item()

    @property
    def min(self) -> float:
        return torch.min(self.data).item()

    @property
    def max(self) -> float:
        return torch.max(self.data).item()

    @property
    def mean(self) -> float:
        return torch.mean(self.data).item()

    @property
    def quartiles(self) -> Iterable[float]:
        quartiles = torch.quantile(self.data, torch.tensor([0.25, 0.5, 0.75])).tolist()
        return tuple(quartiles)

    @property
    def stddev(self) -> float:
        return torch.std(self.data).item() if len(self.data) > 1 else 0.0

    @property
    def stddev_outliers(self) -> int:
        """
        Count of StdDev outliers: what's beyond (Mean - StdDev, Mean - StdDev)
        """
        count = 0
        q0 = self.mean - self.stddev
        q4 = self.mean + self.stddev
        for val in self.data:
            if val < q0 or val > q4:
                count += 1
        return count

    @property
    def rounds(self) -> int:
        return len(self.data)

    @property
    def median(self) -> float:
        return torch.median(self.data).item()

    @property
    def ops(self) -> float:
        return self.rounds / self.total if self.total > 0.0 else 0.0

    def to_dict(self):
        return {
            "median": self.median,
            "ops": self.ops,
            "q1": self.quartiles[0],
            "q3": self.quartiles[2],
            "rounds": self.rounds,
            "stddev_outliers": self.stddev_outliers,
            "stddev": self.stddev,
            "mean": self.mean,
            "max": self.max,
            "min": self.min,
            "total": self.total,
        }

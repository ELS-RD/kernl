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
    def percentiles(self) -> Iterable[float]:
        percentiles = torch.quantile(self.data, torch.tensor([0.5, 0.2, 0.8])).tolist()
        return tuple(percentiles)

    @property
    def stddev(self) -> float:
        return torch.std(self.data).item() if len(self.data) > 1 else 0.

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
        return self.rounds / self.total if self.total > 0. else 0.

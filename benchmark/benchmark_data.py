from typing import Tuple

import torch


class BenchmarkData(object):
    def __init__(self, data):
        self.data: torch.FloatTensor = data

    def __bool__(self):
        return bool(self.data)

    def __nonzero__(self):
        return bool(self.data)

    @property
    def total(self) -> float:
        return torch.sum(self.data).item()

    @property
    def min(self):
        return torch.min(self.data).item()

    @property
    def max(self) -> float:
        return torch.max(self.data).item()

    @property
    def mean(self) -> float:
        return float(torch.mean(self.data))

    @property
    def percentiles(self) -> Tuple[float]:
        percentiles = torch.quantile(self.data, torch.tensor([0.5, 0.2, 0.8])).tolist()
        return tuple(percentiles)

    @property
    def stddev(self) -> float:
        if len(self.data) > 1:
            return torch.std(self.data).item()
        else:
            return 0.

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
        if self.total:
            return self.rounds / self.total
        return 0.

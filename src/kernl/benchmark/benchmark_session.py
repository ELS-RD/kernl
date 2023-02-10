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

import operator
from collections import defaultdict
from typing import List

import pytest
import torch
from tabulate import tabulate
from termcolor import colored

from kernl.benchmark.benchmark_result import BenchmarkResult


class BenchmarkSession(object):
    def __init__(self, config: pytest.Config):
        self.benchmarks: List[BenchmarkResult] = []
        self.config = config

    @staticmethod
    def print_results(benchmarks: List[BenchmarkResult]):
        compare = []
        for benchmark in benchmarks:
            compare.append(
                [
                    benchmark.data_gpu.median,
                    benchmark.data_gpu.mean,
                    benchmark.data_gpu.min,
                    benchmark.data_gpu.max,
                    benchmark.data_full.median,
                    benchmark.data_full.mean,
                    benchmark.data_full.min,
                    benchmark.data_full.max,
                ]
            )
        tensor = torch.Tensor(compare)
        min_values, min_indexes = torch.min(tensor, dim=0)
        max_values, max_indexes = torch.max(tensor, dim=0)

        table = []
        for benchmark_idx, benchmark in enumerate(benchmarks):
            to_append = [
                benchmark.name,
            ]
            for idx, v in enumerate(compare[benchmark_idx]):
                text = f"{round(v, 4)} ({round(float(max_values[idx] / v), 2)})"
                if benchmark_idx == min_indexes[idx]:
                    text = colored(text, "green")
                if benchmark_idx == max_indexes[idx]:
                    text = colored(text, "red")
                to_append.append(text)
            table.append(to_append)
        print(
            tabulate(
                table,
                headers=[
                    "Name",
                    "Median (CUDA)",
                    "Mean (CUDA)",
                    "Min (CUDA)",
                    "Max (CUDA)",
                    "Median",
                    "Mean",
                    "Min",
                    "Max",
                ],
            )
        )

    def get_groups(self, benchmarks: List[BenchmarkResult], group_by_expression: str):
        groups = defaultdict(list)
        for bench in benchmarks:
            key = ()
            for grouping in group_by_expression.split(","):
                if grouping == "group":
                    key += (bench.group,)
                elif grouping == "name":
                    key += (bench.name,)
                elif grouping == "fullname":
                    key += (bench.fullname,)
                elif grouping == "func":
                    key += (bench.func,)
                elif grouping == "fullfunc":
                    key += (bench.fullfunc,)
                elif grouping == "param":
                    key += (bench.param,)
                elif grouping.startswith("param:"):
                    start_index = len("param:")
                    param_name = grouping[start_index:]
                    key += ("%s=%s" % (param_name, bench.params[param_name]),)
                else:
                    raise NotImplementedError("Unsupported grouping %r." % grouping)
            groups[" ".join(map(str, key))].append(bench)

        for grouped_benchmarks in groups.values():
            grouped_benchmarks.sort(key=operator.attrgetter("fullname" if "full" in group_by_expression else "name"))
        return sorted(groups.items(), key=lambda pair: pair[0] or "")

    def finish(self):
        grouped = self.get_groups(self.benchmarks, self.config.option.benchmark_group_by)
        for group, benchmarks in grouped:
            print("\n" + group)
            self.print_results(benchmarks)

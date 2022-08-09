import pytest
import torch
from time import time_ns
from tabulate import tabulate
from termcolor import colored
from collections import defaultdict
import operator


class BenchmarkData(object):
    fields = (
        "min", "max", "mean", "stddev", "rounds", "median", "stddev_outliers", "ops", "total", "percentiles"
    )

    def __init__(self, data):
        self.data = data

    def __bool__(self):
        return bool(self.data)

    def __nonzero__(self):
        return bool(self.data)

    def as_dict(self):
        return dict(
            (field, getattr(self, field))
            for field in self.fields
        )

    @property
    def total(self):
        return float(torch.sum(self.data))

    @property
    def min(self):
        return float(torch.min(self.data))

    @property
    def max(self):
        return float(torch.max(self.data))

    @property
    def mean(self):
        return float(torch.mean(self.data))

    @property
    def percentiles(self):
        percentiles = torch.quantile(self.data, torch.tensor([0.5, 0.2, 0.8])).tolist()
        return tuple(percentiles)

    @property
    def stddev(self):
        if len(self.data) > 1:
            return float(torch.std(self.data))
        else:
            return 0

    @property
    def stddev_outliers(self):
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
    def rounds(self):
        return len(self.data)

    @property
    def median(self):
        return float(torch.median(self.data))

    @property
    def ops(self):
        if self.total:
            return self.rounds / self.total
        return 0


class BenchmarkResult(object):
    def __init__(self, fixture, data_gpu, data_full):
        self.name = fixture.name
        self.fullname = fixture.fullname
        self.data_gpu = data_gpu
        self.data_full = data_full
        self.param = fixture.param
        self.params = fixture.params
        self.fixture = fixture

    def __bool__(self):
        return bool(self.data)

    def __nonzero__(self):
        return bool(self.data)


class BenchmarkFixture(object):
    _precisions = {}

    def __init__(self, node, add_result, warmup=25, rep=100, grad_to_none=None, group=None):
        self.name = node.name
        self.fullname = node._nodeid
        self.warmup = warmup
        self.rep = rep
        self.grad_to_none = grad_to_none
        self.add_result = add_result
        if hasattr(node, 'callspec'):
            self.param = node.callspec.id
            self.params = node.callspec.params
        else:
            self.param = None
            self.params = None
        self.group = group

    def __call__(self, function_to_benchmark, *args, **kwargs):
        # Estimate the runtime of the function
        function_to_benchmark(*args, **kwargs)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            function_to_benchmark(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        # compute number of warmup and repeat
        n_warmup = max(1, int(self.warmup / estimate_ms))
        n_repeat = max(1, int(self.rep / estimate_ms))
        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2
        # doesn't contain any input data before the run
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
        cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')
        # Warm-up
        for _ in range(n_warmup):
            function_to_benchmark(*args, **kwargs)
        # Benchmark
        for i in range(n_repeat):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if self.grad_to_none is not None:
                for x in self.grad_to_none:
                    x.grad = None
            # we clear the L2 cache before each run
            cache.zero_()
            # record time of `fn`
            start_event[i].record()
            function_to_benchmark(*args, **kwargs)
            end_event[i].record()
        # Record clocks
        torch.cuda.synchronize()
        times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
        gpu_data = BenchmarkData(times)

        cpu_times = []
        # Benchmark
        for i in range(n_repeat):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if self.grad_to_none is not None:
                for x in self.grad_to_none:
                    x.grad = None

            cache.zero_()
            torch.cuda.synchronize()
            start = time_ns()
            function_to_benchmark(*args, **kwargs)
            torch.cuda.synchronize()
            cpu_times.append((time_ns() - start) / 1000000)
        cpu_data = BenchmarkData(torch.Tensor(cpu_times))

        self.add_result(BenchmarkResult(self, gpu_data, cpu_data))


class BenchmarkSession(object):
    def __init__(self, config):
        self.benchmarks = []
        self.config = config

    def print_results(self, benchmarks):
        compare = []
        for benchmark in benchmarks:
            compare.append([
                benchmark.data_gpu.median,
                benchmark.data_gpu.mean,
                benchmark.data_gpu.min,
                benchmark.data_gpu.max,
                benchmark.data_full.median,
                benchmark.data_full.mean,
                benchmark.data_full.min,
                benchmark.data_full.max,
            ])
        tensor = torch.Tensor(compare)
        min_values, min_indexes = torch.min(tensor, dim=0)
        _, max_indexes = torch.max(tensor, dim=0)

        table = []
        for benchmark_idx, benchmark in enumerate(benchmarks):
            to_append = [
                benchmark.name,
            ]
            for idx, v in enumerate(compare[benchmark_idx]):
                text = f"{round(v, 4)} ({round(float(v / min_values[idx]), 2)})"
                if benchmark_idx == min_indexes[idx]:
                    text = colored(text, "green")
                if benchmark_idx == max_indexes[idx]:
                    text = colored(text, "red")
                to_append.append(text)
            table.append(to_append)
        print(tabulate(table,
                       headers=["Name", "Median (CUDA)", "Mean (CUDA)", "Min (CUDA)", "Max (CUDA)", "Median", "Mean",
                                "Min", "Max"]))

    def get_groups(self, benchmarks, group_by_expression):
        groups = defaultdict(list)
        for bench in benchmarks:
            key = ()
            for grouping in group_by_expression.split(','):
                if grouping == "group":
                    key += bench.group,
                elif grouping == "name":
                    key += bench.name,
                elif grouping == "fullname":
                    key += bench.fullname,
                elif grouping == "func":
                    key += bench.name.split("[")[0],
                elif grouping == "fullfunc":
                    key += bench.fullname.split("[")[0],
                elif grouping == "param":
                    key += bench.param,
                elif grouping.startswith("param:"):
                    param_name = grouping[len("param:"):]
                    key += '%s=%s' % (param_name, bench.params[param_name]),
                else:
                    raise NotImplementedError("Unsupported grouping %r." % grouping)
            groups[' '.join(map(str, key))].append(bench)

        for grouped_benchmarks in groups.values():
            grouped_benchmarks.sort(key=operator.attrgetter("fullname" if "full" in group_by_expression else "name"))
        return sorted(groups.items(), key=lambda pair: pair[0] or "")

    def finish(self):
        grouped = self.get_groups(self.benchmarks, self.config.option.benchmark_group_by)
        for (group, benchmarks) in grouped:
            print("\n" + group)
            self.print_results(benchmarks)

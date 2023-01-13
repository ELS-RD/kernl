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

import gc
from time import perf_counter_ns
from typing import Callable, Iterable, Optional

import torch
from _pytest.python import Function

from kernl.benchmark.benchmark_data import BenchmarkData
from kernl.benchmark.benchmark_result import BenchmarkResult


class BenchmarkFixture(object):
    _precisions = {}

    def __init__(
        self,
        node: Function,
        add_result: Callable,
        warmup: int = 25,
        rep: int = 100,
        grad_to_none: Optional[Iterable[torch.Tensor]] = None,
    ):
        self.name = node.name
        self.fullname = node._nodeid
        self.warmup = warmup
        self.rep = rep
        self.grad_to_none = grad_to_none
        self.add_result = add_result
        if hasattr(node, "callspec"):
            self.param = node.callspec.id
            self.params = node.callspec.params
        else:
            self.param = None
            self.params = None

    def __call__(self, function_to_benchmark: Callable, *args, **kwargs):
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
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

        return_value = None
        # Warm-up
        for _ in range(n_warmup):
            return_value = function_to_benchmark(*args, **kwargs)
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

        gc.collect()
        torch.cuda.empty_cache()

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
            start = perf_counter_ns()
            function_to_benchmark(*args, **kwargs)
            torch.cuda.synchronize()
            cpu_times.append((perf_counter_ns() - start) * 1e-6)  # convert to ms
        cpu_data = BenchmarkData(torch.Tensor(cpu_times))
        gc.collect()
        torch.cuda.empty_cache()

        self.add_result(BenchmarkResult(self, gpu_data, cpu_data))

        return return_value

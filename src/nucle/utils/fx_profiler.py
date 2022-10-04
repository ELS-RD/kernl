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

import time
from typing import Any, Dict, List

import numpy as np
import tabulate
import torch
import torch.fx
from torch.fx import GraphModule, Interpreter


# Originaly taken from
# https://github.com/pytorch/tutorials/blob/4918fe4bf3df96b233ca134c985ffb09ea45f92c/intermediate_source/fx_profiling_tutorial.py


class ProfilingInterpreter(Interpreter):
    def __init__(self, module: GraphModule, mode: str = "cpu"):
        super().__init__(module)
        self.mode = mode
        # We are going to store away two things here:
        #
        # 1. A list of total runtimes for ``mod``. In other words, we are
        #    storing away the time ``mod(...)`` took each time this
        #    interpreter is called.
        self.total_runtime_sec: List[float] = []
        # 2. A map from ``Node`` to a list of times (in seconds) that
        #    node took to run. This can be seen as similar to (1) but
        #    for specific sub-parts of the model.
        self.runtimes_sec: Dict[torch.fx.Node, List[float]] = {}

    ######################################################################
    # Next, let's override our first method: ``run()``. ``Interpreter``'s ``run``
    # method is the top-level entrypoint for execution of the model. We will
    # want to intercept this so that we can record the total runtime of the
    # model.

    def run(self, *args) -> Any:
        # Record the time we started running the model
        t_start = time.time()
        # Run the model by delegating back into Interpreter.run()
        return_val = super().run(*args)
        # Record the time we finished running the model
        t_end = time.time()
        # Store the total elapsed time this model execution took in the
        # ProfilingInterpreter
        self.total_runtime_sec.append(t_end - t_start)
        return return_val

    ######################################################################
    # Now, let's override ``run_node``. ``Interpreter`` calls ``run_node`` each
    # time it executes a single node. We will intercept this so that we
    # can measure and record the time taken for each individual call in
    # the model.

    def run_node(self, n: torch.fx.Node) -> Any:
        elapsed_s = 0
        if self.mode == "cpu":
            torch.cuda.synchronize()
            t_start = time.perf_counter_ns()
            return_val = super().run_node(n)
            torch.cuda.synchronize()
            t_end = time.perf_counter_ns()
            elapsed_s = (t_end - t_start) * 1e-9
        if self.mode == "gpu":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            return_val = super().run_node(n)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_s = start_event.elapsed_time(end_event) * 1e-3

        # If we don't have an entry for this node in our runtimes_sec
        # data structure, add one with an empty list value.
        self.runtimes_sec.setdefault(n, [])
        # Record the total elapsed time for this single invocation
        # in the runtimes_sec data structure
        self.runtimes_sec[n].append(elapsed_s)
        return return_val

    ######################################################################
    # Finally, we are going to define a method (one which doesn't override
    # any ``Interpreter`` method) that provides us a nice, organized view of
    # the data we have collected.

    def summary(self, should_sort: bool = False) -> str:
        # Build up a list of summary information for each node
        node_summaries: List[List[Any]] = []
        # Calculate the mean runtime for the whole network. Because the
        # network may have been called multiple times during profiling,
        # we need to summarize the runtimes. We choose to use the
        # arithmetic mean for this.
        mean_total_runtime = np.mean(self.total_runtime_sec)

        # For each node, record summary statistics
        for node, runtimes in self.runtimes_sec.items():
            # Similarly, compute the mean runtime for ``node``
            mean_runtime = np.mean(runtimes)
            # For easier understanding, we also compute the percentage
            # time each node took with respect to the whole network.
            pct_total = mean_runtime / mean_total_runtime * 100
            # Record the node's type, name of the node, mean runtime, and
            # percent runtim
            name = str(node)
            if node.op == "call_module":
                name += str(type(dict(self.module.named_modules())[node.target]))
            node_summaries.append([node.op, name, mean_runtime, pct_total])

        # One of the most important questions to answer when doing performance
        # profiling is "Which op(s) took the longest?". We can make this easy
        # to see by providing sorting functionality in our summary view
        if should_sort:
            node_summaries.sort(key=lambda s: s[2], reverse=True)

        # Use the ``tabulate`` library to create a well-formatted table
        # presenting our summary information
        headers: List[str] = ["Op type", "Op", "Average runtime (s)", "Pct total runtime"]
        return tabulate.tabulate(node_summaries, headers=headers)

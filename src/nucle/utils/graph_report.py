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

import torch
from tabulate import tabulate


def graph_report(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    summary = []
    used_modules = {}
    for n in gm.graph.nodes:
        summary.append([n.op, n.name, n.target, n.args, n.kwargs])
        if n.op == "call_module":
            used_modules[n.target] = modules[n.target]

    return "\n----------\n".join(
        [
            tabulate(summary, headers=["opcode", "name", "target", "args", "kwargs"]),
            "Used modules",
            tabulate([[k, v] for k, v in used_modules.items()], headers=["name", "target_type"]),
        ]
    )

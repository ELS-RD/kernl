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


def remove_dropout(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for n in gm.graph.nodes:
        is_dropout_module = n.op == "call_module" and isinstance(modules[n.target], torch.nn.Dropout)
        is_dropout_function = n.target == torch.nn.functional.dropout
        # If the target matches one of the patterns
        if is_dropout_module or is_dropout_function:
            # Set the insert point, add the new node, and replace all uses
            # of `n` with the new node
            with gm.graph.inserting_after(n):
                # new_node = gm.graph.call_function(torch.nn.Identity, n.args, n.kwargs)
                n.replace_all_uses_with(n.args[0])
            # Remove the old node from the graph
            gm.graph.erase_node(n)
    gm.recompile()

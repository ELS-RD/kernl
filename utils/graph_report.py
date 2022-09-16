import torch
from tabulate import tabulate


def graph_report(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    summary = []
    used_modules = {}
    for n in gm.graph.nodes:
        summary.append(
            [n.op, n.name, n.target, n.args, n.kwargs])
        if n.op == "call_module":
            used_modules[n.target] = modules[n.target]

    return "\n----------\n".join([
        tabulate(summary,
                 headers=['opcode', 'name', 'target', 'args', 'kwargs']),
        "Used modules",
        tabulate([[k, v] for k, v in used_modules.items()],
                 headers=['name', 'target_type']),
    ])

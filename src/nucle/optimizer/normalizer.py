import functools
import operator

import torch


NORMALIZED_OPERATORS = {
    operator.iadd: torch.add,
    operator.iand: torch.bitwise_and,
    operator.ifloordiv: functools.partial(torch.div, rounding_mode="floor"),
    operator.imod: torch.remainder,
    operator.imul: torch.mul,
    operator.imatmul: torch.matmul,
    operator.ior: torch.bitwise_or,
    operator.ipow: torch.pow,
    operator.isub: torch.sub,
    operator.itruediv: torch.div,
    operator.ixor: torch.bitwise_xor,
}


def normalize_operators(gm: torch.fx.GraphModule):
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target in NORMALIZED_OPERATORS:
            with gm.graph.inserting_after(n):
                new_node = gm.graph.call_function(NORMALIZED_OPERATORS[n.target], n.args, n.kwargs)
                n.replace_all_uses_with(new_node)
            gm.graph.erase_node(n)
    gm.recompile()

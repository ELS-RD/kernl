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
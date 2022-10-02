from typing import Iterable

from torch.fx import Node


def compare_static_args(arg1, arg2):
    if type(arg1) != type(arg2):
        return False
    if isinstance(arg1, Node) and isinstance(arg2, Node):
        return True
    if isinstance(arg1, Iterable):
        if len(arg1) != len(arg2):
            return False
        return all([compare_static_args(arg1[j], arg2[j]) for j in range(0, len(arg1))])
    return arg1 == arg2


def static_args_are_equal(pn: Node, gn: Node) -> bool:
    if len(pn.args) != len(gn.args):
        return False
    for i in range(0, len(pn.args)):
        if not compare_static_args(pn.args[i], gn.args[i]):
            return False
    return True

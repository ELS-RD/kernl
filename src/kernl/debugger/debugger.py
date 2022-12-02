import itertools

from kernl.debugger.core import ExecutionContext
from kernl.debugger.memory_map import MemoryMap
import torch
import triton.language as tl

from kernl.debugger.tl_lang import TritonLangProxy


class GridSelector:
    def __init__(self, func):
        self.func = func

    def __getitem__(self, grid):
        return DebuggerFunction(self.func, grid)

    def __call__(self, *args, **kwargs):
        return DebuggerFunction(self.func)(*args, **kwargs)


tl_method_backup = {}


def get_proxy_method(proxy, name):
    method = getattr(proxy, name)

    def fun(*args, **kwarg):
        return method(*args, **kwarg)

    return fun


def attach_triton(module, proxy):
    method_list = [func for func in dir(TritonLangProxy) if callable(getattr(TritonLangProxy, func)) and func[0] != "_"]
    for name in method_list:
        if hasattr(module, name):
            tl_method_backup[name] = getattr(module, name)
            setattr(module, name, get_proxy_method(proxy, name))


def detach_triton(module):
    for (name, method) in tl_method_backup.items():
        setattr(module, name, method)

def program_ids_from_grid(grid):
    iterator = itertools.product(*[range(v) for v in tuple(reversed(grid))])
    return map(lambda v:tuple(reversed(v)), iterator)

class DebuggerFunction:
    def __init__(self, func, grid=(1,)):
        self.func = func
        self.grid = grid

    def __call__(self, *args, **kwargs):
        memory = MemoryMap()

        def convert_arg(arg):
            if torch.is_tensor(arg):
                ptr = memory.add_tensor(arg)
                return torch.tensor([ptr], dtype=torch.int64, device="cuda")
            return arg

        new_args = tuple()
        for arg in args:
            new_args = new_args + (convert_arg(arg),)

        new_kwargs = {}
        for (name, arg) in kwargs:
            new_kwargs[name] = convert_arg(arg)

        for program_id in program_ids_from_grid(self.grid):
            proxy = TritonLangProxy(memory, ExecutionContext(program_id, self.grid))

        attach_triton(tl, proxy)
        self.func(*new_args, **new_kwargs)
        detach_triton(tl)


def triton_debug(func):
    return GridSelector(func)

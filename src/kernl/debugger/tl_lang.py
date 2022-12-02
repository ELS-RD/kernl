import torch

from kernl.debugger.memory_map import MemoryMap

from kernl.debugger.core import ExecutionContext

class TritonLangProxy:
    _memory_map: MemoryMap
    _context: ExecutionContext

    def __init__(self, memory_map: MemoryMap, context: ExecutionContext):
        self._memory_map = memory_map
        self._context = context

    # Types

    int8 = torch.int8
    int16 = torch.int16
    int32 = torch.int32
    int64 = torch.int64
    uint8 = torch.uint8
    bfloat16 = torch.bfloat16
    float32 = torch.float32
    float64 = torch.float64
    float16 = torch.float16

    @property
    def void(self):
        raise NotImplemented()

    @property
    def int1(self):
        raise NotImplemented()

    @property
    def float8(self):
        raise NotImplemented()

    @property
    def uint16(self):
        raise NotImplemented()

    @property
    def uint32(self):
        raise NotImplemented()

    @property
    def uint64(self):
        raise NotImplemented()

    @property
    def pi32_t(self):
        raise NotImplemented()

    # Program functions

    def load(self,
             pointer: torch.Tensor,
             mask: torch.Tensor = None,
             other=0.0,
             cache_modifier='',
             eviction_policy='',
             volatile=False
             ):
        return self._memory_map.load(pointer, mask, other)

    def store(self,
              pointer: torch.Tensor,
              value: torch.Tensor,
              mask=None):
        return self._memory_map.store(pointer, value, mask)

    def program_id(self, axis):
        assert axis < len(self._context.program_id)
        return torch.tensor([self._context.program_id[axis]], dtype=torch.int32, device="cuda")

    def num_programs(self, axis):
        assert axis < len(self._context.program_size)
        return torch.tensor([self._context.program_size[axis]], dtype=torch.int32, device="cuda")

    def arange(self, start, end):
        return torch.arange(start=start, end=end, dtype=torch.int32, device="cuda")

    def zeros(self, shape, dtype):
        return torch.zeros(size=shape, dtype=dtype, device="cuda")

    def dequantize(self, input, scale, shift, nbit, dst_ty=float16):
        raise NotImplemented()

    def broadcast(self, input, other):
        raise NotImplemented()

    def broadcast_to(self, input, shape):
        raise NotImplemented()

    def cat(self, input, shape):
        raise NotImplemented()

    def reshape(self, input, shape):
        raise NotImplemented()

    def dot(self, input, other, trans_a=False, trans_b=False, allow_tf32=True):
        assert input.dtype == other.dtype
        if trans_a:
            input = input.T
        if trans_b:
            other = other.T
        return torch.matmul(input=input, other=other)

    def atomic_cas(self, pointer, cmp, val):
        raise NotImplemented()

    def atomic_xchg(self, pointer, val, mask=None):
        raise NotImplemented()

    def atomic_add(self, pointer, val, mask=None):
        raise NotImplemented()

    def atomic_max(self, pointer, val, mask=None):
        raise NotImplemented()

    def atomic_min(self, pointer, val, mask=None):
        raise NotImplemented()

    def atomic_and(self, pointer, val, mask=None):
        raise NotImplemented()

    def atomic_or(self, pointer, val, mask=None):
        raise NotImplemented()

    def atomic_xor(self, pointer, val, mask=None):
        raise NotImplemented()

    def where(self, condition, x, y):
        return torch.where(condition, x, y)

    def umulhi(self, x, y):
        raise NotImplemented()

    def fdiv(self, x, y, ieee_rounding=False):
        raise NotImplemented()

    def exp(self, x):
        return torch.exp(x)

    def log(self, x):
        return torch.log(x)

    def cos(self, x):
        return torch.cos(x)

    def sin(self, x):
        return torch.sin(x)

    def sqrt(self, x):
        return torch.sqrt(x)

    def globaltimer(self):
        raise NotImplemented()

    def clock(self):
        raise NotImplemented()

    def debug_barrier(self):
        raise NotImplemented()

    def multiple_of(self, input, values):
        return input

    def max_contiguous(self, input, values):
        return input

    def abs(self, x):
        return torch.abs(x)

    def cdiv(self, x, div):
        return (x + div - 1) // div

    def minimum(self, x, y):
        return torch.minimum(x, y)

    def maximum(self, x, y):
        return torch.maximum(x, y)

    def sigmoid(self, x):
        raise NotImplemented()

    def softmax(self, x, ieee_rounding=False):
        raise NotImplemented()

    def ravel(self, x):
        raise NotImplemented()

    def swizzle2d(self, i, j, size_i, size_j, size_g):
        raise NotImplemented()

    def zeros_like(self, input):
        raise NotImplemented()

    def max(self, input, axis=None):
        if axis is None:
            return torch.max(input)
        return torch.max(input, dim=axis).values

    def argmax(self, input, axis):
        raise NotImplemented()

    def min(self, input, axis=None):
        if axis is None:
            return torch.min(input)
        return torch.min(input, dim=axis).values

    def argmin(self, input, axis):
        raise NotImplemented()

    def sum(self, input, axis=None):
        if axis is None:
            return torch.sum(input)
        return torch.sum(input, dim=axis)

    def xor_sum(self, input, axis):
        raise NotImplemented()
import torch
import triton


def type_of(key):
    if isinstance(key, (torch.dtype, triton.language.dtype)):
        ty = {
            torch.bool: "i1",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
            torch.float32: "fp32",
            torch.float64: "fp64",
            torch.uint8: "u8",
            torch.int8: "i8",
            torch.int16: "i16",
            torch.int32: "i32",
            torch.int64: "i64",
            triton.language.uint8: "u8",
            triton.language.uint16: "u16",
            triton.language.uint32: "u32",
            triton.language.uint64: "u64",
            triton.language.float8: "fp8",
        }[key]
        return f"*{ty}"
    if key is None:
        return "*i8"
    assert isinstance(key, str)
    return key


def key_of(arg):
    if hasattr(arg, "dtype"):
        return arg.dtype
    elif isinstance(arg, bool):
        return "i1"
    elif isinstance(arg, int):
        if -(2 ** 31) <= arg and arg <= 2 ** 31 - 1:
            return "i32"
        elif 2 ** 31 <= arg and arg <= 2 ** 32 - 1:
            return "u32"
        elif 2 ** 63 <= arg and arg <= 2 ** 64 - 1:
            return "u64"
        else:
            return "i64"
    elif isinstance(arg, float):
        return "fp32"
    elif arg is None:
        return None
    else:
        raise TypeError(f"Unsupported type {type(arg)} for {arg}")

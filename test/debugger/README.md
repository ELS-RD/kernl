# Triton debugger (EXPERIMENTAL)

## Usage

- Replace `@triton.jit` by `@triton_debug` 
- Replace `@triton.autotune` by `triton_debug_autotune`

## Limitations

- It doesn't follow triton semantics but pytorch ones
- Some triton features are not available
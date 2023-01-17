# Kernel writing conventions

## Variable suffixes

We prefer suffixes to prefixes because:

- avoids looking at the middle of the variable to find the interesting part
- It allows quick comparaison, for example `q = q_ptr + q_offs`, everything starts we `q` and we reduce mistakes
- It follows the pronunciation

Suffixes list:

- `_idx` integer representing an index
- `_size` integer representing a size
- `_off` integer representing an offset
- `_offs` vector or matrix of integers to be used as offset for pointers
- `_ptrs` vector or matrix of pointers
- `_ptr` single pointer
- `_range_offs` output of `tl.arange(0, N)` to be used as offset for pointers
-  `_ptr_mask` bool matrix or vector to be used at mask for load and store operations
- `_LOAD_MASK_NEEDED` boolean, usually used as constexpr to determine if we need to use a mask when loading or storing tensor

## Variable prefixes

- `is_`, `has_` or `should_` for booleans

## Dimension naming

- Dimension is singular
- If dimension follows variable name from a formula. You can use this name. Example MNK for matmul
- Use `col` or `row` singular if you don't have a name for the last two dimensions

## Tensor Stride naming

`p_d_stride`

- `p` is the name of the tensor
- `d` is the name of the dimension

For example:

- `q_batch_stride`
- `output_row_stride`

## Tensor Size naming

`p_d_size`

- `p` is the name of the tensor
- `d` is the name of the dimension

For example:

- `q_batch_size`
- `output_row_size`
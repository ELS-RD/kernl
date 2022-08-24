import torch

from utils.debugger import TritonDebugger

torch.random.manual_seed(123)


def test_add():
    vec_len = 25
    block_size = 64  # not a vec len multiple to test masks
    x = torch.rand(vec_len)
    y = torch.rand_like(x)
    o = torch.zeros_like(x)
    tl = TritonDebugger([TritonDebugger.cdiv(vec_len, block_size)], inputs=[x, y, o])

    def add_kernel(
            x_ptr,  # *Pointer* to first input vector
            y_ptr,  # *Pointer* to second input vector
            output_ptr,  # *Pointer* to output vector
            n_elements,  # Size of the vector
            BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
            # NOTE: `constexpr` so it can be used as a shape value
    ):
        # There are multiple 'program's processing different data. We identify which program
        # we are here
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
        # This program will process inputs that are offset from the initial data.
        # for instance, if you had a vector of length 256 and block_size of 64, the programs
        # would each access the elements [0:64, 64:128, 128:192, 192:256].
        # Note that offsets is a list of pointers
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # Create a mask to guard memory operations against out-of-bounds accesses
        mask = offsets < n_elements

        # Load x and y from DRAM, masking out any extra elements in case the input is not a
        # multiple of the block size
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        # Write x + y back to DRAM
        tl.store(output_ptr + offsets, output, mask=mask)

    while tl.has_next():
        tl.increment()
        add_kernel(
            x_ptr=tl.tensor_ptr[x],
            y_ptr=tl.tensor_ptr[y],
            output_ptr=tl.tensor_ptr[o],
            n_elements=x.numel(),
            BLOCK_SIZE=block_size,
        )
    assert torch.allclose(o, x + y)


def test_softmax():
    vec_len = 250
    x = torch.rand((16, vec_len))
    o = torch.zeros_like(x)
    block_size = 256  # not a vec len multiple to test masks
    tl = TritonDebugger([TritonDebugger.cdiv(x.nelement(), block_size)], inputs=[x, o])

    def softmax_kernel(
            output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
            BLOCK_SIZE: tl.constexpr
    ):
        # The rows of the softmax are independent, so we parallelize across those
        row_idx = tl.program_id(0)
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
        # Substract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

    while tl.has_next():
        tl.increment()
        softmax_kernel(
            output_ptr=tl.tensor_ptr[o],
            input_ptr=tl.tensor_ptr[x],
            input_row_stride=x.stride(0),
            output_row_stride=o.stride(0),
            n_cols=vec_len,
            BLOCK_SIZE=block_size,
        )
    assert torch.allclose(o, torch.softmax(x, dim=1)), f"{o} != {torch.softmax(x, dim=1)}"
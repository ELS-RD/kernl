import torch

from utils.debugger import TritonDebugger

torch.random.manual_seed(123)


def test_launch():
    vec_len = 256
    block_size = 64
    x = torch.rand(vec_len)
    y = torch.rand_like(x)
    o = torch.zeros_like(x)
    tl = TritonDebugger([vec_len // block_size], inputs=[x, y, o])

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
        add_kernel(
            x_ptr=tl.tensor_ptr[x],
            y_ptr=tl.tensor_ptr[y],
            output_ptr=tl.tensor_ptr[o],
            n_elements=x.numel(),
            BLOCK_SIZE=block_size,
        )
    assert torch.allclose(o, x + y)

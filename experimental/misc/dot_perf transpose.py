import torch
import triton
import triton.language as tl


@triton.jit
def kernel(
    Vec,
    Matrix,
    Out,
    vec_batch_stride,
    vec_n_head_stride,
    vec_size_m_stride,
    vec_seqlen_stride,
    matrix_batch_stride,
    matrix_n_head_stride,
    matrix_seqlen_stride,  # 64
    matrix_d_head_stride,  # 1
    out_batch_stride,
    out_n_head_stride,
    out_size_m_stride,  # not used, shape is 1
    out_d_head_stride,  # 1
    D_HEAD: tl.constexpr,
    SIZE_N: tl.constexpr,
):
    n_block_idx = tl.program_id(0)  # 1024 / 16 = 64
    batch_idx = tl.program_id(1)  # 5
    n_head_idx = tl.program_id(2)  # 20

    size_n_arange = tl.arange(0, SIZE_N)
    d_head_arange = tl.arange(0, D_HEAD)
    print("size_n_arange", size_n_arange)
    print("d_head_arange", d_head_arange)
    vec_ptr = (  # QKt
        Vec
        + batch_idx * vec_batch_stride
        + n_head_idx * vec_n_head_stride
        + (n_block_idx * SIZE_N + size_n_arange)[:, None] * vec_seqlen_stride  # multiplication needs a col vector
    )
    vec = tl.load(vec_ptr).to(tl.float32)

    matrix_ptr = (  # V
        Matrix
        + batch_idx * matrix_batch_stride
        + n_head_idx * matrix_n_head_stride
        + d_head_arange[None, :] * matrix_d_head_stride
        + (n_block_idx * SIZE_N + size_n_arange)[:, None] * matrix_seqlen_stride
    )

    mat = tl.load(matrix_ptr).to(tl.float32)
    print("vec", vec)
    print("mat", mat)
    result = vec * mat
    print("result mul", result)
    result = tl.sum(result, axis=0)
    print("result sum", result)
    out_ptr = Out + batch_idx * out_batch_stride + n_head_idx * out_n_head_stride + d_head_arange * out_d_head_stride
    tl.atomic_add(out_ptr, result)


def triton_wrapper(vec: torch.Tensor, matrix: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    # print("matrix_stride", matrix.stride())
    # print("vec_stride", vec.stride())
    _, _, seq_len, d_head = matrix.shape
    SIZE_N = 128
    grid = (triton.cdiv(seq_len, SIZE_N), batch_size, n_head)
    kernel[grid](
        vec,
        matrix,
        output,
        *vec.stride(),
        *matrix.stride(),
        *output.stride(),
        d_head,  # D_HEAD
        SIZE_N,  # SIZE_N
        num_warps=1,
        num_stages=1,
    )
    return output.half()


def reference_pytorch(
    vec: torch.Tensor, matrix: torch.Tensor, output: torch.Tensor, transpose_mat: bool
) -> torch.Tensor:
    if transpose_mat:
        matrix = matrix.transpose(-1, -2)
    return torch.matmul(input=vec, other=matrix, out=output)


batch_size = 5
n_head = 20
seq_len_k = 1024
d_head = 64

cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

out = torch.zeros((batch_size, n_head, 1, seq_len_k), dtype=torch.float16, device="cuda")
out_t = torch.zeros_like(out)
expected = torch.zeros_like(out)


print("CUDA times")
n_repeat = 1000
start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

qkt = torch.randn((batch_size, n_head, 1, seq_len_k), dtype=torch.float16, device="cuda")
v = torch.randn((batch_size, n_head, seq_len_k, d_head), dtype=torch.float16, device="cuda")
out_qktv = torch.zeros((batch_size, n_head, 1, d_head), dtype=torch.float32, device="cuda")
expected_qktv = torch.zeros((batch_size, n_head, 1, d_head), dtype=torch.float16, device="cuda")

# warmup
for _ in range(n_repeat):
    triton_wrapper(vec=qkt, matrix=v, output=out_qktv)
    reference_pytorch(vec=qkt, matrix=v, output=expected_qktv, transpose_mat=False)
    expected_qktv.zero_()
    out_qktv.zero_()

# run PyTorch
for i in range(n_repeat):
    cache.zero_()
    expected_qktv.zero_()
    torch.cuda.synchronize()
    start_event[i].record()
    reference_pytorch(vec=qkt, matrix=v, output=expected_qktv, transpose_mat=False)
    end_event[i].record()

times_pytorch_qktv = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))

# run Triton
for i in range(n_repeat):
    cache.zero_()
    out_qktv.zero_()
    torch.cuda.synchronize()
    start_event[i].record()
    triton_wrapper(vec=qkt, matrix=v, output=out_qktv)
    end_event[i].record()

times_triton_t_qktv = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))


assert torch.allclose(out_qktv.float(), expected_qktv.float(), atol=1e-1), f"{out_qktv}\n{expected_qktv}"
print(f"{'tl.sum(qkt * v, 1)':<20}{times_triton_t_qktv.item() :.9f}")
print(f"{'qkt @ v':<20}{times_pytorch_qktv.item() :.9f}")

# note: for the per-transposed, if size n >= n head, timings are similar to not transposed version.

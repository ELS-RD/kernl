#  Copyright 2022 Lefebvre Sarrut
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import torch

from conftest import assert_all_close, set_seed

from kernl.utils.debugger import TritonDebugger


@set_seed()
def test_add():
    vec_len = 25
    block_size = 64  # not a vec len multiple to test masks
    x = torch.rand(vec_len, device="cuda")
    y = torch.rand_like(x, device="cuda")
    o = torch.zeros_like(x, device="cuda")
    tl = TritonDebugger([TritonDebugger.cdiv(vec_len, block_size)], inputs=[x, y, o], shuffle=True)

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
        tl.new_program()
        add_kernel(
            x_ptr=tl.get_ptr(x),
            y_ptr=tl.get_ptr(y),
            output_ptr=tl.get_ptr(o),
            n_elements=x.numel(),
            BLOCK_SIZE=block_size,
        )
    assert_all_close(o, x + y)
    assert tl.total_gm_read == x.nelement() + y.nelement()
    assert tl.total_gm_write == o.numel()


@set_seed()
def test_softmax():
    ncols = 250
    nrows = 16
    block_ncols = 256  # do not match vec_len to use masks
    x = torch.rand((nrows, ncols), device="cuda")
    o = torch.zeros_like(x, device="cuda")
    tl = TritonDebugger([TritonDebugger.cdiv(x.nelement(), block_ncols)], inputs=[x, o], shuffle=True)

    def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
        # The rows of the softmax are independent, so we parallelize across those
        row_idx = tl.program_id(0)
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
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
        tl.new_program()
        softmax_kernel(
            output_ptr=tl.get_ptr(o),
            input_ptr=tl.get_ptr(x),
            input_row_stride=x.stride(0),
            output_row_stride=o.stride(0),
            n_cols=ncols,
            BLOCK_SIZE=block_ncols,
        )
    assert_all_close(o, torch.softmax(x, dim=1))
    assert tl.total_gm_read == x.nelement()
    assert tl.total_gm_write == o.nelement()


@set_seed()
def test_matmul():
    m, n, k = 16, 4, 32
    assert k % 32 == 0
    block_m, block_n, block_k = 4, 2, 4
    A = torch.rand((m, k), device="cuda", dtype=torch.float16)
    B = torch.rand((k, n), device="cuda", dtype=torch.float16)
    C = torch.zeros((m, n), device="cuda", dtype=torch.float16)
    tl = TritonDebugger(
        [TritonDebugger.cdiv(m, block_m) * TritonDebugger.cdiv(n, block_n)], inputs=[A, B, C], shuffle=True
    )

    def leaky_relu(x):
        x = x + 1
        return tl.where(x >= 0, x, 0.01 * x)

    def matmul_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        # Matrix dimensions
        M,
        N,
        K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
        # by to get the element one row down (A has M rows)
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse
        # See above `L2 Cache Optimizations` section for details
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_n] pointers
        # see above `Pointer Arithmetics` section for details
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            # Note that for simplicity, we don't apply a mask here.
            # This means that if K is not a multiple of BLOCK_SIZE_K,
            # this will access out-of-bounds memory and produce an
            # error or (worse!) incorrect results.
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            # We accumulate along the K dimension
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        # you can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        if ACTIVATION == "leaky_relu":
            accumulator = leaky_relu(accumulator)
        c = accumulator.to(tl.float16)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    while tl.has_next():
        tl.new_program()
        matmul_kernel(
            # Pointers to matrices
            a_ptr=tl.get_ptr(A),
            b_ptr=tl.get_ptr(B),
            c_ptr=tl.get_ptr(C),
            # Matrix dimensions
            M=m,
            N=n,
            K=k,
            # The stride variables represent how much to increase the ptr by when moving by 1
            # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
            # by to get the element one row down (A has M rows)
            stride_am=A.stride(0),
            stride_ak=A.stride(1),
            stride_bk=B.stride(0),
            stride_bn=B.stride(1),
            stride_cm=C.stride(0),
            stride_cn=C.stride(1),
            # Meta-parameters
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=8,
            ACTIVATION="leaky_relu",
        )

    def leaky_relu_pytorch(x):
        x = x + 1
        return torch.where(x >= 0, x, 0.01 * x)

    assert_all_close(C, leaky_relu_pytorch(A @ B), atol=1e-1)
    assert tl.total_gm_write == m * n
    # we load tile a and tile b for each position on M and N, and repeat along K axis
    assert tl.total_gm_read == ((block_m * block_k) + (block_k * block_n)) * (k / block_k) * (n / block_n) * (
        m / block_m
    )


@set_seed()
def test_dropout():
    p = 0.5
    x = torch.randn(size=(10, 1000), device="cuda")
    o = torch.zeros_like(x)
    block_m = 32
    tl = TritonDebugger([TritonDebugger.cdiv(x.numel(), block_m)], inputs=[x, o], shuffle=True)

    def _seeded_dropout(
        x_ptr,
        output_ptr,
        n_elements,
        p,
        seed,
        BLOCK_SIZE: tl.constexpr,
    ):
        # compute memory offsets of elements handled by this instance
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # load data from x
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # randomly prune it
        random = tl.rand(seed, offsets)
        x_keep = random > p
        # write-back
        output = tl.where(x_keep, x / (1 - p), 0.0)
        tl.store(output_ptr + offsets, output, mask=mask)

    while tl.has_next():
        tl.new_program()
        _seeded_dropout(
            x_ptr=tl.get_ptr(x),
            output_ptr=tl.get_ptr(o),
            n_elements=x.numel(),
            p=p,
            seed=123,
            BLOCK_SIZE=block_m,
        )

    assert_all_close(torch.sum(o == 0) / x.numel(), torch.tensor(p, device="cuda"), atol=0.1)
    # check L1 norm are similar (+/- 10%)
    assert_all_close(torch.linalg.norm(x, dim=1, ord=1), torch.linalg.norm(o, dim=1, ord=1), rtol=0.1)
    assert tl.total_gm_read == x.numel()
    assert tl.total_gm_write == o.numel() == x.numel()


@set_seed()
def test_layernorm():
    M, N = 32, 64
    BLOCK_SIZE = 16  # need to be a power of 2
    x_shape = (M, N)
    w_shape = (N,)
    weight = torch.rand(w_shape, device="cuda")
    bias = torch.rand(w_shape, device="cuda")
    x = -2.3 + 0.5 * torch.randn(x_shape, device="cuda")
    dy = 0.1 * torch.randn_like(x)

    out = torch.zeros_like(x)
    mean = torch.zeros((M,), device="cuda")
    rstd = torch.zeros((M,), device="cuda")
    eps = 1e-5

    tl = TritonDebugger([M], inputs=[x, weight, bias, dy, mean, rstd, out], shuffle=True)

    def _layer_norm_fwd_fused(
        Out,
        A,
        Weight,
        Bias,
        Mean,
        Rstd,
        stride,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # position of elements processed by this program
        row = tl.program_id(0)
        Out += row * stride
        A += row * stride
        # compute mean
        mean = 0
        _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(A + cols, mask=cols < N, other=0.0, eviction_policy="evict_last").to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=0) / N
        # compute variance
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(A + cols, mask=cols < N, other=0.0, eviction_policy="evict_last").to(tl.float32)
            a = tl.where(cols < N, a - mean, 0.0)
            _var += a * a
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)

        # write-back mean/rstd
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)
        # multiply by weight and add bias
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            weight = tl.load(Weight + cols, mask=mask)
            bias = tl.load(Bias + cols, mask=mask)
            a = tl.load(A + cols, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
            a_hat = (a - mean) * rstd
            out = a_hat * weight + bias
            # # write-back
            tl.store(Out + cols, out, mask=mask)

    while tl.has_next():
        tl.new_program()
        _layer_norm_fwd_fused(
            Out=tl.get_ptr(out),
            A=tl.get_ptr(x),
            Weight=tl.get_ptr(weight),
            Bias=tl.get_ptr(bias),
            Mean=tl.get_ptr(mean),
            Rstd=tl.get_ptr(rstd),
            stride=x.stride(0),
            N=N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    assert_all_close(mean, torch.mean(x, dim=1), atol=0.1)
    assert_all_close(rstd, 1 / torch.sqrt(torch.var(x, dim=1, unbiased=False) + eps), atol=0.1)
    assert_all_close(
        out, torch.layer_norm(input=x, normalized_shape=w_shape, weight=weight, bias=bias, eps=eps), atol=0.1
    )
    # read M times a block size of the 5 inputs
    assert tl.total_gm_read == M * (5 * BLOCK_SIZE * (N / BLOCK_SIZE))
    # mean + std + output
    assert tl.total_gm_write == M + M + M * N


@set_seed()
def test_layernorm_welford_variance():
    import torch

    M, N = 2, 200
    BLOCK_SIZE = 16  # need to be a power of 2
    x_shape = (M, N)
    w_shape = (N,)
    weight = torch.rand(w_shape, device="cuda")
    bias = torch.rand(w_shape, device="cuda")
    x = -2.3 + 0.5 * torch.randn(x_shape, device="cuda")
    dy = 0.1 * torch.randn_like(x)

    out = torch.zeros_like(x)
    mean = torch.zeros((M,), device="cuda")
    rstd = torch.zeros((M,), device="cuda")
    eps = 1e-5

    tl = TritonDebugger([M], inputs=[x, weight, bias, dy, mean, rstd, out], shuffle=False)

    def _layer_norm_fwd_fused(
        Out,
        A,
        Weight,
        Bias,
        Mean,
        Rstd,
        stride,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # position of elements processed by this program
        row = tl.program_id(0)
        Out += row * stride
        A += row * stride
        # compute mean
        mean = 0.0
        var = 0.0
        # _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for start in range(0, N, BLOCK_SIZE):
            end = min((start + BLOCK_SIZE), N)
            nb_block_col = end - start
            cols = start + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            a = tl.load(A + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)

            old_mean = mean
            block_mean = tl.sum(a * mask, axis=0) / nb_block_col
            block_var = tl.sum((a - block_mean) * a * mask, axis=0)

            old_var = var
            mean = old_mean + tl.sum((a - old_mean) * mask, axis=0) / end

            delta = block_mean - old_mean
            delta2 = delta * delta
            var = old_var + block_var + delta2 * (start * nb_block_col) / end

        var = var / N

        rstd = 1 / tl.sqrt(var + eps)

        # write-back mean/rstd
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)
        # multiply by weight and add bias
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            weight = tl.load(Weight + cols, mask=mask)
            bias = tl.load(Bias + cols, mask=mask)
            a = tl.load(A + cols, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
            a_hat = (a - mean) * rstd
            out = a_hat * weight + bias
            # # write-back
            tl.store(Out + cols, out, mask=mask)

    while tl.has_next():
        tl.new_program()
        _layer_norm_fwd_fused(
            Out=tl.get_ptr(out),
            A=tl.get_ptr(x),
            Weight=tl.get_ptr(weight),
            Bias=tl.get_ptr(bias),
            Mean=tl.get_ptr(mean),
            Rstd=tl.get_ptr(rstd),
            stride=x.stride(0),
            N=N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    assert_all_close(mean, torch.mean(x, dim=1), atol=0.1)
    assert_all_close(rstd, 1 / torch.sqrt(torch.var(x, dim=1, unbiased=False) + eps), atol=0.1)
    assert_all_close(
        out, torch.layer_norm(input=x, normalized_shape=w_shape, weight=weight, bias=bias, eps=eps), atol=0.1
    )
    # read M times a block size of the 5 inputs
    assert tl.total_gm_read == M * (4 * BLOCK_SIZE * (N / BLOCK_SIZE))
    # mean + std + output
    assert tl.total_gm_write == M + M + M * N


@set_seed()
def test_flash_attention():
    Z, H, N_CTX, D_HEAD = 3, 2, 2048, 64
    q = torch.empty((Z, H, N_CTX, D_HEAD), device="cuda", dtype=torch.float16).normal_(mean=0, std=0.5)
    k = torch.empty((Z, H, N_CTX, D_HEAD), device="cuda", dtype=torch.float16).normal_(mean=0, std=0.5)
    v = torch.empty((Z, H, N_CTX, D_HEAD), device="cuda", dtype=torch.float16).normal_(mean=0, std=0.5)
    sm_scale = 0.3
    dout = torch.randn_like(q).float()
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p, dim=-1)
    ref_out = torch.matmul(p, v)
    tmp = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device="cuda", dtype=torch.float32)

    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device="cuda")
    m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device="cuda")

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    BLOCK = 128
    tl = TritonDebugger(
        [TritonDebugger.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1]],
        inputs=[q, k, v, dout, L, m, tmp],
        shuffle=True,
    )

    def _fwd_kernel(
        Q,
        K,
        V,
        sm_scale,
        TMP,
        L,
        M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
        Out,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        Z,
        H,
        N_CTX,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        off_k = off_hz * stride_qh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
        # Initialize pointers to Q, K, V
        q_ptrs = Q + off_q
        k_ptrs = K + off_k
        v_ptrs = V + off_v
        # initialize pointer to m and l
        t_ptrs = TMP + off_hz * N_CTX + offs_m
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # load q: it will stay in SRAM throughout
        q = tl.load(q_ptrs)
        # loop over k, v and update accumulator
        for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs + start_n * stride_kn)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k, trans_b=True)
            qk *= sm_scale
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            tl.store(t_ptrs, acc_scale)
            acc_scale = tl.load(t_ptrs)  # BUG: have to store and immediately load
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs + start_n * stride_vk)
            p = p.to(tl.float16)
            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        # rematerialize offsets to save registers
        start_m = tl.program_id(0)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # write back l and m
        l_ptrs = L + off_hz * N_CTX + offs_m
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, l_i)
        tl.store(m_ptrs, m_i)
        # initialize pointers to output
        offs_n = tl.arange(0, BLOCK_DMODEL)
        off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_ptrs = Out + off_o
        tl.store(out_ptrs, acc)

    while tl.has_next():
        tl.new_program()
        _fwd_kernel(
            tl.get_ptr(q),
            tl.get_ptr(k),
            tl.get_ptr(v),
            sm_scale,
            tl.get_ptr(tmp),
            tl.get_ptr(L),
            tl.get_ptr(m),
            tl.get_ptr(dout),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk,
        )

    assert_all_close(dout, ref_out.float(), atol=1e-2)

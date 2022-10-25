import time

import deepspeed
import torch
from transformers import AutoModel


deepspeed.init_distributed("nccl")

model = AutoModel.from_pretrained(pretrained_model_name_or_path="bert-base-uncased").eval().cuda()
# full conversion to fp16
model = model.half()


for shape in [(bs, seq_l) for bs in [1, 8, 32] for seq_l in [16, 32, 64, 128, 256, 384, 512] if bs * seq_l < 10000]:
    inputs = {
        "input_ids": torch.randint(10, 10000, shape, device="cuda", dtype=torch.long),
        "attention_mask": torch.ones(shape, device="cuda", dtype=torch.long),
    }

    # rebuild to use CUDA graphs
    model_ds = deepspeed.init_inference(
        model,
        dtype=torch.float16,
        mp_size=1,
        replace_with_kernel_inject=True,
        replace_method="auto",
        enable_cuda_graph=True,
    )
    model_ds.profile_model_time()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        output_torch = model(**inputs)
    output_deepspeed = model_ds(**inputs)
    assert torch.allclose(output_deepspeed.last_hidden_state.float(), output_torch.last_hidden_state, 1e-1, 1e-1)

    # warmup
    for _ in range(10):
        model_ds(**inputs)

    timings = list()
    torch.cuda.synchronize()
    for _ in range(10):
        start = time.perf_counter()
        model_ds(**inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append(end - start)

    print(f"{shape}: {torch.median(torch.tensor(timings)):.4f}")

import time

import torch
import torch._dynamo as torchdynamo
from transformers import AutoModel


model = AutoModel.from_pretrained(pretrained_model_name_or_path="bert-base-uncased").eval().cuda()


@torchdynamo.optimize("inductor")
def apply(*args, **kwargs):
    return model(*args, **kwargs)


for shape in [(bs, seq_l) for bs in [1, 8, 32] for seq_l in [16, 32, 64, 128, 256, 384, 512] if bs * seq_l < 10000]:
    inputs = {
        "input_ids": torch.randint(10, 10000, shape, device="cuda", dtype=torch.long),
        "attention_mask": torch.ones(shape, device="cuda", dtype=torch.long),
    }

    timings = list()

    # warmup
    # torch.inference_mode doesn't work with dynamo+inductor
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        for _ in range(10):
            output_inductor = apply(**inputs)

        output_torch = model(**inputs)
        assert torch.allclose(output_inductor.last_hidden_state, output_torch.last_hidden_state, 1e-1, 1e-1)

        torch.cuda.synchronize()
        for _ in range(10):
            start = time.perf_counter()
            apply(**inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

        torchdynamo.reset()

    print(f"{shape}: {torch.median(torch.tensor(timings)):.4f}")

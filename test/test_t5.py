from typing import List

import torch
import torchdynamo
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.kernl.optimizer.dynamo_backend import dynamo_backend_ofi


def test_t5():
    torchdynamo.config.cache_size_limit = 512
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model = model.eval().cuda()
    task = "translate English to French: The house is wonderful."
    inputs = tokenizer(task, return_tensors="pt", padding=True).to("cuda")

    with torch.inference_mode(), torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            min_length=20,
            max_length=20,
            do_sample=False,
        )
        assert "La maison est merveilleuse." in tokenizer.batch_decode(output_sequences, skip_special_tokens=True)[0]

    model.forward2 = model.forward
    model.encoder.forward2 = model.encoder.forward
    model.decoder.forward2 = model.decoder.forward

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        return dynamo_backend_ofi(gm)

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return model.forward2(*args, **kwargs)

    def run_encoder(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return model.encoder.forward2(*args, **kwargs)

    def run_decoder(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return model.decoder.forward2(*args, **kwargs)

    model.forward = run
    model.encoder.forward = run_encoder
    model.decoder.forward = run_decoder

    with torch.inference_mode(), torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            min_length=20,
            max_length=20,
            do_sample=False,
        )
        assert "La maison est merveilleuse." in tokenizer.batch_decode(output_sequences, skip_special_tokens=True)[0]

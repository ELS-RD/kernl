from test.models.bert import get_model_optimized

import torch
import torchdynamo
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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

    model.encoder.forward = get_model_optimized(model.encoder.forward)
    model.decoder.forward = get_model_optimized(model.decoder.forward)

    with torch.inference_mode(), torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            min_length=20,
            max_length=20,
            do_sample=False,
        )
        assert "La maison est merveilleuse." in tokenizer.batch_decode(output_sequences, skip_special_tokens=True)[0]

import time

import torch
import torch._dynamo as torchdynamo
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from kernl.model_optimization import optimize_model


torchdynamo.config.cache_size_limit = 512

audio_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")


@staticmethod
def fix_reorder_cache(past, beam_idx):
    reordered_past = ()
    for layer_past in past:
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
        )
    return reordered_past


WhisperForConditionalGeneration._reorder_cache = fix_reorder_cache


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to("cuda").eval()

model.model.decoder.forward_before = model.model.decoder.forward


def wrapper_stride(*args, **kwargs):
    o = model.model.decoder.forward_before(*args, **kwargs)
    past_key_values = list(o.past_key_values)
    for idx in range(len(past_key_values)):
        layer = past_key_values[idx]
        v = layer[-1]
        if v.stride(2) != 1:
            v_col_major = v.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
            past_key_values[idx] = layer[:-1] + (v_col_major,)

    o.past_key_values = tuple(past_key_values)

    return o


model.model.decoder.forward = wrapper_stride


optimize_model(model.model.decoder)

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

max_len = 50
timing = list()
inputs_1 = processor(audio_dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=16_000).input_features.to(
    "cuda"
)
with torch.inference_mode(), torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
    model.generate(inputs_1, min_length=max_len, max_length=max_len, num_beams=5, do_sample=False)
    torch.cuda.synchronize()
    for audio in audio_dataset:
        inputs = processor(audio["audio"]["array"], return_tensors="pt", sampling_rate=16_000).input_features.to("cuda")
        torch.cuda.synchronize()
        start = time.time()
        predicted_ids = model.generate(inputs, min_length=1, max_length=max_len, num_beams=5, do_sample=False)
        torch.cuda.synchronize()
        timing.append(time.time() - start)
        print(time.time() - start)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)[0]
        print(transcription)

print(f"Average time: {sum(timing) / len(timing)}", f"Complete time: {sum(timing)}")
print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

import time

import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from kernl.model_optimization import optimize_model


torch.set_float32_matmul_precision("high")
# torchdynamo.config.cache_size_limit = 512
# torchdynamo.config.dynamic_shapes = True
max_len = 50
num_beams = 5
model_name = "openai/whisper-large-v2"

audio_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# audio_dataset = load_dataset("librispeech_asr", "clean", split="test")


def get_tokens(item: dict[str, dict]) -> torch.Tensor:
    tensor = processor(item["audio"]["array"], return_tensors="pt", sampling_rate=16_000).input_features
    return tensor.cuda()


processor = WhisperProcessor.from_pretrained(model_name)
inputs_1 = get_tokens(audio_dataset[0])

model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda").eval()

timings_original = list()
transcriptions = list()
with torch.inference_mode(), torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
    # warmup
    model.generate(inputs_1, min_length=max_len, max_length=max_len, num_beams=num_beams, do_sample=False)
    torch.cuda.synchronize()
    for audio in audio_dataset:
        inputs = get_tokens(audio)
        torch.cuda.synchronize()
        start = time.time()
        predicted_ids = model.generate(inputs, min_length=1, max_length=max_len, num_beams=num_beams, do_sample=False)
        torch.cuda.synchronize()
        timings_original.append(time.time() - start)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)[0]
        transcriptions.append(transcription)

assert len(audio_dataset) == len(transcriptions)


# apply efficiency fix to HuggingFace implementation of whisper to limit memory footprint
@staticmethod
def fix_reorder_cache(past, beam_idx):
    reordered_past = ()
    for layer_past in past:
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
        )
    return reordered_past


WhisperForConditionalGeneration._reorder_cache = fix_reorder_cache

model.model.decoder.forward_before = model.model.decoder.forward

optimize_model(model.model.decoder)
nb_diff = 0
timings_optimized = list()
print("difference between original and optimized model:")
with torch.inference_mode(), torch.autocast(dtype=torch.float16, cache_enabled=True, device_type="cuda"):
    start = time.time()
    model.generate(inputs_1, min_length=max_len, max_length=max_len, num_beams=num_beams, do_sample=False)
    torch.cuda.synchronize()
    print(f"time to warmup: {time.time() - start:.2f}s")
    for original_modem_transcription, audio in zip(transcriptions, audio_dataset):
        inputs = get_tokens(audio)
        torch.cuda.synchronize()
        start = time.time()
        predicted_ids = model.generate(inputs, min_length=1, max_length=max_len, num_beams=num_beams, do_sample=False)
        torch.cuda.synchronize()
        timings_optimized.append(time.time() - start)
        optimized_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)[0]
        nb_diff += original_modem_transcription != optimized_transcription

print("timings")
print(f"[original] average: {sum(timings_original) / len(timings_original)}s / complete: {sum(timings_original)}s")
print(f"[optimized] average: {sum(timings_optimized) / len(timings_optimized)}s / complete: {sum(timings_optimized)}s")
print(f"output differences: {nb_diff}/{len(audio_dataset)} ({nb_diff / len(audio_dataset) * 100:.2f}%)")

print("memory footprint")
print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

# before modification without stride fix
# difference between original and optimzed model:
# optimized model modifies 31.83% (# 834 examples) of the dataset
# [original] average: 1.0774035485646196s / complete: 2822.7972972393036s
# [optimized] average: 0.45561475826583747s / complete: 1193.7106666564941s
# torch.cuda.memory_allocated: 10.874530GB
# torch.cuda.memory_reserved: 14.064453GB
# torch.cuda.max_memory_reserved: 14.064453GB
# after modification
# difference between original and optimzed model:
# time to warmup: 694.1382637023926
# optimized model modifies 2.06% (# 54/2620 examples)
# [original] average: 1.0491925114893732s / complete: 2748.8843801021576s
# [optimized] average: 0.4339728889574531s / complete: 1137.0089690685272s
# torch.cuda.memory_allocated: 10.873960GB
# torch.cuda.memory_reserved: 13.365234GB
# torch.cuda.max_memory_reserved: 13.853516GB

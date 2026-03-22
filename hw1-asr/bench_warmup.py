"""
Warmup-aware benchmark: 2 warmup passes to populate autotune cache,
then 3 clean measurement passes. Reports mean and min.
Usage: python bench_warmup.py <folder>  (e.g. glm_asr_triton_template)
"""
import sys
import os
import time
import numpy as np
import soundfile as sf

folder = sys.argv[1] if len(sys.argv) > 1 else "glm_asr_triton_template"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), folder))

AUDIO_PATH = os.path.join(os.path.dirname(__file__), "test_audio.wav")

from weight_loader import load_model_from_hf

print(f"Loading model from {folder}...")
model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

audio, sr = sf.read(AUDIO_PATH)
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

import torch
input_features = inputs["input_features"]
input_ids = inputs.get("input_ids", None)

# Move to GPU
if hasattr(input_features, "cuda"):
    input_features = input_features.cuda()

WARMUP = 2
MEASURE = 3

def run_inference():
    """Run full pipeline and return elapsed ms."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(input_features, max_new_tokens=60)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0

print(f"\nRunning {WARMUP} warmup passes (autotune cache fill)...")
for i in range(WARMUP):
    t = run_inference()
    print(f"  Warmup {i+1}: {t:.1f}ms")

print(f"\nRunning {MEASURE} measurement passes...")
times = []
for i in range(MEASURE):
    t = run_inference()
    times.append(t)
    print(f"  Run {i+1}: {t:.1f}ms")

print(f"\n{'='*50}")
print(f"Folder : {folder}")
print(f"Mean   : {np.mean(times):.1f}ms")
print(f"Min    : {np.min(times):.1f}ms")
print(f"Std    : {np.std(times):.1f}ms")
print(f"{'='*50}")

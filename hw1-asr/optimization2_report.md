# Optimization 2: Kernel Fusion

**Hardware:** NVIDIA L4 (Ada Lovelace, sm_89, 60 SMs, 24 GB GDDR6, 48 MB L2)
**Framework:** Triton (Python)
**Files modified:** `glm_asr_triton_template/attention.py`, `glm_asr_triton_template/layers.py`

---

## 1. Background: Why Kernel Fusion Matters

Every time a GPU kernel finishes, its output lives in global memory (GDDR6 on the L4). The next kernel must read it back from global memory to continue the computation. This round-trip — write from registers to GDDR6, then read from GDDR6 back into registers — costs bandwidth and latency even when the intermediate result is used immediately and never needed again.

For modern transformers this is the dominant overhead in multi-stage pipelines like attention and linear-with-activation, where the logical computation is: `output = f(g(x))`. The baseline implementation runs two separate kernels:
1. Kernel A: `tmp = g(x)` — writes `tmp` to global memory
2. Kernel B: `output = f(tmp)` — reads `tmp` back, writes `output`

A fused kernel runs the same computation in a single pass: load `x`, compute `g(x)` in registers, immediately apply `f(·)`, store `output`. The intermediate `tmp` never touches global memory. This is not a numerical change — results are bit-identical — it is purely a memory-traffic reduction.

Additionally, each separate kernel launch incurs a fixed CPU-side submission overhead (~5–10 µs on CUDA) plus a GPU-side scheduling delay. Fusing N kernels into 1 eliminates N−1 of those overheads. For a model with 60 transformer layers (32 encoder + 28 decoder) that each run 3–4 separate small kernels, this adds up to hundreds of microseconds per inference.

---

## 2. What the Baseline Code Does (and Why It's Wasteful)

### 2.1 Attention Pipeline — Four Kernels, Three Intermediate Tensors

The baseline `scaled_dot_product_attention` in `attention.py` splits the computation into four sequential steps, each writing to global memory before the next step can begin:

```
Step 1: attention_scores_kernel
    Input:  Q (BH×seq_q×D), K (BH×seq_k×D)           (read from global memory)
    Output: scores (BH×seq_q×seq_k)                    (written to global memory)

Step 2: PyTorch causal mask addition
    Input:  scores (BH×seq_q×seq_k)                    (read from global memory)
    Output: masked_scores (BH×seq_q×seq_k)              (written to global memory)

Step 3: softmax_inplace_kernel
    Input:  masked_scores (BH×seq_q×seq_k)              (read from global memory)
    Output: attn_weights (BH×seq_q×seq_k)               (written in-place)

Step 4: attention_output_kernel
    Input:  attn_weights (BH×seq_q×seq_k), V (BH×seq_k×D)  (read from global memory)
    Output: context (BH×seq_q×D)                        (written to global memory)
```

The `scores` tensor passes through global memory **three times** (written in step 1, read+written in step 2, read+written in step 3) before it is finally consumed in step 4. These are pure overhead — the data never needs to leave the chip after the initial computation.

**Concrete sizes for the text decoder on this model:**
- seq_k ≈ 100–130 tokens (93 projected audio tokens + generated tokens for this audio)
- 28 Q heads, head_dim = 128
- scores tensor per layer: 28 × 1 × 128 × 4B ≈ 14 KB per decode step
- Over 28 decoder layers: 28 × 14 KB × 3 roundtrips ≈ **1.2 MB of avoidable global memory traffic per generated token**
- Over 13 generated tokens: **~16 MB** total

In addition, the original code **allocates padded copies** of Q, K, and V when their dimensions are not already powers of two. For example, if `seq_k = 100`, the code allocates a `seq_k_padded = 128` tensor, copies the data in, runs the kernel with BLOCK_K=128, then discards the padded buffer. This allocation-and-copy is entirely unnecessary because the Triton kernel already handles boundary conditions via `tl.load(..., mask=..., other=0.0)`.

### 2.2 Linear Bias Addition — Separate PyTorch Op After Every Kernel

The `Linear._forward_triton` method calls `linear_kernel_tf32` to compute the matmul and then adds the bias as a **separate PyTorch tensor operation**:

```python
# Before Optimization 2
linear_kernel_tf32[grid](x_2d, weight_t, output, M, N, K, ...)  # writes output

if self.has_bias:
    output = output + self.bias_param   # reads output, adds bias, writes output again
```

This post-kernel bias addition requires:
1. Reading the entire `output` tensor (M × N floats) from global memory
2. Loading the bias vector (N floats) from global memory
3. Writing the updated `output` back to global memory

All audio encoder linear layers use `bias=True`:
- 4 attention projections (Q, K, V, out) × 32 encoder layers = **128 separate bias additions**
- 2 MLP projections (fc1, fc2) × 32 encoder layers = **64 separate bias additions**

That is **192 avoidable global memory roundtrips** for the full audio encoding pass, each touching a tensor of size (M × N) = (375 × 1280) floats ≈ 1.9 MB. Even at L2 cache hit rates, this is measurable overhead.

---

## 3. The Fixes: Two Independent Fusions

### Fusion 1 — Full Attention Kernel (`fused_attention_kernel`)

A single Triton kernel replaces all four attention steps. The key insight is that `attention_scores_kernel` already loads all of K into registers (as a BLOCK_K × BLOCK_D matrix) and computes a `scores` vector of length BLOCK_K. At this point, the scores live entirely in GPU registers — there is no reason to store them to global memory before masking and normalizing.

The fused kernel, for each `(batch_head, query_position)` tile:

```
1. Load Q[pid_q]                               (head_dim floats → registers)
2. Load K[0:seq_k, :]                          (seq_k × head_dim floats → registers)
3. scores = (K @ Q) * scale                    (seq_k floats, entirely in registers)
4. Mask padding:   scores[j ≥ seq_k] = −∞      (in registers, zero cost)
5. Causal mask:    scores[j > threshold] = −∞  (in registers, zero cost)
6. Softmax:        max → subtract → exp → sum → divide  (in registers)
7. Load V[0:seq_k, :]                          (seq_k × head_dim floats → registers)
8. out = Σ_j attn_weights[j] * V[j, :]        (head_dim floats, in registers)
9. Store out                                   (first and only write to global memory)
```

Steps 3–6 never touch global memory. The raw scores and attention weights exist only as register arrays while the corresponding K and V data are also in registers (or L1/L2 cache). A single global memory write at step 9 produces the final output.

**Causal mask formula in the fused kernel:**

The causal threshold must handle two regimes without branching on `seq_q`:

```
threshold = seq_k − seq_q + pid_q
```

- **Audio encode** (bidirectional, `is_causal=False`): no masking applied at all
- **Text decode** (`seq_q=1`, `seq_k=N`, `pid_q=0`): threshold = `N − 1` → nothing is masked → the single query attends to all N cached keys ✓
- **Text prefill** (`seq_q=N`, `seq_k=N`, `pid_q=i`): threshold = `i` → mask keys `j > i` ✓

This single formula correctly handles the KV-cache decode case where `seq_q ≠ seq_k` without any special-casing.

**Padding removal:** The original code allocated padded copies of Q, K, V to round their dimensions up to powers of two before calling the kernel. The fused kernel receives the original unpadded tensors and uses `mask=(key_idx < seq_k) & (col_idx < head_dim)` in every `tl.load`, so boundary conditions are handled naturally. `seq_k_padded` and `head_dim_padded` are still computed (as Triton `constexpr` block sizes) but no allocation or data copy is needed.

**Applicability:** The fused kernel is used when `seq_k_padded ≤ MAX_ATTENTION_DIM = 256` and no external `attention_mask` tensor is provided. For this model:
- **Text decoder:** seq_k ≈ 100–130 → padded to 128 ≤ 256 → **fused kernel active**
- **Audio encoder:** seq_k ≈ 375 → padded to 512 > 256 → falls back to PyTorch einsum (cuBLAS handles large attention efficiently)

The fallback path (PyTorch einsum) is retained unchanged for larger sequences and external masks.

### Fusion 2 — Bias Folding into `linear_kernel_tf32`

Two new parameters are added to the matmul kernel:

```python
@triton.autotune(configs=autotune_configs, key=["M", "N", "K"])
@triton.jit
def linear_kernel_tf32(
    a_ptr, b_ptr, c_ptr,
    bias_ptr,               # ← new: pointer to bias vector (N,)
    M, N, K,
    ...,
    HAS_BIAS: tl.constexpr, # ← new: compile-time flag
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    ...
    # accumulate matmul tiles as before
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        ...
        acc += tl.dot(a, b)

    # Fused bias addition — in registers, before the single store
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]   # broadcast over the M dimension

    tl.store(c_ptrs, acc, mask=mask_c)
```

When `HAS_BIAS=True`, the bias vector is loaded once per output tile (N floats, which is the same data the `output + bias_param` PyTorch op reads anyway) and added to the accumulator in registers before the output is stored. The output tensor is written exactly once.

When `HAS_BIAS=False` (text decoder projections, MLP layers), Triton specializes the kernel at compile time with the `if HAS_BIAS` branch completely removed — there is no runtime cost for the flag check.

**`Linear._forward_triton` after the change:**

```python
has_bias = self.has_bias and self.bias_param is not None
bias_t = self.bias_param if has_bias else x_2d  # dummy pointer never dereferenced

linear_kernel_tf32[grid](
    x_2d, self._weight_t_padded, output, bias_t,
    M, N, K, ..., HAS_BIAS=has_bias,
)
# No post-kernel bias add — it was done inside the kernel
```

The `else x_2d` dummy pointer is a standard Triton idiom: when `HAS_BIAS=False` the pointer is passed to satisfy the function signature but is never loaded, so no out-of-bounds access occurs.

---

## 4. Code Changes Summary

### `attention.py`

| Before | After |
|--------|-------|
| `attention_scores_kernel` writes raw scores to `scores` tensor | Removed from hot path |
| PyTorch `scores + causal_mask` writes masked scores back | Removed — done in registers |
| `softmax_inplace_kernel` reads and rewrites scores in-place | Removed from hot path |
| `attention_output_kernel` reads softmax weights + V | Replaced by `fused_attention_kernel` |
| Explicit padding of Q, K, V tensors | Removed — masks handle boundaries |
| 3–4 kernel launches + 2 intermediate tensors per attention call | 1 kernel launch, 0 intermediate tensors |

New function added:
```python
@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    scale, seq_q, seq_k, head_dim,
    is_causal: tl.constexpr,
    ...,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # All of: score computation, causal masking, softmax, V-weighted sum
    # in a single kernel — no intermediate global memory writes
```

### `layers.py`

| Before | After |
|--------|-------|
| `linear_kernel_tf32(a, b, c, M, N, K, ...)` | `linear_kernel_tf32(a, b, c, bias, M, N, K, ..., HAS_BIAS)` |
| `output = output + self.bias_param` after kernel | Removed — done inside kernel |
| 192 separate bias-add ops for audio encoder | 0 separate bias-add ops |

---

## 5. Benchmarking Methodology

Same methodology as Optimization 1:

- **Tool:** `benchmark_student.py --warmup 2 --runs 3`
- **Warmup passes (2):** JIT-compile all kernels and populate the autotune cache. This ensures the timed runs measure steady-state inference, not compilation overhead.
- **Timed passes (3):** Three complete end-to-end ASR inferences on `test_audio.wav` (3.5 s audio, 13-token transcription). Wall-clock time includes the full pipeline: audio encoder (32 layers) → projector → text decoder (28 layers × 13 autoregressive steps).
- **Hardware:** Single NVIDIA L4 GPU, no other heavy workloads running.
- **Correctness gate:** `Accuracy: 100.0%` required. Any regression in transcription quality would indicate a numerical error in the fusion.

---

## 6. Results

| Implementation | Run 1 | Run 2 | Run 3 | Mean ± Std | vs Example |
|---|---|---|---|---|---|
| Example (reference) | 1146.7 ms | 1148.3 ms | 1144.0 ms | **1146.4 ± 1.8 ms** | baseline |
| Template after Opt 1 | — | — | — | **1024.9 ± 4.6 ms** | −10.8% |
| Template after Opt 2 | 992.1 ms | 997.5 ms | 986.5 ms | **992.0 ± 5.5 ms** | **−13.5%** |

Additional gain from Optimization 2 alone (vs. post-Opt-1 baseline): **−3.2%** (1024.9 → 992.0 ms).

All runs produced:
```
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

---

## 7. Analysis: Where Does the Speedup Come From?

### Fused Attention Kernel

The text decoder uses the fused kernel for all 28 layers (seq_k ≈ 128 ≤ 256). Each decode step previously required:
- 1 `attention_scores_kernel` launch + write of scores (28 × 1 × 128 × 4B ≈ 14 KB per layer)
- 1 PyTorch causal-mask add (read + write the same 14 KB)
- 1 `softmax_inplace_kernel` launch (read + write the same 14 KB)
- 1 `attention_output_kernel` launch (reads scores + V)

After fusion:
- 1 `fused_attention_kernel` launch (reads Q, K, V; writes output only)
- The 14 KB scores tensor is never allocated or written

Over 13 generated tokens: 3 kernel launches × 28 layers × 13 tokens = **1,092 kernel launches eliminated**. Each launch saves the fixed overhead of a CUDA kernel dispatch (~5–10 µs) plus the global memory traffic for the scores tensor.

The audio encoder attention falls back to PyTorch einsum (seq_k=375 > 256), so no improvement there — but PyTorch/cuBLAS handles large attention efficiently anyway.

### Bias Folding

The 192 post-kernel bias additions in the audio encoder each read and write a tensor of up to 375 × 5120 floats ≈ 7.5 MB (for fc1 output). At L4's memory bandwidth of ~300 GB/s, reading 7.5 MB takes ~25 µs. Even if L2 cache absorbs most hits, eliminating 192 separate kernel dispatch events (each with ~5 µs overhead) saves ~1 ms just in scheduling alone.

The bias folding also slightly improves instruction-level parallelism: the bias load (`tl.load(bias_ptr + offs_n)`) and the accumulator stores are independent operations that the GPU's instruction scheduler can overlap.

---

## 8. Trade-offs and Limitations

### Fused Attention Kernel

**Scope:** Only activated when `seq_k_padded ≤ 256`. For the audio encoder (seq_k ≈ 375 → padded to 512) and for long text generation sequences (seq_k > 256), the PyTorch einsum fallback is used. This means the fused kernel primarily benefits the text decoder in this model, not the audio encoder.

**No external attention mask:** The fused kernel skips the `attention_mask` path. If an explicit padding mask were needed (e.g., batch inference with variable-length audio), the code falls back automatically to the original PyTorch path. For single-utterance inference (the standard case), `attention_mask=None` and the fused path is always taken for eligible sequence lengths.

**Register pressure:** Loading the full K matrix (BLOCK_K × BLOCK_D = 128 × 128 = 16,384 floats = 64 KB) into registers in one block is high. Triton may spill some values to L1/shared memory. This is inherent to the non-tiled approach and is why Optimization 3 (FlashAttention with tiling over seq_k) is the next step — it reduces peak register usage by processing K in smaller tiles.

**Autotune not applied:** The fused kernel uses fixed BLOCK_K and BLOCK_D (next power of two of actual seq_k and head_dim). These are not autotuned because the "block size" here is dictated by the full sequence length that must fit in one block for non-tiled attention. Optimization 3 will add tiling with autotuned tile sizes.

### Bias Folding

**Only affects `HAS_BIAS=True` linears:** Text decoder linear layers typically use no bias (`bias=False`), so the folding has zero effect on the decoder path. The gain is concentrated in the audio encoder's 192 biased linear calls.

**Autotune key unchanged:** The autotune key remains `["M", "N", "K"]`. The presence of bias (a single vector add) does not meaningfully affect which tile configuration is optimal for the matmul. Triton's compile-time specialization on `HAS_BIAS` ensures separate kernel variants are compiled for biased vs. non-biased layers, with no runtime overhead.

---

## 9. Cumulative Progress

| Optimization | Key Change | Speedup vs Example |
|---|---|---|
| None (template baseline) | Hardcoded 64×64×32 tiles, padding | +~5% slower than example |
| Opt 1: Autotune | `@triton.autotune` with 5 configs, padding removed from linear | −10.8% |
| Opt 2: Kernel Fusion | Fused attention kernel + bias folding in linear | **−13.5%** |

Combined, Optimizations 1 and 2 bring the template from 5% slower than the reference to **13.5% faster**, a swing of ~18 percentage points over the unoptimized baseline.

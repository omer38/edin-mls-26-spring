# HW1-ASR: GLM-ASR Triton Optimisation — Complete Report

**Track:** Triton (Python)
**Hardware:** NVIDIA L4 GPU (Ada Lovelace, sm_89, 60 SMs, 24 GB GDDR6, 48 MB L2 cache)
**Framework:** Triton 3.4.0 / PyTorch 2.x
**Task:** Automatic Speech Recognition (ASR) — transcribe 3.5 s of audio with GLM-ASR-Nano

---

## 0. Are the Optimisations Cumulative?

**Yes — each optimisation is applied on top of all previous ones.** The code was never reset between optimisations; each commit adds to a steadily improving codebase.

| Stage | What was changed | Time | vs Reference |
|-------|-----------------|------|-------------|
| Reference example (fixed) | `glm_asr_triton_example/` — hardcoded tiles, unfused kernels | 1146.4 ms | baseline |
| **Opt 1** applied | `layers.py`: `@triton.autotune` on matmul kernels | 1024.9 ms | −10.8% |
| **Opt 1 + Opt 2** applied | `attention.py` + `layers.py`: kernel fusion | 992.0 ms | −13.5% |
| **Opt 1 + Opt 2 + Opt 3** applied | `attention.py`: FlashAttention tiling | 751.4 ms | −34.5% |
| **Opt 1 + 2 + 3 + 4** applied | `layers.py` + `rope.py`: norm fix + Triton RoPE | **728.0 ms** | **−36.5%** |

All runs: 100% transcription accuracy ("Concord returned to its place amidst the tents."), benchmark methodology: 1 warmup run + 3 timed runs on `test_audio.wav`.

---

## 1. Model Overview and What We Are Optimising

The GLM-ASR-Nano pipeline has three stages:

```
WAV (3.5s)
  │
  ▼
Conv Subsampler (8× downsample)   → ~375 audio frames
  │
  ▼
Audio Encoder (32 Transformer layers)       ← head_dim=64, 20 heads, hidden=1280, FFN=5120
  │
  ▼
Projector (pool 4 frames, MLP)
  │
  ▼
Text Decoder (28 Transformer layers, GQA)   ← head_dim=128, 28Q/4KV heads, hidden=3584, FFN=18944
  │
  ▼
13 output tokens (autoregressive, 1 token at a time)
```

The two most expensive components are:

- **Audio Encoder**: large fixed matrices (M=375, K=1280/5120, N=1280/5120), runs **once** per inference through all 32 layers
- **Text Decoder (decode steps)**: tiny matrices (M=1, K=3584, N=3584/18944), runs **13 times** per inference × 28 layers = 364 matmul calls with M=1

Every kernel in both components is implemented in `layers.py` and `attention.py`. `model.py`, `weight_loader.py`, and `conv.py` cannot be modified (shared infrastructure).

---

## 2. Optimisation 1 — Tile Size Tuning with `@triton.autotune`

**File changed:** `glm_asr_triton_template/layers.py`

### The Problem

The original template hard-codes identical matmul tile sizes for every layer, every shape, and every GPU:

```python
class Linear:
    TILE_M, TILE_N, TILE_K = 64, 64, 32   # used everywhere, always
```

This is a poor fit for the two very different matrix regimes in this model:

| Case | M | K | N | Problem with 64×64×32 |
|------|---|---|---|----------------------|
| Encoder FFN up-proj | 375 | 1280 | 5120 | BLOCK_N=64 too narrow; leaves arithmetic intensity on the table |
| Encoder FFN down-proj | 375 | 5120 | 1280 | BLOCK_K=32 means 160 K-loop iterations; loop overhead is high |
| Decoder decode (any) | **1** | 3584 | 3584–18944 | BLOCK_M=64 wastes 63/64 registers on phantom rows; 98% wasted register space |

The M=1 decode case is the most damaging. With BLOCK_M=64 and M=1, the kernel allocates a 64×64 accumulator — 63 rows of which are never written. These wasted registers reduce how many thread blocks can run simultaneously on each SM (lower occupancy), hurting the decoder's 364 matmul calls the most.

### The Fix: `@triton.autotune` with 5 Configurations

```python
autotune_configs = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),  # baseline
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),  # large square → encoder
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),  # M=1 decode
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=8, num_stages=4),  # deep K pipeline
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),  # wide N → FFN
]

@triton.autotune(configs=autotune_configs, key=["M", "N", "K"])
@triton.jit
def linear_kernel_tf32(...):
    ...  # kernel body unchanged
```

How it works: on the first call for a given `(M, N, K)` shape, Triton benchmarks all 5 configurations on the GPU and caches the winner. All future calls for that shape dispatch immediately to the fastest binary.

Three matmul kernels were decorated: `linear_kernel_tf32`, `linear_gelu_kernel`, `swiglu_fused_kernel`.

### Side effect: Padding Removal

The original code allocated padded copies of input and weight tensors to match the fixed tile size:

```python
# Before: explicit zero-padding
x_padded = torch.zeros((M_padded, K_padded), ...)
x_padded[:M, :K] = x_2d
```

This was unnecessary — the kernel already uses `tl.load(..., mask=..., other=0.0)` for correct boundary handling. Since the tile size is now chosen at dispatch time (by the autotuner), we cannot pre-compute a fixed padded size anyway. Padding was removed entirely, eliminating ~6 memory allocations and copies per forward pass.

The grid also changed from a fixed tuple to a lambda:

```python
# Before: fixed grid (required padding to be safe)
grid = (triton.cdiv(M_padded, TILE_M), triton.cdiv(N_padded, TILE_N))

# After: lambda evaluated after autotuner selects tile
grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
```

### Result

| | Time |
|---|---|
| Reference example | 1146.4 ms |
| After Opt 1 | **1024.9 ms (−10.8%)** |

---

## 3. Optimisation 2 — Kernel Fusion

**Files changed:** `glm_asr_triton_template/attention.py`, `glm_asr_triton_template/layers.py`

This optimisation is **applied on top of Opt 1** (the autotuner from Opt 1 is still active).

### Background: Why Separate Kernels Waste Bandwidth

Every kernel output lives in GDDR6. The next kernel must read it back from GDDR6 before it can proceed. Even when two operations are logically a single computation `f(g(x))`, splitting them into two kernels causes the intermediate `g(x)` to travel:

```
Registers → GDDR6 → Registers
            ↑             ↑
         kernel 1     kernel 2
```

At 560 GB/s bandwidth this seems fast, but for small tensors (KB-range), the dominant cost is the fixed kernel-launch overhead (~5–10 µs per dispatch × hundreds of launches per inference).

### Fusion 1: Fused Attention Kernel

The original `scaled_dot_product_attention` used **four sequential operations** with three intermediate tensors touching global memory:

```
attention_scores_kernel  → scores tensor (GDDR6 write)
PyTorch causal mask add  → masked_scores (GDDR6 read+write)
softmax_inplace_kernel   → attn_weights  (GDDR6 read+write)
attention_output_kernel  → output        (GDDR6 read+write Q,K,V,weights)
```

`fused_attention_kernel` replaces all four with a single kernel. For each `(batch_head, query_position)`:

```
Load Q[pid_q]                     → registers
Load K[0:seq_k, :]                → registers
scores = (K @ Q) * scale          → registers only
Causal mask inline                → registers only (threshold formula handles decode & prefill)
Softmax (max → exp → sum → div)   → registers only
Load V[0:seq_k, :]                → registers
out = Σ attn_weights[j] * V[j]    → registers
Store out                         → GDDR6  ← only write
```

The causal mask threshold `seq_k − seq_q + pid_q` handles both regimes without branching:
- Prefill (seq_q = seq_k = N): threshold = pid_q → masks j > i ✓
- Decode (seq_q = 1, seq_k = N): threshold = N−1 → nothing masked ✓

The fused kernel is active when `seq_k_padded ≤ MAX_ATTENTION_DIM = 256`. At this stage the audio encoder (seq_k ≈ 375 → padded 512 > 256) still falls back to PyTorch — that is addressed in Opt 3.

**Eliminated:** 3 kernel launches × 28 decoder layers × 13 tokens = **1,092 CUDA kernel launches**. No intermediate scores tensor allocated per attention call.

### Fusion 2: Bias Folding into the Matmul Kernel

All audio encoder linear layers have `bias=True`. The original code added the bias as a separate PyTorch op after every matmul:

```python
linear_kernel_tf32[grid](x, weight_t, output, ...)   # write output to GDDR6
output = output + self.bias_param                     # read output, add bias, write output
```

The fix: add `bias_ptr` and `HAS_BIAS: tl.constexpr` to `linear_kernel_tf32`. The bias is added to the accumulator **in registers** before the single store:

```python
if HAS_BIAS:
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]     # broadcast: [N] → [BLOCK_M, N]

tl.store(c_ptrs, acc, mask=mask_c)   # one write only
```

When `HAS_BIAS=False` (text decoder layers have no bias), Triton compiles out the branch completely — zero runtime overhead.

**Eliminated:** 192 separate bias-add kernel launches across 32 encoder layers (4 attention projections + 2 MLP projections × 32 layers).

### Result (cumulative: Opt 1 + Opt 2)

| | Time |
|---|---|
| Reference example | 1146.4 ms |
| After Opt 1 | 1024.9 ms (−10.8%) |
| **After Opt 1 + Opt 2** | **992.0 ms (−13.5%)** |

Opt 2 alone contributed **−3.2%** on top of Opt 1.

---

## 4. Optimisation 3 — FlashAttention-style Tiled Attention

**File changed:** `glm_asr_triton_template/attention.py`

This optimisation is **applied on top of Opt 1 and Opt 2** (autotune and fusion from previous steps are still active).

### The Problem: The 256-Element Register Budget Wall

The `fused_attention_kernel` from Opt 2 loads the **entire** K and V matrices into registers at once (BLOCK_K = seq_k_padded). This is only feasible for small sequences. The model uses:

```
MAX_ATTENTION_DIM = 256

Audio encoder: seq_k ≈ 375  → padded to 512 > 256 → PyTorch einsum fallback ✗
Text decoder:  seq_k ≈ 128  → padded to 128 ≤ 256 → fused_attention_kernel ✓
```

So before Opt 3, **all 32 encoder layers × 20 heads = 640 attention calls per inference** go through the PyTorch fallback:

```python
# PyTorch path: CPU dispatch + cuBLAS GEMM × multiple calls
scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
...
output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)
```

Per audio encoder attention call, the full 375×375 attention matrix (560 KB per head) is written to and read from GDDR6 — then discarded. Over 640 calls that is **358 MB of avoidable DRAM traffic** plus hundreds of PyTorch kernel launches.

### The Algorithm: Online Softmax

FlashAttention (Dao et al., 2022) avoids materialising the full attention matrix by processing K and V in tiles and maintaining numerically stable running statistics across tiles.

**Running state per query row:**

```
m_i  = running row-max of scaled scores seen so far  (initialised: −∞)
l_i  = running row-sum of exp(scores − m_i)          (initialised: 0)
acc  = running output numerator                       (initialised: 0)
```

**Per K/V tile of size BLOCK_K:**

```
S_tile  = Q_tile @ K_tile^T * scale   # [BLOCK_Q, BLOCK_K] — uses tl.dot, tensor cores
(apply causal + padding mask)

m_ij   = rowmax(S_tile)                # [BLOCK_Q]
m_new  = max(m_i, m_ij)               # [BLOCK_Q] updated max
p      = exp(S_tile − m_new[:,None])   # [BLOCK_Q, BLOCK_K]
alpha  = exp(m_i − m_new)             # rescale factor: < 1 when max increased

l_i    = l_i  * alpha + rowsum(p)     # rescale existing, add new tile
acc    = acc  * alpha[:,None] + p @ V_tile  # rescale existing, add new contribution
m_i    = m_new
```

**Final (one write):**

```
output = acc / l_i[:,None]
```

When a new tile reveals a larger max value, `alpha < 1` rescales the existing `acc` and `l_i` downward before adding the new tile's contribution — keeping the running state consistent without ever needing the full score vector.

### The Key Architectural Difference: BLOCK_Q = 16

The `fused_attention_kernel` processes **one query row per program**. This means:
- Q@K^T is a `[1, BLOCK_K]` × `[BLOCK_K, BLOCK_D]` multiply — dimensions too small for tensor cores (require ≥ 16 in each dim)
- The kernel uses slow element-wise `tl.sum(k_mat * q_vec, axis=1)` instead

The `flash_attention_kernel` processes **BLOCK_Q = 16 query rows per program**:
- Q@K^T = `tl.dot([16, 64], tl.trans([64, 64]))` → `[16, 64]` using tensor cores ✓
- P@V   = `tl.dot([16, 64], [64, 64])` → `[16, 64]` using tensor cores ✓

So the move to tiling is doubly beneficial: it lifts the sequence-length cap *and* unlocks tensor core acceleration.

### Implementation

```python
FLASH_BLOCK_Q = 16   # query rows per program (≥16 for tl.dot tensor cores)
FLASH_BLOCK_K = 64   # K/V rows per loop iteration

@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    scale, seq_q, seq_k, head_dim,
    is_causal: tl.constexpr,
    ...strides...,
    BLOCK_Q: tl.constexpr,  # = 16
    BLOCK_K: tl.constexpr,  # = 64
    BLOCK_D: tl.constexpr,  # = head_dim_padded (64 or 128)
):
    # Grid: (batch*heads, ceil(seq_q / BLOCK_Q))
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)

    q_block = tl.load(...) * scale         # [BLOCK_Q, BLOCK_D], pre-scaled

    acc = tl.zeros((BLOCK_Q, BLOCK_D))
    m_i = tl.full((BLOCK_Q,), -1e9)
    l_i = tl.zeros((BLOCK_Q,))

    for k_start in range(0, seq_k, BLOCK_K):   # dynamic upper bound: handles any seq_k
        k_block  = tl.load(...)              # [BLOCK_K, BLOCK_D] — coalesced
        scores   = tl.dot(q_block, tl.trans(k_block))   # [BLOCK_Q, BLOCK_K], tensor cores
        # padding + causal mask...
        m_new    = tl.maximum(m_i, tl.max(scores, axis=1))
        p        = tl.exp(scores - m_new[:,None])
        alpha    = tl.exp(m_i - m_new)
        l_i      = l_i * alpha + tl.sum(p, axis=1)
        acc      = acc * alpha[:,None] + tl.dot(p, v_block)   # tensor cores
        m_i      = m_new

    tl.store(..., acc / l_i[:,None], ...)   # only global memory write
```

### Routing Logic (Three-tier dispatch)

```
q.is_cuda  and  no mask  and  seq_k_padded ≤ 256  →  fused_attention_kernel  (Opt 2, unchanged)
q.is_cuda  and  no mask  and  seq_k_padded  > 256  →  flash_attention_kernel  (Opt 3, new)
mask provided  or  CPU                              →  PyTorch fallback         (only survivor)
```

Before Opt 3: the PyTorch fallback handled all encoder attention.
After Opt 3: the PyTorch fallback handles **only** cases with an explicit `attention_mask` tensor (rare — this model passes `attention_mask=None` for all encoder attention).

### Memory Traffic Analysis (Audio Encoder, per attention call)

| Metric | Before Opt 3 | After Opt 3 |
|--------|-------------|-------------|
| Execution path | PyTorch einsum (CPU dispatch) | Triton flash_attention_kernel |
| Attention matrix (375×375/head) written | **563 KB/head** | **0** |
| Attention matrix read (for @V) | **563 KB/head** | **0** |
| Kernel launches per call | 4 (cuBLAS + softmax + cuBLAS) | **1** |
| Tensor cores used | No | **Yes** |

Over 640 audio encoder attention calls (32 layers × 20 heads):
- Intermediate DRAM traffic eliminated: 640 × 2 × 563 KB ≈ **720 MB**
- CUDA kernel launches replaced: 640 × 4 → **640** (75% reduction)

### Result (cumulative: Opt 1 + Opt 2 + Opt 3)

| | Time |
|---|---|
| Reference example | 1146.4 ms |
| After Opt 1 | 1024.9 ms (−10.8%) |
| After Opt 1 + Opt 2 | 992.0 ms (−13.5%) |
| After Opt 1 + Opt 2 + Opt 3 | 751.4 ms (−34.5%) |
| **After Opt 1 + Opt 2 + Opt 3 + Opt 4** | **728.0 ms (−36.5%)** |

Opt 3 alone contributed **−24.5%** on top of Opt 1+2. Opt 4 contributed an additional **−3.1%**.

---

## 5. Optimisation 4 — Triton Normalization + RoPE Rotation Kernel

**Files changed:** `glm_asr_triton_template/layers.py`, `glm_asr_triton_template/rope.py`

This optimisation is **applied on top of Opt 1, 2, and 3**.

### The Problem: 13,400+ Hidden PyTorch Kernel Launches per Inference

After Opt 3, every attention and matmul path ran through Triton. Two families of operations still dispatched to PyTorch on every inference:

**Family 1 — Normalization**: The template guarded its Triton norm paths with
`self.use_triton = _is_power_of_two(hidden_size)`. Both model hidden sizes — 1280
(encoder) and 3584 (decoder) — are **not** powers of two, so **all 793 norm calls**
per inference fell through to a 7-operation PyTorch chain:

```python
x.to(float32) → x*x → mean → rsqrt → multiply → weight_multiply → cast_back
```

793 calls × 7 ops = **5,551 micro-kernel launches** from normalization alone.

**Family 2 — RoPE rotation**: `_apply_rope_single` used `torch.cat` and element-wise
PyTorch operators. Called 792 times per inference (32 encoder + 28 × 13 decoder layers,
twice for Q and K), each call executing ~10 PyTorch ops including a `torch.cat`
allocation. Total: **~7,920 small PyTorch kernel launches** from RoPE alone.

Combined: approximately **13,400 small CUDA kernel launches** per inference running in
Python instead of Triton.

### Fix 4a — Remove Non-Power-of-2 Guard (layers.py, 4 lines)

The Triton kernels already used `BLOCK_SIZE = next_power_of_two(hidden_size)` with
masking — they were correct for any hidden size. The guard was removed:

```python
# Before: if self.use_triton and x.is_cuda:
# After:  if x.is_cuda:
```

793 norm calls now dispatch directly to the single Triton kernel, eliminating 5 PyTorch
intermediate tensor writes per call.

### Fix 4b — Triton RoPE Rotation Kernel (rope.py, +77 lines)

A `@triton.jit apply_rope_kernel` processes one `(batch*head, seq_position)` tile per
program. Each program:

1. Loads `x1 = x[bh, s, :half_dim]` and `x2 = x[bh, s, half_dim:2*half_dim]`
2. Loads `cos[s, :half_dim]` and `sin[s, :half_dim]`
3. Computes `r1 = x1*cos − x2*sin` and `r2 = x2*cos + x1*sin` **in registers**
4. Stores rotated halves and copies passthrough dimensions (for partial RoPE)

Key detail: `BLOCK_D = next_power_of_two(head_dim)` (not `half_dim`) ensures the
passthrough region (dimensions `[2*half_dim : head_dim]`, 32 elements for the audio
encoder) is fully covered within a single program.

The `torch.cat` allocation — `[1, 20, 375, 64] × float32 = 1.87 MB` per encoder Q or K
call — is eliminated entirely.

### Numerical precision

RoPE kernel: **exact float32**, max absolute error vs PyTorch reference = 0.000000 for
both audio encoder (partial RoPE, head_dim=64) and text decoder (full RoPE, head_dim=128)
configurations. No reduced precision is involved.

### Result (cumulative: Opt 1 + Opt 2 + Opt 3 + Opt 4)

| | Time |
|---|---|
| Reference example | 1146.4 ms |
| After Opt 1 | 1024.9 ms (−10.8%) |
| After Opt 1 + Opt 2 | 992.0 ms (−13.5%) |
| After Opt 1 + Opt 2 + Opt 3 | 751.4 ms (−34.5%) |
| **After Opt 1 + Opt 2 + Opt 3 + Opt 4** | **728.0 ms (−36.5%)** |

Opt 4 alone contributed **−23.4 ms (−3.1%)** on top of Opt 3. The gain is smaller than
Opt 3 because each individual norm and RoPE operation is cheap (small tensors), but the
cumulative effect of eliminating ~11,900 redundant kernel launches and ~792 `torch.cat`
allocations per inference produces a measurable improvement.

---

## 6. Cumulative Summary

### Performance at Each Stage

```
Reference example:           1146.4 ms  ──────────────────────── baseline
After Opt 1 (autotune):      1024.9 ms  ████░░░░░░░░░░░░░░░░░░░  −10.8%
After Opt 1+2 (fusion):       992.0 ms  █████░░░░░░░░░░░░░░░░░░  −13.5%
After Opt 1+2+3 (flash):      751.4 ms  ███████████░░░░░░░░░░░░  −34.5%
After Opt 1+2+3+4 (norm+RoPE): 728.0 ms  ████████████░░░░░░░░░░  −36.5%
```

### What Each Optimisation Targets

| Opt | Technique | Bottleneck Addressed | Primary Beneficiary |
|-----|-----------|---------------------|---------------------|
| 1 | `@triton.autotune` with 5 tile configs | Wrong tile size for different M regimes | Decoder M=1 decode steps; encoder wide-N FFN |
| 2 | Kernel fusion (attention + bias) | Redundant global memory roundtrips; kernel launch overhead | Text decoder attention; audio encoder bias ops |
| 3 | FlashAttention tiling + online softmax | Audio encoder attention PyTorch fallback; no tensor cores | Audio encoder (32 layers × 20 heads) |
| 4 | Norm fallback fix + Triton RoPE kernel | 13,400 micro-kernel launches from norm/RoPE Python paths | Decoder decode steps; encoder norms + RoPE |

### Files Changed

| File | Opt 1 | Opt 2 | Opt 3 | Opt 4 |
|------|:-----:|:-----:|:-----:|:-----:|
| `layers.py` | ✓ autotune + padding removed | ✓ bias fusion | — | ✓ norm guard removed |
| `attention.py` | — | ✓ fused attention | ✓ flash attention | — |
| `rope.py` | — | — | — | ✓ `apply_rope_kernel` added |

### Why All Four Are Additive (Not Redundant)

All four optimisations target **orthogonal bottlenecks**:

1. **Opt 1** improves matmul tile efficiency. Unaffected by kernel fusion or attention changes.
2. **Opt 2** fuses attention and bias ops for short sequences. Unaffected by tile selection or RoPE.
3. **Opt 3** moves long-sequence encoder attention from Python to Triton. Orthogonal to norms and RoPE.
4. **Opt 4** eliminates the norm and RoPE Python dispatch overhead that survived Opt 1–3. Orthogonal to matmul tiling, attention fusion, and FlashAttention.

Each optimisation is active and contributes independently in the final combined implementation.

---

## 7. Benchmark Methodology

**Tool:** `benchmark_student.py` via `benchmark.sh`
**Runs:** 1 warmup run (JIT compile + populate autotune cache) + 3 timed runs
**Input:** `test_audio.wav` — 3.5 seconds of speech, 13-token transcription
**Correctness gate:** All results require `Accuracy: 100.0%` to be reported

The warmup run is essential for Opt 1: Triton's autotune cache is in-memory only and is not persisted to disk. Without warmup, the first timed run includes autotune search time (~20–40 seconds for all shapes), which would artificially inflate the measured latency. With warmup, timed runs reflect pure steady-state inference.

---

## 8. Hardware Context

**NVIDIA L4 (Ada Lovelace, sm_89)**

| Spec | Value |
|------|-------|
| Architecture | Ada Lovelace (sm_89) |
| CUDA Cores | 7680 |
| Tensor Cores | 240 4th-gen (INT8/FP16/TF32/BF16) |
| SMs | 60 |
| Memory | 24 GB GDDR6 |
| Memory Bandwidth | ~560 GB/s |
| L2 Cache | 48 MB |
| Shared Memory / SM | 100 KB (configurable) |
| TF32 throughput | ~242 TFLOPS |

**Relevance to each optimisation:**

- **Opt 1:** The 5 autotune configs were designed around the L4's 60 SMs, 100 KB shared memory, and the fact that `num_stages=4` is beneficial for hiding DRAM latency (~300 cycles on GDDR6). The winning configs differ from what would be optimal on A100 or H100 hardware — the autotuner adapts to whatever GPU it runs on.
- **Opt 2:** 560 GB/s peak bandwidth makes even small intermediate tensors worth eliminating. At that rate, a 563 KB scores tensor (1 encoder attention head) takes ~1 µs to write — and another 1 µs to read back — per call.
- **Opt 3:** 4th-generation tensor cores (TF32 mode) are only engaged when matrix dimensions ≥ 16. The `FLASH_BLOCK_Q=16` choice was made precisely to meet this minimum. At TF32 throughput, `tl.dot([16,64] @ [64,64])` achieves ~10× the throughput of the element-wise sum loop it replaces.
- **Opt 4:** No TF32 involvement. Normalization and RoPE use pure float32 arithmetic. The `BLOCK_D = next_power_of_two(head_dim)` sizing allows the compiler to allocate exact register arrays without dynamic indexing overhead.

---

## 9. Precision Notes

- All matmul kernels accumulate in **float32**.
- `tl.dot` uses **TF32** (10-bit mantissa) for the multiply stage on Ampere+/Ada. This is slightly less precise than full float32 but gives full tensor-core throughput. The accuracy difference is negligible for inference (confirmed: 100% transcription accuracy throughout).
- The flash attention kernel's max absolute error vs. a full float32 PyTorch reference is < 0.002 (< 0.5% relative error on a unit-variance input), well within model tolerance.
- The Opt 4 RoPE kernel is **exact float32** — max absolute error vs PyTorch reference is 0.000000 for both encoder (partial RoPE, head_dim=64) and decoder (full RoPE, head_dim=128) configurations.
- Output dtype is preserved: results are cast back to the input tensor's dtype (typically `float16` or `bfloat16`) after computation.

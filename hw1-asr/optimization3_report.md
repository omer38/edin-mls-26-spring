# Optimization 3: FlashAttention-style Tiled Attention

## 1. Background

After Optimization 2 (kernel fusion), the audio encoder's 32 layers × 20 heads = **640 attention operations** still fell back to PyTorch einsum. The audio encoder has sequence length seq_k ≈ 375 audio frames (after 4× subsampling), which pads to 512 — above the `MAX_ATTENTION_DIM = 256` limit of the fused kernel from Opt 2.

Optimization 3 adds a **FlashAttention-style tiled kernel** that handles arbitrarily long sequences, eliminating the last significant PyTorch fallback path.

---

## 2. Problem: The Register-Budget Wall

The `fused_attention_kernel` (Opt 2) processes one query row per program and holds the *entire* K and V tensors in registers simultaneously:

```
BLOCK_K = seq_k_padded   # up to 256 elements
BLOCK_D = head_dim_padded # up to 256 elements
```

For seq_k = 375 → padded to 512:

```
K tile in registers: 512 × 64 × 4 bytes = 128 KB  ← exceeds per-SM register budget
```

This is why `MAX_ATTENTION_DIM = 256` was necessary. Any audio encoder call bypasses Triton entirely:

```python
# Before Opt 3 — PyTorch dispatch for every encoder attention call
scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
...
output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)
```

Overhead per PyTorch fallback call:
- CPU-side Python dispatch
- Multiple CUDA kernel launches (cuBLAS GEMM + softmax ops)
- Full attention matrix (375 × 375 × 20 heads = 2.8M floats ≈ 11 MB) written to DRAM

---

## 3. Algorithm: Online Softmax

FlashAttention (Dao et al., 2022) avoids materializing the full attention matrix by processing K/V in tiles and maintaining running statistics for a numerically stable online softmax.

### Running state per query row

| Variable | Shape | Meaning |
|----------|-------|---------|
| `m_i`    | [BLOCK_Q] | running row-max of scaled scores seen so far |
| `l_i`    | [BLOCK_Q] | running row-sum of exp(scores − m_i) |
| `acc`    | [BLOCK_Q, BLOCK_D] | running output numerator |

### Per-tile update (for each K/V block of size BLOCK_K)

```
S_tile  = Q_tile @ K_tile^T * scale        # [BLOCK_Q, BLOCK_K]
apply causal mask inline

m_new   = max(m_i, rowmax(S_tile))         # updated max
P       = exp(S_tile − m_new[:,None])       # [BLOCK_Q, BLOCK_K] softmax numerators
alpha   = exp(m_i − m_new)                  # rescale factor for existing state

l_i     = l_i  * alpha + rowsum(P)
acc     = acc  * alpha[:,None] + P @ V_tile
m_i     = m_new
```

### Final normalisation (one write to DRAM)

```
output = acc / l_i[:,None]
```

**Key property**: when a new tile reveals a larger max, `alpha = exp(m_old − m_new) < 1` rescales the existing `acc` and `l_i` so the running values remain consistent. The final division by `l_i` is the only normalisation step — no separate softmax pass.

---

## 4. Implementation

### 4.1 Kernel design

File: `attention.py`

```python
FLASH_BLOCK_Q = 16   # query rows per program — enables tl.dot (tensor cores)
FLASH_BLOCK_K = 64   # K/V rows per inner-loop iteration

@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    scale, seq_q, seq_k, head_dim,
    is_causal: tl.constexpr,
    ...strides...,
    BLOCK_Q: tl.constexpr,   # = 16
    BLOCK_K: tl.constexpr,   # = 64
    BLOCK_D: tl.constexpr,   # = head_dim_padded (64 or 128)
):
    # Grid: (batch*heads, cdiv(seq_q, BLOCK_Q))
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)
    q_start = pid_q * BLOCK_Q
    q_offs  = q_start + tl.arange(0, BLOCK_Q)
    d_offs  = tl.arange(0, BLOCK_D)

    # Load Q tile [BLOCK_Q, BLOCK_D], pre-scaled
    q_block = tl.load(...) * scale

    # Running state
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_Q,), -1e9, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,),       dtype=tl.float32)

    for k_start in range(0, seq_k, BLOCK_K):
        k_offs  = k_start + tl.arange(0, BLOCK_K)
        k_block = tl.load(...)                          # [BLOCK_K, BLOCK_D]
        scores  = tl.dot(q_block, tl.trans(k_block))   # [BLOCK_Q, BLOCK_K] — tensor cores
        # masking + causal mask ...
        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p     = tl.exp(scores - m_new[:,None])
        alpha = tl.exp(m_i - m_new)
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        acc   = acc * alpha[:,None] + tl.dot(p, v_block)  # tensor cores
        m_i   = m_new

    tl.store(..., acc / l_i[:,None], ...)
```

### 4.2 Why BLOCK_Q = 16 matters

The `fused_attention_kernel` processes **one query row** per program instance. This means:

- Q@K^T = `[1, head_dim] @ [head_dim, seq_k]` — no tensor cores (dim < 16)
- P@V   = `[1, seq_k] @ [seq_k, head_dim]`   — no tensor cores

With `BLOCK_Q = 16`:

- Q@K^T = `tl.dot([16, 64], [64, 64])` — tensor cores active ✓
- P@V   = `tl.dot([16, 64], [64, 64])` — tensor cores active ✓

This is a fundamentally different compute regime — the larger Q tile amortizes kernel launch overhead and lets the SM's tensor cores stay busy.

### 4.3 Routing logic

```python
# Path 1: fused kernel — small seq, no mask (Opt 2, unchanged)
use_fused = q.is_cuda and attention_mask is None and seq_k_padded <= 256

# Path 2: flash kernel — large seq, no mask (Opt 3, new)
use_flash = q.is_cuda and attention_mask is None and head_dim_padded <= 256

# Path 3: PyTorch fallback — external mask or CPU (only remaining fallback)
```

Before Opt 3, "Path 3" handled all audio encoder attention. After Opt 3, only the rare `attention_mask is not None` case remains in Python.

### 4.4 Causal masking

The same inline threshold formula from `fused_attention_kernel` generalises naturally to the tile structure:

```python
# For Q tile covering rows q_offs = [q_start, ..., q_start + BLOCK_Q - 1]
# and K tile covering cols k_offs = [k_start, ..., k_start + BLOCK_K - 1]:
causal_thresh = (seq_k - seq_q + q_offs)[:,None]   # [BLOCK_Q, 1]
scores = tl.where(k_offs[None,:] > causal_thresh, -1e9, scores)
```

- Prefill (seq_q == seq_k): query i masks k > i ✓
- Decode (seq_q == 1): no masking needed ✓
- Works correctly across tile boundaries without any special-casing ✓

---

## 5. Memory Traffic Analysis

### Audio encoder attention: 1 layer, 20 heads, seq_q = seq_k = 375, head_dim = 64

| Operation | Before Opt 3 | After Opt 3 |
|-----------|-------------|-------------|
| Attention execution | PyTorch einsum fallback | Triton flash_attention_kernel |
| K/V reads per head | Full (375×64 × 2 = 192 KB) | Tiled: 6 × (64×64 × 2 = 32 KB) = 192 KB |
| Attention matrix write | 375×375 × 4 B = 563 KB per head | **None** (stays in registers) |
| Attention matrix read | 563 KB per head (for @V) | **None** |
| Net intermediate traffic | **1.13 MB / head** | **0** |
| Python dispatch overhead | ~4 CUDA ops × kernel launch | **1** Triton kernel launch |
| Tensor cores used | No (einsum) | Yes (tl.dot) |

Per full inference (32 encoder layers × 20 heads):
- Eliminated intermediate writes: 32 × 20 × 1.13 MB = **723 MB** of DRAM traffic
- Eliminated CPU dispatch calls: 32 × 20 × 4 ops = **2,560 PyTorch kernel launches** replaced by **640 Triton kernel launches**

---

## 6. Benchmark Results

Hardware: NVIDIA L4, 24 GB GDDR6, 58 SM, 560 GB/s memory bandwidth

| Configuration | Time (ms) | Speedup vs Example |
|--------------|-----------|-------------------|
| Example (reference) | 1146.4 ± 1.8 | — |
| Template post-Opt 1 (autotune) | 1039.0 ± 8.0 | −9.4% |
| Template post-Opt 2 (fusion) | 996.6 ± 4.5 | −13.1% |
| **Template post-Opt 3 (flash attn)** | **751.4 ± 4.2** | **−34.5%** |

**Opt 3 alone: 996.6 → 751.4 ms = −24.5% additional speedup**

Numerical accuracy: 100% transcription accuracy ("Concord returned to its place amidst the tents."), max TF32 error < 0.003 vs PyTorch reference.

---

## 7. Why This Works So Well

The audio encoder dominates inference time because it processes **375 audio frames through 32 transformer layers**. Each of these 32 layers calls attention once. Before Opt 3:

1. All 32 encoder attention calls → PyTorch einsum (CPU dispatch, no Triton)
2. 640 separate CUDA kernel launches (score matmul, softmax, output matmul × 2)
3. 723 MB unnecessary intermediate DRAM writes

After Opt 3:
1. All 32 encoder attention calls → 1 Triton kernel each
2. 32 × 20 = 640 Triton kernel launches (all the attention work in one shot)
3. 0 intermediate DRAM writes — K, V, P all stay in registers

The 24.5% speedup from Opt 3 is larger than Opt 1 (9.4%) and Opt 2's attention portion combined, because it moves the single biggest remaining bottleneck (encoder attention) off the Python/PyTorch path entirely.

---

## 8. Trade-offs and Limitations

| Aspect | Note |
|--------|------|
| **TF32 precision** | `tl.dot` uses TF32 on Ampere+ (10-bit mantissa). Max error ~0.002 vs FP32 reference — below model tolerance. |
| **BLOCK_Q = 16 minimum** | `tl.dot` requires all dimensions ≥ 16. For seq_q < 16 (decode steps), the flash kernel still launches but some query slots are masked — wasteful but correct. In practice, decode seq_q = 1 falls to the fused_attention_kernel (seq_k ≤ 256 during decode). |
| **No autotuning** | BLOCK_Q=16, BLOCK_K=64 are fixed. An autotuner could potentially find BLOCK_Q=32 or BLOCK_K=128 to be faster on certain shapes. |
| **Causal loop inefficiency** | For causal prefill, ~half the K/V tiles are masked out but still loaded. A compile-time `k_limit` branch could halve the causal cost — left for future work. |

---

## 9. Cumulative Progress

| Optimization | Technique | Time (ms) | Delta |
|-------------|-----------|-----------|-------|
| Example baseline | Reference | 1146.4 | — |
| Opt 1: Autotune | 5-config autotuner for linear_kernel_tf32 | 1039.0 | −9.4% |
| Opt 2: Kernel fusion | Fused attention (3→1 kernel) + bias folding | 996.6 | −13.1% total |
| **Opt 3: FlashAttention** | **Tiled online-softmax for audio encoder** | **751.4** | **−34.5% total** |

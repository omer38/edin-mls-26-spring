"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)

    *** TODO: Implement this kernel ***
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention score computation
    # ============================================================================
    #
    # Step 1: Load query vector for this position
    # Step 2: Load all keys for this batch_head
    # Step 3: Compute dot-product scores and scale
    # Step 4: Store scores

    # YOUR CODE HERE

    col_idx = tl.arange(0, BLOCK_D)
    key_idx = tl.arange(0, BLOCK_K)

    q_base = q_ptr + pid_bh * stride_q0 + pid_q * stride_q1
    q_vec = tl.load(q_base + col_idx * stride_q2, mask=col_idx < head_dim, other=0.0)

    k_base = k_ptr + pid_bh * stride_k0
    k_mat = tl.load(
        k_base + key_idx[:, None] * stride_k1 + col_idx[None, :] * stride_k2,
        mask=(key_idx[:, None] < seq_k) & (col_idx[None, :] < head_dim),
        other=0.0,
    )

    dot = tl.sum(k_mat * q_vec[None, :], axis=1)
    s_base = scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1
    tl.store(s_base + key_idx * stride_s2, dot * scale, mask=key_idx < seq_k)


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads * seq_q,)
    """
    row = tl.program_id(0)

    # ============================================================================
    # TODO: Implement softmax
    # ============================================================================
    #
    # Step 1: Load scores row with masking
    # Step 2: Subtract max for stability
    # Step 3: Compute exp and normalize
    # Step 4: Store back

    # YOUR CODE HERE

    col_idx = tl.arange(0, BLOCK_SIZE)
    valid = col_idx < seq_k
    row_ptr = scores_ptr + row * stride_s
    x = tl.load(row_ptr + col_idx, mask=valid, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    tl.store(row_ptr + col_idx, numerator / tl.sum(numerator, axis=0), mask=valid)


@triton.jit
def attention_output_kernel(
    attn_ptr,
    v_ptr,
    output_ptr,
    seq_k,
    head_dim,
    stride_w0,
    stride_w1,
    stride_w2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute attention output: attn_weights @ V
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention output computation
    # ============================================================================
    #
    # Step 1: Load attention weights for this query
    # Step 2: Load all values for this batch_head
    # Step 3: Compute weighted sum
    # Step 4: Store output

    # YOUR CODE HERE

    key_idx = tl.arange(0, BLOCK_K)
    col_idx = tl.arange(0, BLOCK_D)

    w_base = attn_ptr + pid_bh * stride_w0 + pid_q * stride_w1
    weights = tl.load(w_base + key_idx * stride_w2, mask=key_idx < seq_k, other=0.0)

    v_base = v_ptr + pid_bh * stride_v0
    val_mat = tl.load(
        v_base + key_idx[:, None] * stride_v1 + col_idx[None, :] * stride_v2,
        mask=(key_idx[:, None] < seq_k) & (col_idx[None, :] < head_dim),
        other=0.0,
    )

    result = tl.sum(val_mat * weights[:, None], axis=0)
    o_base = output_ptr + pid_bh * stride_o0 + pid_q * stride_o1
    tl.store(o_base + col_idx * stride_o2, result, mask=col_idx < head_dim)


@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )


@triton.jit
def fused_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    scale,
    seq_q,
    seq_k,
    head_dim,
    is_causal: tl.constexpr,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused attention kernel: Q·K^T * scale + causal_mask → softmax → @V.

    Replaces three separate kernels (attention_scores_kernel, softmax_inplace_kernel,
    attention_output_kernel) with a single pass:
      1. Compute scaled dot-product scores for this query (in registers).
      2. Apply causal mask inline — no intermediate scores tensor written.
      3. Compute numerically stable softmax in registers.
      4. Compute weighted sum with V and store the output.

    Grid: (batch * num_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    col_idx = tl.arange(0, BLOCK_D)
    key_idx = tl.arange(0, BLOCK_K)

    # -------------------------------------------------------------------------
    # Step 1: Load query vector for this position
    # -------------------------------------------------------------------------
    q_vec = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + col_idx * stride_q2,
        mask=col_idx < head_dim,
        other=0.0,
    )

    # -------------------------------------------------------------------------
    # Step 2: Load all keys and compute scaled dot-product scores
    # -------------------------------------------------------------------------
    k_mat = tl.load(
        k_ptr + pid_bh * stride_k0 + key_idx[:, None] * stride_k1 + col_idx[None, :] * stride_k2,
        mask=(key_idx[:, None] < seq_k) & (col_idx[None, :] < head_dim),
        other=0.0,
    )
    scores = tl.sum(k_mat * q_vec[None, :], axis=1) * scale

    # Mask out padding positions (key indices beyond actual seq_k)
    scores = tl.where(key_idx < seq_k, scores, -1e9)

    # -------------------------------------------------------------------------
    # Step 3: Apply causal mask inline (no global memory write needed)
    #
    # Threshold formula: mask position j if j > (seq_k - seq_q + pid_q)
    #   - Prefill (seq_q == seq_k == N):  threshold = pid_q  → mask j > i  ✓
    #   - Decode  (seq_q == 1, seq_k == N): threshold = N-1  → nothing masked ✓
    # -------------------------------------------------------------------------
    if is_causal:
        causal_threshold = seq_k - seq_q + pid_q
        scores = tl.where(key_idx > causal_threshold, -1e9, scores)

    # -------------------------------------------------------------------------
    # Step 4: Numerically stable softmax entirely in registers
    # -------------------------------------------------------------------------
    scores_max = tl.max(scores, axis=0)
    scores_exp = tl.exp(scores - scores_max)
    attn_weights = scores_exp / tl.sum(scores_exp, axis=0)

    # -------------------------------------------------------------------------
    # Step 5: Load values and compute weighted sum
    # -------------------------------------------------------------------------
    v_mat = tl.load(
        v_ptr + pid_bh * stride_v0 + key_idx[:, None] * stride_v1 + col_idx[None, :] * stride_v2,
        mask=(key_idx[:, None] < seq_k) & (col_idx[None, :] < head_dim),
        other=0.0,
    )
    out = tl.sum(v_mat * attn_weights[:, None], axis=0)

    # -------------------------------------------------------------------------
    # Step 6: Store output — first and only write to global memory
    # -------------------------------------------------------------------------
    tl.store(
        output_ptr + pid_bh * stride_o0 + pid_q * stride_o1 + col_idx * stride_o2,
        out,
        mask=col_idx < head_dim,
    )


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention.

    Triton fast path (when seq_k and head_dim fit in MAX_ATTENTION_DIM and no
    external attention_mask is provided): uses fused_attention_kernel which
    computes Q·K^T, applies the causal mask, runs softmax, and multiplies by V
    all in a single kernel — no intermediate scores tensor is ever written to
    global memory.

    Falls back to PyTorch einsum when the sequence is too long for the current
    block-register budget or when an external attention_mask is supplied.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    seq_k_padded = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    # Use the fused Triton kernel when:
    #   • tensors are on CUDA
    #   • no external attention_mask (handled inline for causal; external masks
    #     require a separate add which is not worth adding extra kernel args for)
    #   • block sizes fit in the per-SM register budget
    use_triton = (
        q.is_cuda
        and attention_mask is None
        and seq_k_padded <= MAX_ATTENTION_DIM
        and head_dim_padded <= MAX_ATTENTION_DIM
    )

    if use_triton:
        # Reshape to (batch*heads, seq, dim) — no explicit zero-padding needed;
        # the fused kernel uses boundary masks (other=0.0) instead.
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32).contiguous()
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32).contiguous()
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32).contiguous()

        # Output buffer — head_dim_padded == head_dim for all standard dims (64, 128).
        output = torch.empty(
            (batch * num_heads, seq_q, head_dim_padded),
            dtype=torch.float32,
            device=q.device,
        )

        grid = (batch * num_heads, seq_q)
        fused_attention_kernel[grid](
            q_flat, k_flat, v_flat, output,
            float(scale), seq_q, seq_k, head_dim,
            is_causal,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    # ------------------------------------------------------------------
    # PyTorch fallback: used when seq_k > MAX_ATTENTION_DIM or when an
    # external attention_mask is provided.
    # ------------------------------------------------------------------
    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)

    return output.to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2 :] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    print("\nOutput statistics:")
    print(f"  Mean: {float(output.mean()):.4f}")
    print(f"  Std:  {float(output.std()):.4f}")
    print(f"  Min:  {float(output.min()):.4f}")
    print(f"  Max:  {float(output.max()):.4f}")

    print("\nTriton Attention working!")

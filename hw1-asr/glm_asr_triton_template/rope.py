"""
Triton Rotary Position Embeddings (RoPE)
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement RoPE using Triton kernels
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for RoPE
# ============================================================================

@triton.jit
def compute_freqs_kernel(
    positions_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    half_dim,
    stride_pos,
    stride_inv,
    stride_cos0,
    stride_cos1,
    stride_sin0,
    stride_sin1,
    BLOCK: tl.constexpr,
):
    """
    Compute cos and sin for rotary embeddings.

    *** TODO: Implement this kernel ***

    Grid: (seq_len,)
    """
    pid = tl.program_id(0)

    # ============================================================================
    # TODO: Implement frequency computation
    # ============================================================================
    #
    # Step 1: Load position as scalar
    # Step 2: Load inverse frequencies
    # Step 3: Compute freqs = position * inv_freq
    # Step 4: Compute cos and sin
    # Step 5: Store concatenated cos/sin

    # YOUR CODE HERE
    offs = tl.arange(0, BLOCK)
    mask = offs < half_dim

    pos = tl.load(positions_ptr + pid * stride_pos)
    inv_freqs = tl.load(inv_freq_ptr + offs * stride_inv, mask=mask, other=0.0)
    freqs = pos * inv_freqs

    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    cos_base = cos_ptr + pid * stride_cos0
    sin_base = sin_ptr + pid * stride_sin0
    tl.store(cos_base + offs * stride_cos1, cos, mask=mask)
    tl.store(cos_base + (offs + half_dim) * stride_cos1, cos, mask=mask)
    tl.store(sin_base + offs * stride_sin1, sin, mask=mask)
    tl.store(sin_base + (offs + half_dim) * stride_sin1, sin, mask=mask)



# ============================================================================
# Triton RoPE Rotation Kernel
# ============================================================================

@triton.jit
def apply_rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    seq_len, half_dim, head_dim,
    stride_xbh, stride_xs, stride_xd,
    stride_obh, stride_os, stride_od,
    BLOCK_D: tl.constexpr,
):
    """
    Apply RoPE rotation to one (batch*head, seq_pos) tile.
    Grid: (batch * num_heads * seq_len,)

    x shape: [batch*num_heads, seq_len, head_dim]  (x_flat)
    cos/sin shape: [seq_len, half_dim]  (already sliced to half_dim)

    Rotation:
      out[..., :half_dim]           = x[..., :half_dim]*cos  - x[..., half_dim:2*half_dim]*sin
      out[..., half_dim:2*half_dim] = x[..., half_dim:2*half_dim]*cos + x[..., :half_dim]*sin
      out[..., 2*half_dim:]         = x[..., 2*half_dim:]  (passthrough)
    """
    pid = tl.program_id(0)
    bh = pid // seq_len
    s  = pid - bh * seq_len

    d = tl.arange(0, BLOCK_D)

    # Load x1 = x[bh, s, :half_dim]
    x1 = tl.load(x_ptr + bh * stride_xbh + s * stride_xs + d * stride_xd,
                  mask=d < half_dim, other=0.0)

    # Load x2 = x[bh, s, half_dim : 2*half_dim]
    x2 = tl.load(x_ptr + bh * stride_xbh + s * stride_xs + (half_dim + d) * stride_xd,
                  mask=d < half_dim, other=0.0)

    # Load cos/sin for this position: cos[s, :half_dim]
    cos = tl.load(cos_ptr + s * half_dim + d, mask=d < half_dim, other=1.0)
    sin = tl.load(sin_ptr + s * half_dim + d, mask=d < half_dim, other=0.0)

    # Rotated first half
    r1 = x1 * cos - x2 * sin
    # Rotated second half
    r2 = x2 * cos + x1 * sin

    # Store first half
    tl.store(out_ptr + bh * stride_obh + s * stride_os + d * stride_od,
             r1, mask=d < half_dim)
    # Store second half
    tl.store(out_ptr + bh * stride_obh + s * stride_os + (half_dim + d) * stride_od,
             r2, mask=d < half_dim)

    # Passthrough: copy x[2*half_dim:head_dim] unchanged (masked if no passthrough)
    pass_d = 2 * half_dim + d
    pass_mask = (d < BLOCK_D) & (pass_d < head_dim)
    p = tl.load(x_ptr + bh * stride_xbh + s * stride_xs + pass_d * stride_xd,
                mask=pass_mask, other=0.0)
    tl.store(out_ptr + bh * stride_obh + s * stride_os + pass_d * stride_od,
             p, mask=pass_mask)


# ============================================================================
# RoPE Classes
# ============================================================================

class RotaryEmbedding:
    """Rotary Position Embedding using Triton."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        self.rotary_dim = int(dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.inv_freq = inv_freq

        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos and sin using Triton kernel."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)
        sin_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)

        if device.type == "cuda":
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)

            block = triton.next_power_of_2(half_dim)
            compute_freqs_kernel[(seq_len,)](
                positions,
                self.inv_freq,
                cos_cache,
                sin_cache,
                seq_len,
                half_dim,
                positions.stride(0),
                self.inv_freq.stride(0),
                cos_cache.stride(0),
                cos_cache.stride(1),
                sin_cache.stride(0),
                sin_cache.stride(1),
                BLOCK=block,
            )
        else:
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            freqs = positions[:, None] * self.inv_freq[None, :]
            cos_half = torch.cos(freqs)
            sin_half = torch.sin(freqs)
            cos_cache[:, :half_dim] = cos_half
            cos_cache[:, half_dim : half_dim * 2] = cos_half
            sin_cache[:, :half_dim] = sin_half
            sin_cache[:, half_dim : half_dim * 2] = sin_half

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for given positions."""
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len, device=x.device)
        elif self.cos_cached.device != x.device:
            self._update_cache(self.max_seq_len_cached, device=x.device)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]
                sin = sin[0]
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)

        return cos, sin


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ROPE_DIM = 256


def _apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply RoPE rotation using Triton kernel (CUDA) or PyTorch (CPU/fallback)."""
    batch, num_heads, seq_len, _ = x.shape
    BLOCK_D = next_power_of_two(head_dim)   # must cover full head_dim (including passthrough)
    use_triton = x.is_cuda and BLOCK_D <= MAX_ROPE_DIM

    if use_triton:
        x_f = x.reshape(batch * num_heads, seq_len, head_dim).to(torch.float32).contiguous()
        out  = torch.empty_like(x_f)

        # cos/sin come in as [seq, rotary_dim]; we only need [:seq_len, :half_dim]
        cos_s = cos[:seq_len, :half_dim].to(torch.float32).contiguous()
        sin_s = sin[:seq_len, :half_dim].to(torch.float32).contiguous()

        grid = (batch * num_heads * seq_len,)
        apply_rope_kernel[grid](
            x_f, cos_s, sin_s, out,
            seq_len, half_dim, head_dim,
            x_f.stride(0), x_f.stride(1), x_f.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_D=BLOCK_D,
        )
        return out.reshape(batch, num_heads, seq_len, head_dim).to(x.dtype)

    # PyTorch fallback
    cos_sl = cos[:seq_len]
    sin_sl = sin[:seq_len]
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:half_dim * 2]
    cos_e = cos_sl[None, None, :, :]
    sin_e = sin_sl[None, None, :, :]
    x1_rot = x1 * cos_e - x2 * sin_e
    x2_rot = x2 * cos_e + x1 * sin_e
    if head_dim > half_dim * 2:
        return torch.cat([x1_rot, x2_rot, x[..., half_dim * 2:]], dim=-1)
    return torch.cat([x1_rot, x2_rot], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings.
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape

    if rotary_dim is None:
        rotary_dim = head_dim

    half_dim = rotary_dim // 2

    if cos.shape[1] > half_dim:
        cos = cos[:, :half_dim]
        sin = sin[:, :half_dim]

    cos = cos.to(torch.float32).contiguous()
    sin = sin.to(torch.float32).contiguous()

    q_out = _apply_rope_single(q, cos, sin, half_dim, head_dim)
    k_out = _apply_rope_single(k, cos, sin, half_dim, head_dim)

    return q_out.to(q.dtype), k_out.to(k.dtype)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to partial dimensions."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)


if __name__ == "__main__":
    print("Testing Triton RoPE...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=1024)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    cos, sin = rope(q)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"Q rotated shape: {q_rot.shape}")
    print(f"K rotated shape: {k_rot.shape}")

    print("\nTesting partial RoPE (50%):")
    rope_partial = RotaryEmbedding(dim=head_dim, partial_rotary_factor=0.5)
    cos_p, sin_p = rope_partial(q)
    q_rot_p, k_rot_p = apply_partial_rotary_pos_emb(q, k, cos_p, sin_p, head_dim // 2)
    print(f"Q rotated (partial) shape: {q_rot_p.shape}")

    # ── Numerical correctness: compare Triton kernel against PyTorch reference ──
    print("\nNumerical correctness tests:")
    import torch

    def rope_pytorch_ref(x, cos, sin, half_dim, head_dim):
        """Pure PyTorch RoPE reference."""
        cos_sl = cos[:x.shape[2]]
        sin_sl = sin[:x.shape[2]]
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:half_dim * 2]
        cos_e = cos_sl[None, None, :, :]
        sin_e = sin_sl[None, None, :, :]
        x1_rot = x1 * cos_e - x2 * sin_e
        x2_rot = x2 * cos_e + x1 * sin_e
        if head_dim > half_dim * 2:
            return torch.cat([x1_rot, x2_rot, x[..., half_dim * 2:]], dim=-1)
        return torch.cat([x1_rot, x2_rot], dim=-1)

    # Test 1: full RoPE — text decoder dims (head_dim=128, half_dim=64, no passthrough)
    q_dec = torch.randn(1, 28, 16, 128, device=device)
    rope_dec = RotaryEmbedding(dim=128, max_position_embeddings=256)
    cos_d, sin_d = rope_dec(q_dec)
    cos_d = cos_d.to(device); sin_d = sin_d.to(device)
    cos_half = cos_d[:, :64].to(torch.float32).contiguous()
    sin_half = sin_d[:, :64].to(torch.float32).contiguous()
    triton_out = _apply_rope_single(q_dec, cos_half, sin_half, 64, 128)
    ref_out    = rope_pytorch_ref(q_dec.to(torch.float32), cos_half, sin_half, 64, 128)
    err1 = (triton_out.float() - ref_out).abs().max().item()
    status1 = "PASS" if err1 < 1e-4 else "FAIL"
    print(f"  Full RoPE (head_dim=128, half_dim=64): max_err={err1:.6f}  [{status1}]")

    # Test 2: partial RoPE — audio encoder dims (head_dim=64, rotary_dim=32, half_dim=16, passthrough=32)
    q_enc = torch.randn(1, 20, 375, 64, device=device)
    rope_enc = RotaryEmbedding(dim=64, partial_rotary_factor=0.5)
    cos_e2, sin_e2 = rope_enc(q_enc)
    cos_e2 = cos_e2.to(device); sin_e2 = sin_e2.to(device)
    cos_h2 = cos_e2[:, :16].to(torch.float32).contiguous()
    sin_h2 = sin_e2[:, :16].to(torch.float32).contiguous()
    triton_out2 = _apply_rope_single(q_enc, cos_h2, sin_h2, 16, 64)
    ref_out2    = rope_pytorch_ref(q_enc.to(torch.float32), cos_h2, sin_h2, 16, 64)
    err2 = (triton_out2.float() - ref_out2).abs().max().item()
    status2 = "PASS" if err2 < 1e-4 else "FAIL"
    print(f"  Partial RoPE (head_dim=64, half_dim=16, passthrough=32): max_err={err2:.6f}  [{status2}]")

    print("\nTriton RoPE working!")

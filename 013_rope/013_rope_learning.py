"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
===============================================================================
rope_learning_module.py — Rotary Position Embedding (RoPE), complete module
===============================================================================


(ELI5)
------
Imagine each word holds a little flag on a circular track.  We rotate the
flag by an angle that depends on where the word sits in the sentence.  When
two words "talk" in attention, they compare their flags — and since flags
differ by the angle between positions, the comparison automatically depends
on the *distance* between the words, not on absolute indices.  That's RoPE.

Purpose
-------
Educational, production-quality implementation of RoPE:
  * Correct mathematical core (rotation of 2-D pairs, per-head, per-batch).
  * Happy-path demonstration with synthetic Q/K tensors.
  * Edge cases (odd head_dim, seq > cache, empty seq, wrong dtype, bad theta).
  * Input validation with actionable error messages.
  * Verification of the two properties that make RoPE work:
       1. Norm preservation:  ||RoPE(x, pos)|| == ||x||
       2. Relative-position dependence:
          <RoPE(q,m), RoPE(k,n)> depends only on (m - n)
  * ELI5 walkthrough with 2-D toy vectors.
  * Matplotlib figures saved to disk (fig_rope_rotation.png,
    fig_rope_relative_position.png, fig_rope_scaling.png).

Design principles
-----------------
- DRY: a single `_build_cache` and a single `_get_cos_sin` power all
  scaling modes; `rotate()` is the only public entry point.
- Dataclasses: every stateful concept is a `@dataclass` so it is trivially
  instantiable, inspectable, and comparable.
- Fail loud, fail early: input validation happens once, up front.

Installation
------------
    pip3 install torch numpy pandas matplotlib

Usage
-----
    python3 rope_learning_module.py

Dependencies
------------
Python 3.10+, torch, numpy, pandas, matplotlib.
PyTorch is used (rather than only NumPy) because RoPE lives inside attention
heads whose tensors are PyTorch by convention; using torch keeps the code
directly droppable into a real model.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ----------------------------------------------------------------------
# Third-party
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Reproducibility — RoPE itself is deterministic, but synthetic data
# generation and any bootstrap tests should be reproducible.
torch.manual_seed(42)
np.random.seed(42)


# ======================================================================
# SECTION 1 — CONFIGURATION
# ======================================================================

@dataclass
class RoPEConfig:
    """
    Hyperparameters for a RoPE layer.

    Attributes
    ----------
    head_dim : int
        Dimension of each attention head's Q/K vectors.  Must be even
        because RoPE rotates *pairs* of dimensions.
    max_seq_len : int
        Maximum sequence length to precompute cos/sin caches for.  If a
        longer sequence is passed, the cache is extended lazily.
    theta : float
        Base frequency.  Larger values lengthen the "wavelength" of the
        slowest-rotating dimensions, which improves long-context
        extrapolation.  Llama-2 uses 10,000; Llama-3 uses 500,000.
    scaling_type : str
        {"none", "linear"}.  "none" = vanilla RoPE.  "linear" = divide
        positions by `scaling_factor` before computing angles (the
        simplest long-context trick).  NTK/YaRN are exercise material.
    scaling_factor : float
        Used only when `scaling_type == "linear"`.  Must be >= 1.0.
    partial_rotary_fraction : float
        Fraction of dimension-pairs to rotate.  `1.0` rotates all pairs.
        Values < 1.0 leave the remaining pairs untouched (used in some
        multimodal models to preserve content dimensions).
    """

    head_dim: int
    max_seq_len: int = 8192
    theta: float = 500_000.0
    scaling_type: str = "none"
    scaling_factor: float = 1.0
    partial_rotary_fraction: float = 1.0

    def __post_init__(self) -> None:
        """Validate every field once, up front."""
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be a positive even integer, got {self.head_dim}. "
                "RoPE rotates pairs of dimensions."
            )
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}.")
        if self.theta <= 0:
            raise ValueError(f"theta must be > 0, got {self.theta}.")
        if self.scaling_type not in {"none", "linear"}:
            raise ValueError(
                f"scaling_type must be 'none' or 'linear', got {self.scaling_type!r}. "
                "NTK/YaRN are left as exercises."
            )
        if self.scaling_type == "linear" and self.scaling_factor < 1.0:
            raise ValueError(
                f"scaling_factor must be >= 1.0 for 'linear' scaling, "
                f"got {self.scaling_factor}."
            )
        if not (0.0 < self.partial_rotary_fraction <= 1.0):
            raise ValueError(
                "partial_rotary_fraction must be in (0.0, 1.0], "
                f"got {self.partial_rotary_fraction}."
            )


# ======================================================================
# SECTION 2 — CORE RoPE IMPLEMENTATION
# ======================================================================

class RotaryPositionEmbedding:
    """
    Rotary Position Embedding (RoPE) as in Su et al., 2021.

    Given Q or K of shape [B, H, L, d_head], `rotate` returns a tensor of
    the same shape in which each pair `(x_{2i}, x_{2i+1})` has been rotated
    by an angle `pos * freq_i`, where
        freq_i = 1 / theta ** (2i / d_head).
    """

    def __init__(self, config: RoPEConfig) -> None:
        self.config = config
        # Caches are lazily (re)built by `_build_cache` and stored so
        # `rotate` is a pure function of its inputs afterwards.
        self._cos_cache: Optional[torch.Tensor] = None  # shape [max_seq_len, d_head/2]
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_len: int = 0
        self._build_cache(self.config.max_seq_len)

    # ------------------------------------------------------------------
    # Cache management — shared by every scaling mode (DRY)
    # ------------------------------------------------------------------
    def _build_cache(self, seq_len: int) -> None:
        """
        Precompute cos/sin tables up to `seq_len` (overwrites any existing
        cache — deliberately simple; extend-only path is `_ensure_cache`).
        """
        d = self.config.head_dim
        # inv_freq has shape [d/2].  This is the frequency for each pair.
        inv_freq = 1.0 / (
            self.config.theta ** (torch.arange(0, d, 2).float() / d)
        )

        # Position-dependent scalar used to phase the rotation.
        pos = torch.arange(seq_len).float()
        if self.config.scaling_type == "linear":
            # Linear scaling simply compresses positions.
            pos = pos / self.config.scaling_factor

        # Outer product -> [seq_len, d/2] angular phases.
        angles = torch.outer(pos, inv_freq)
        self._cos_cache = torch.cos(angles)      # [seq_len, d/2]
        self._sin_cache = torch.sin(angles)      # [seq_len, d/2]
        self._cache_len = seq_len

    def _ensure_cache(self, seq_len: int) -> None:
        """Extend the cache if a longer sequence is requested."""
        if seq_len > self._cache_len:
            self._build_cache(seq_len)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(x: torch.Tensor, head_dim: int) -> None:
        """Fail with actionable messages instead of cryptic shape errors."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Expected a torch.Tensor, got {type(x).__name__}. "
                "Convert inputs with torch.as_tensor(...)."
            )
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4-D tensor [B, H, L, d_head], got shape {tuple(x.shape)}."
            )
        if x.shape[-1] != head_dim:
            raise ValueError(
                f"Last dimension must equal head_dim={head_dim}, "
                f"got {x.shape[-1]}."
            )
        if not x.is_floating_point():
            raise TypeError(
                f"Expected a floating-point tensor, got dtype {x.dtype}. "
                "RoPE operates in real-valued vector space."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rotate(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply RoPE to `x`.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, H, L, d_head]; Q or K tensor from an attention head.
        position_ids : torch.Tensor, optional
            1-D long tensor of length L, giving the *absolute* position of
            each token.  If None, positions are assumed to be 0..L-1.

        Returns
        -------
        torch.Tensor
            Same shape and dtype as `x`, with rotations applied to the
            first `2 * int(partial_rotary_fraction * d_head / 2)` dimensions.
        """
        self._validate(x, self.config.head_dim)

        B, H, L, d = x.shape
        if L == 0:
            # Degenerate case: attention on an empty sequence.
            return x.clone()

        self._ensure_cache(L)

        # Build the position index used to gather from the cos/sin cache.
        if position_ids is None:
            pos = torch.arange(L, device=x.device)
        else:
            if position_ids.dim() != 1 or position_ids.shape[0] != L:
                raise ValueError(
                    f"position_ids must be 1-D of length L={L}, "
                    f"got shape {tuple(position_ids.shape)}."
                )
            pos = position_ids.to(device=x.device, dtype=torch.long)

        # How many dimension-pairs to rotate (the rest pass through).
        n_rot_pairs = int(self.config.partial_rotary_fraction * (d // 2))
        if n_rot_pairs == 0:
            return x.clone()

        # Gather cos/sin for this batch's positions.
        # Shapes: [L, n_rot_pairs] -> [L, 1, n_rot_pairs] for broadcasting.
        cos = self._cos_cache[: max(L, int(pos.max().item()) + 1)][pos]
        sin = self._sin_cache[: max(L, int(pos.max().item()) + 1)][pos]
        cos = cos[:, :n_rot_pairs].unsqueeze(1)   # [L, 1, n_rot_pairs]
        sin = sin[:, :n_rot_pairs].unsqueeze(1)

        # Split last dim into (even, odd) pairs — interleaved convention.
        x1 = x[..., 0 : 2 * n_rot_pairs : 2]      # [B, H, L, n_rot_pairs]
        x2 = x[..., 1 : 2 * n_rot_pairs : 2]
        rest = x[..., 2 * n_rot_pairs :]          # untouched tail

        # Apply the 2-D rotation.
        #   [x1']   [ cos  -sin ] [x1]
        #   [x2'] = [ sin   cos ] [x2]
        rot1 = x1 * cos - x2 * sin                # broadcast over B, H
        rot2 = x1 * sin + x2 * cos

        # Re-interleave back into the last dimension.
        # torch.stack([rot1, rot2], dim=-1) -> [B, H, L, n_rot_pairs, 2]
        # .flatten(-2)                         -> [B, H, L, 2*n_rot_pairs]
        rotated = torch.stack([rot1, rot2], dim=-1).flatten(-2)

        if n_rot_pairs == d // 2:
            return rotated
        return torch.cat([rotated, rest], dim=-1)


# ======================================================================
# SECTION 3 — UTILITIES
# ======================================================================

def generate_synthetic_qk(
    batch_size: int = 2,
    num_heads: int = 4,
    seq_len: int = 16,
    head_dim: int = 8,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic Q/K tensors for teaching RoPE.

    In a real transformer these would be the outputs of learned linear
    projections; here we draw them from N(0, 1) with a fixed seed so that
    every learner sees identical numbers.

    Returns
    -------
    (q, k) : Tuple[torch.Tensor, torch.Tensor]
        Each of shape [batch_size, num_heads, seq_len, head_dim].
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, generator=g)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, generator=g)
    return q, k


def attention_logits(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Raw scaled dot-product attention scores (no softmax).

    Parameters
    ----------
    q, k : torch.Tensor of shape [B, H, Lq, d] and [B, H, Lk, d]

    Returns
    -------
    torch.Tensor of shape [B, H, Lq, Lk]
        `logits[b, h, m, n] = <q[b, h, m], k[b, h, n]> / sqrt(d)`.
    """
    d = q.shape[-1]
    return torch.einsum("bhld,bhnd->bhl n", q, k).clone() if False else (
        torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
    )


# ======================================================================
# SECTION 4 — CORRECTNESS CHECKS (the sanity-check pattern)
# ======================================================================

def verify_norm_preservation(
    head_dim: int = 8, seq_len: int = 16, theta: float = 10_000.0
) -> float:
    """
    Verify that ||RoPE(x, pos)|| == ||x|| for every position.

    Why this matters: the rotation is orthogonal, so the attention
    logit magnitude is preserved — only its direction (via phase shift)
    is changed.  This is the mathematical property behind
    "no extra scaling of attention logits" in RoPE-based LLMs.

    Returns
    -------
    max_abs_error : float
        Should be ~1e-6 or smaller (float32 rounding).
    """
    rope = RotaryPositionEmbedding(RoPEConfig(head_dim=head_dim, theta=theta))
    q, _ = generate_synthetic_qk(seq_len=seq_len, head_dim=head_dim, seed=7)
    q_rot = rope.rotate(q)
    norm_before = q.norm(dim=-1)
    norm_after = q_rot.norm(dim=-1)
    return float((norm_before - norm_after).abs().max().item())


def verify_relative_position_property(
    head_dim: int = 8,
    seq_len: int = 16,
    theta: float = 10_000.0,
    tol: float = 1e-5,
) -> float:
    """
    Verify that <RoPE(q, m), RoPE(k, n)> depends only on (m - n).

    Method: pick one fixed q and k content vector, embed them at every
    (m, n) pair, and check that entries with equal `m - n` agree.

    Returns
    -------
    max_deviation : float
        Maximum deviation between logits sharing the same `m - n`.
        Should be < `tol` (float32 rounding accumulates a little).
    """
    rope = RotaryPositionEmbedding(
        RoPEConfig(head_dim=head_dim, max_seq_len=seq_len, theta=theta)
    )
    # Single fixed content vector for q and k.
    g = torch.Generator().manual_seed(123)
    q0 = torch.randn(1, 1, 1, head_dim, generator=g)
    k0 = torch.randn(1, 1, 1, head_dim, generator=g)

    logits = torch.zeros(seq_len, seq_len)
    for m in range(seq_len):
        q_m = rope.rotate(q0, position_ids=torch.tensor([m]))
        for n in range(seq_len):
            k_n = rope.rotate(k0, position_ids=torch.tensor([n]))
            logits[m, n] = (q_m * k_n).sum()

    # Group by (m - n) and measure within-group spread.
    max_dev = 0.0
    for delta in range(-(seq_len - 1), seq_len):
        vals = [logits[m, n].item() for m in range(seq_len)
                                    for n in range(seq_len)
                                    if m - n == delta]
        if len(vals) > 1:
            spread = max(vals) - min(vals)
            max_dev = max(max_dev, spread)
    return max_dev


# ======================================================================
# SECTION 5 — DEMO SCENARIOS
# ======================================================================

def demo_happy_path() -> pd.DataFrame:
    """
    Canonical use: rotate Q and K, compute attention logits, report
    shapes and norms.  This is the "everything works" scenario.
    """
    print("\n--- Demo: happy path ---")
    cfg = RoPEConfig(head_dim=8, max_seq_len=32, theta=500_000.0)
    rope = RotaryPositionEmbedding(cfg)
    q, k = generate_synthetic_qk(batch_size=2, num_heads=4, seq_len=16,
                                 head_dim=8, seed=42)

    q_rot = rope.rotate(q)
    k_rot = rope.rotate(k)
    logits = attention_logits(q_rot, k_rot)

    assert q_rot.shape == q.shape, "rotate changed the shape!"
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)

    rows = [
        dict(stage="input q", shape=tuple(q.shape), norm=q.norm().item()),
        dict(stage="rotated q", shape=tuple(q_rot.shape), norm=q_rot.norm().item()),
        dict(stage="input k", shape=tuple(k.shape), norm=k.norm().item()),
        dict(stage="rotated k", shape=tuple(k_rot.shape), norm=k_rot.norm().item()),
        dict(stage="attention logits", shape=tuple(logits.shape), norm=logits.norm().item()),
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def demo_edge_case_odd_head_dim() -> None:
    """head_dim must be even; show the actionable error message."""
    print("\n--- Edge case: odd head_dim ---")
    try:
        RoPEConfig(head_dim=7)
    except ValueError as exc:
        print(f"  Raised ValueError as expected:\n    {exc}")
    else:
        raise AssertionError("Expected ValueError was not raised.")


def demo_edge_case_sequence_too_long() -> None:
    """
    Requesting seq_len > max_seq_len should lazily extend the cache,
    not crash — and the result must remain correct (norm-preserving).
    """
    print("\n--- Edge case: sequence longer than cache ---")
    cfg = RoPEConfig(head_dim=8, max_seq_len=8, theta=10_000.0)
    rope = RotaryPositionEmbedding(cfg)
    q, _ = generate_synthetic_qk(seq_len=32, head_dim=8, seed=1)
    q_rot = rope.rotate(q)
    print(f"  q.shape = {tuple(q.shape)}; cache auto-extended to "
          f"{rope._cache_len}.")
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    print("  Norm preserved across all 32 positions: ✓")


def demo_edge_case_empty_sequence() -> None:
    """Attention on an empty sequence: rotate returns a clone, no error."""
    print("\n--- Edge case: empty sequence (L=0) ---")
    rope = RotaryPositionEmbedding(RoPEConfig(head_dim=8, max_seq_len=16))
    q = torch.empty(2, 4, 0, 8)
    q_rot = rope.rotate(q)
    print(f"  q.shape = {tuple(q.shape)}; q_rot.shape = {tuple(q_rot.shape)}")
    assert q_rot.shape == q.shape
    print("  Empty sequence handled gracefully.")


def demo_edge_case_wrong_type() -> None:
    """Non-tensor input must raise a clear TypeError."""
    print("\n--- Edge case: wrong input type ---")
    rope = RotaryPositionEmbedding(RoPEConfig(head_dim=8))
    for bad_input, label in [
        (np.zeros((1, 1, 4, 8), dtype=np.float32), "numpy array"),
        ("not a tensor", "string"),
        ([[[1.0]]], "nested list"),
    ]:
        try:
            rope.rotate(bad_input)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            print(f"  {label:>12s}: raised {type(exc).__name__}: {exc}")
        else:
            raise AssertionError(f"Expected failure for {label}")


def demo_edge_case_integer_dtype() -> None:
    """Integer tensors must be rejected (rotations are real-valued)."""
    print("\n--- Edge case: integer dtype ---")
    rope = RotaryPositionEmbedding(RoPEConfig(head_dim=8))
    q_int = torch.zeros(1, 1, 4, 8, dtype=torch.int32)
    try:
        rope.rotate(q_int)
    except TypeError as exc:
        print(f"  Raised TypeError as expected:\n    {exc}")


def demo_partial_rotary() -> None:
    """
    partial_rotary_fraction=0.5 rotates only the first half of the
    dimension-pairs; the untouched tail must remain identical.
    """
    print("\n--- Demo: partial rotary fraction (0.5) ---")
    cfg = RoPEConfig(head_dim=8, partial_rotary_fraction=0.5)
    rope = RotaryPositionEmbedding(cfg)
    q, _ = generate_synthetic_qk(seq_len=8, head_dim=8, seed=3)
    q_rot = rope.rotate(q)
    # Tail (last 4 dims) must equal original tail.
    assert torch.allclose(q[..., 4:], q_rot[..., 4:]), "tail was modified!"
    # Head (first 4 dims) must differ (rotation happened).
    assert not torch.allclose(q[..., :4], q_rot[..., :4]), "head was unchanged!"
    print("  First 4 dims rotated, last 4 dims unchanged: ✓")


def demo_linear_scaling() -> None:
    """
    Compare vanilla (theta=10k) vs linear-scaled RoPE at long positions.
    Scaling does not change norm preservation but does change angles.
    """
    print("\n--- Demo: linear scaling (long context) ---")
    seq_len = 64
    base = RotaryPositionEmbedding(RoPEConfig(head_dim=8, max_seq_len=seq_len,
                                              theta=10_000.0))
    scaled = RotaryPositionEmbedding(RoPEConfig(head_dim=8, max_seq_len=seq_len,
                                                theta=10_000.0,
                                                scaling_type="linear",
                                                scaling_factor=4.0))
    q, _ = generate_synthetic_qk(seq_len=seq_len, head_dim=8, seed=11)
    q_base = base.rotate(q)
    q_scaled = scaled.rotate(q)
    max_diff = float((q_base - q_scaled).abs().max().item())
    print(f"  max |base - scaled| over all dims: {max_diff:.4f}")
    print("  Norm preserved in both:",
          torch.allclose(q.norm(dim=-1), q_scaled.norm(dim=-1), atol=1e-5))
    print("  Linear scaling compresses rotation angles by factor 4 — useful")
    print("  when extrapolating to sequences 4x longer than training.")


# ======================================================================
# SECTION 6 — VISUALIZATIONS
# ======================================================================

def plot_rotation_curves(save_to: str = "fig_rope_rotation.png") -> None:
    """
    Figure (a)/(b): cos and sin curves for several frequency indices,
    showing that high-frequency dims rotate fast and low-frequency dims
    rotate slowly.  This is the "example input / processing step" view.
    """
    head_dim = 16
    seq_len = 128
    theta = 10_000.0

    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    angles = torch.outer(pos, inv_freq)   # [seq_len, head_dim/2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    idxs = [0, head_dim // 4, head_dim // 2 - 1]

    for i in idxs:
        axes[0].plot(pos.numpy(), angles[:, i].cos().numpy(),
                     label=f"cos (pair {i})")
    axes[0].set_title("RoPE rotation — cosine components")
    axes[0].set_xlabel("Position in sequence")
    axes[0].set_ylabel("cos(θ · pos)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for i in idxs:
        axes[1].plot(pos.numpy(), angles[:, i].sin().numpy(),
                     label=f"sin (pair {i})")
    axes[1].set_title("RoPE rotation — sine components")
    axes[1].set_xlabel("Position in sequence")
    axes[1].set_ylabel("sin(θ · pos)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "RoPE: each dimension-pair rotates at a different frequency "
        "(θ=10,000; head_dim=16)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[Viz] Saved: {save_to}")


def plot_relative_position_heatmap(
    save_to: str = "fig_rope_relative_position.png",
) -> None:
    """
    Figure (c): attention logits[m, n] with a single fixed (q, k).
    Because logits depend only on m − n, every anti-diagonal is constant —
    the "outcome" view that proves RoPE's relative-position property.
    """
    seq_len = 32
    rope = RotaryPositionEmbedding(
        RoPEConfig(head_dim=8, max_seq_len=seq_len, theta=10_000.0)
    )
    g = torch.Generator().manual_seed(7)
    q0 = torch.randn(1, 1, 1, 8, generator=g)
    k0 = torch.randn(1, 1, 1, 8, generator=g)

    logits = torch.zeros(seq_len, seq_len)
    for m in range(seq_len):
        q_m = rope.rotate(q0, position_ids=torch.tensor([m]))
        for n in range(seq_len):
            k_n = rope.rotate(k0, position_ids=torch.tensor([n]))
            logits[m, n] = (q_m * k_n).sum()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(logits.numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_xlabel("K position (n)")
    ax.set_ylabel("Q position (m)")
    ax.set_title(
        "Attention logit <RoPE(q,m), RoPE(k,n)>\n"
        "Constant along each anti-diagonal (depends only on m−n)"
    )
    fig.colorbar(im, ax=ax, label="dot product")
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[Viz] Saved: {save_to}")


def plot_scaling_comparison(save_to: str = "fig_rope_scaling.png") -> None:
    """
    Figure: how the *effective attention window* differs between vanilla
    RoPE and linearly-scaled RoPE.  Plots |<RoPE(q,0), RoPE(k,n)>| for
    n = 0..N.  Vanilla RoPE oscillates rapidly (attends only locally);
    scaled RoPE oscillates slower (attends over longer distances).
    """
    seq_len = 256
    vanilla = RotaryPositionEmbedding(
        RoPEConfig(head_dim=8, max_seq_len=seq_len, theta=10_000.0)
    )
    scaled = RotaryPositionEmbedding(
        RoPEConfig(head_dim=8, max_seq_len=seq_len, theta=10_000.0,
                   scaling_type="linear", scaling_factor=8.0)
    )
    g = torch.Generator().manual_seed(99)
    q = torch.randn(1, 1, 1, 8, generator=g)
    k = torch.randn(1, 1, 1, 8, generator=g)

    q_v = vanilla.rotate(q, position_ids=torch.tensor([0]))
    q_s = scaled.rotate(q, position_ids=torch.tensor([0]))

    positions = torch.arange(seq_len)
    score_v, score_s = [], []
    for n in range(seq_len):
        k_v = vanilla.rotate(k, position_ids=torch.tensor([n]))
        k_s = scaled.rotate(k, position_ids=torch.tensor([n]))
        score_v.append((q_v * k_v).sum().item())
        score_s.append((q_s * k_s).sum().item())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(positions.numpy(), score_v, label="vanilla RoPE (θ=10k)",
            alpha=0.85)
    ax.plot(positions.numpy(), score_s, label="linear scaling ×8 (θ=10k)",
            alpha=0.85)
    ax.set_xlabel("K position (n)  — Q is fixed at position 0")
    ax.set_ylabel("attention logit")
    ax.set_title(
        "RoPE frequency scaling stretches the attention window\n"
        "(compare how quickly the two curves decorrelate)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[Viz] Saved: {save_to}")


# ======================================================================
# SECTION 7 — ELI5 CODE WALKTHROUGH
# ======================================================================

def eli5_walkthrough() -> None:
    """
    The plain-language tutorial.  Runs on 2-D toy vectors so the math is
    human-visible: rotating a single arrow on a page.
    """
    print("\n" + "=" * 70)
    print(" ELI5 walkthrough — RoPE on a single 2-D arrow")
    print("=" * 70)
    print("""
Imagine a word's Q vector is an arrow on a page.  In a real model the
arrow lives in 8, 16, 128 dimensions — but 2-D is enough to see what
RoPE does: it ROTATES the arrow by an angle proportional to the word's
position.  Two words then "compare" their arrows (dot product), and the
comparison depends only on the angle between the arrows — i.e., on how
far apart the words are.
""")
    theta = 100.0          # small theta => visible rotation for teaching
    seq_len = 4
    # inv_freq for a 2-D "head": one pair => one frequency = 1.
    # We'll use theta=100 and explicitly compute angle = pos / theta to see
    # the effect in a 2-D plot; the actual RoPE formula with head_dim=2 is
    # identical for the single pair.
    print(f"Setup: 2-D 'head_dim=2', theta={theta}, positions 0..{seq_len-1}.")

    base_vec = torch.tensor([[1.0, 0.0]])   # arrow pointing right
    for pos in range(seq_len):
        angle = pos / theta
        c, s = math.cos(angle), math.sin(angle)
        # 2-D rotation matrix
        rot = torch.tensor([[c, -s], [s, c]])
        rotated = base_vec @ rot.T
        x, y = rotated[0].tolist()
        print(f"  position {pos}: angle = {angle:.4f} rad -> "
              f"rotated arrow = ({x:+.4f}, {y:+.4f})")

    print("""
Key insight:
  * The magnitude of the arrow never changes — only its direction.
  * A word at position 3 is "rotated more" than a word at position 0.
  * If word A is at position 0 and word B is at position 3, their arrows
    differ by the same angle whether the sentence starts at position 0
    or at position 10,000 — only the DIFFERENCE matters.

That is RoPE in a nutshell.  Real LLMs stack ~64 such pairs per head,
each pair rotating at a different frequency (theta controls the spread
of those frequencies).""")
    print("=" * 70)


# ======================================================================
# SECTION 8 — EXERCISES
# ======================================================================

def list_exercises() -> None:
    """Print the three learner exercises (solutions in the companion file)."""
    print("\n" + "=" * 70)
    print(" EXERCISES")
    print("=" * 70)
    print("""
Exercise 1 — NTK-aware RoPE scaling.
  Implement NTK-aware scaling: replace theta with
        theta' = theta * scaling_factor ** (d / (d - 2))
  and rerun the `plot_scaling_comparison` figure.  Hint: reuse
  `_build_cache`; you only need to change the effective theta.
  Verify that the norm-preservation property still holds.

Exercise 2 — Half-split vs interleaved convention.
  RoPE's pair-ordering is a training choice.  The published RoFormer
  paper uses the HALF-SPLIT convention: pair (i, i + d/2) instead of
  (2i, 2i+1).  Implement `rotate_half_split` and show that
        rotate_interleaved(q)  ≠  rotate_half_split(q)
  as tensors, but both preserve norms and both satisfy the relative
  position property.  Hint: split at d/2, not at even indices.

Exercise 3 — Attention-temperature sanity check.
  In YaRN, attention logits are further multiplied by a temperature
  `1 / sqrt(t)` before softmax.  Implement a `attention_with_temperature`
  helper and plot the softmax-normalized attention distribution at
  positions 0..N for t=1 and t=4.  Hint: temperature does not change
  *relative-order* of logits — only how peaked the softmax is.

Hints are intentionally not full solutions; see
`rope_exercise_solutions.py` for reference implementations.
""")


# ======================================================================
# SECTION 9 — MAIN
# ======================================================================

def main() -> None:
    """Run every demonstration, verification, and figure in one pass."""
    print("=" * 78)
    print(" RoPE — Complete Learning Module")
    print("=" * 78)

    # ---- Correctness first (this is the "sanity_check" pattern) ----
    print("\n[Sanity] Verifying RoPE's two fundamental properties...")
    norm_err = verify_norm_preservation()
    rel_err = verify_relative_position_property()
    print(f"  Norm preservation error:        {norm_err:.3e}  (expect < 1e-5)")
    print(f"  Relative-position deviation:    {rel_err:.3e}  (expect < 1e-5)")
    assert norm_err < 1e-5, "norm preservation failed!"
    assert rel_err < 1e-5, "relative-position property failed!"
    print("  ✓ RoPE passes both sanity checks.")

    # ---- Scenarios ----
    demo_happy_path()
    demo_edge_case_odd_head_dim()
    demo_edge_case_sequence_too_long()
    demo_edge_case_empty_sequence()
    demo_edge_case_wrong_type()
    demo_edge_case_integer_dtype()
    demo_partial_rotary()
    demo_linear_scaling()

    # ---- ELI5 tutorial ----
    eli5_walkthrough()

    # ---- Figures ----
    print("\n[Viz] Generating figures...")
    plot_rotation_curves()
    plot_relative_position_heatmap()
    plot_scaling_comparison()

    # ---- Exercises ----
    list_exercises()

    print("\n" + "=" * 78)
    print(" DONE — figures: fig_rope_rotation.png, "
          "fig_rope_relative_position.png, fig_rope_scaling.png")
    print("=" * 78)


if __name__ == "__main__":
    main()

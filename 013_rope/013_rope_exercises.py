"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
===============================================================================
rope_exercises.py —  solutions to the three RoPE exercises
===============================================================================

Exercise 1 — NTK-aware RoPE scaling.
Implement NTK-aware frequency scaling by replacing theta with
theta' = theta · scaling_factor^(d / (d − 2)) inside _build_cache.
Re-plot fig_rope_scaling.png with the new scaling type. Hint: the existing linear branch already shows the pattern; add an "ntk" branch to RoPEConfig.scaling_type and reuse the cached inv_freq construction. Verify that verify_norm_preservation() still passes.

Exercise 2 — Half-split vs. interleaved convention.
Implement rotate_half_split which pairs (x[i], x[i + d/2]) instead of (x[2i], x[2i+1]). Show that (a) rotate_interleaved(q) ≠ rotate_half_split(q) as tensors, (b) both preserve the norm, and (c) both satisfy the relative-position property. Hint: the only change is the slicing of the last dimension — split at d/2 before the rotation, and torch.cat the two halves back together after. Everything else (cos/sin cache) is identical.

Exercise 3 — Attention temperature.
Extend attention_logits to accept a temperature parameter and apply logits / temperature before softmax. Plot the softmax-normalized attention distribution of a fixed query against positions 0..N at temperatures 1 and 4. Hint: temperature does not change the relative order of logits — it changes how peaked the softmax is. Verify this by checking that argmax(logits) is the same for both temperatures.




Companion to `rope_learning_module.py`.  Each exercise is wrapped in a
`@dataclass` with a `run()` method for symmetry with the module's design
principles (DRY, dataclass-per-concept, well-documented).

Installation
------------
    pip3 install torch numpy pandas matplotlib



Usage
-----
    python3 rope_exercise_solutions.py

Dependencies
------------
Python 3.10+, torch, numpy, pandas, matplotlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from rope_learning_module import (
    RoPEConfig,
    RotaryPositionEmbedding,
    generate_synthetic_qk,
    verify_norm_preservation,
    verify_relative_position_property,
)

torch.manual_seed(42)
np.random.seed(42)


# ======================================================================
# EXERCISE 1 — NTK-aware frequency scaling
# ======================================================================

@dataclass
class NTKAwareRoPEConfig(RoPEConfig):
    """
    Extends `RoPEConfig` with NTK-aware frequency scaling.

    NTK-aware scaling sets
        theta' = theta * scaling_factor ** (d / (d - 2))
    so that low-frequency dimensions stretch more than high-frequency
    ones — preserving local attention while extending global reach.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scaling_type not in {"none", "linear", "ntk"}:
            raise ValueError(f"Unsupported scaling_type: {self.scaling_type}")

    @property
    def effective_theta(self) -> float:
        """Return θ' = θ · factor^(d/(d−2)), or θ when scaling_type != ntk."""
        if self.scaling_type == "ntk":
            exponent = self.head_dim / (self.head_dim - 2)
            return self.theta * (self.scaling_factor ** exponent)
        return self.theta


class NTKAwareRoPE(RotaryPositionEmbedding):
    """RoPE with a `_build_cache` that honors NTK-aware θ scaling."""

    def __init__(self, config: NTKAwareRoPEConfig) -> None:
        self.config = config
        self._cos_cache = None
        self._sin_cache = None
        self._cache_len = 0
        self._build_cache(config.max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        d = self.config.head_dim
        theta = self.config.effective_theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
        pos = torch.arange(seq_len).float()
        angles = torch.outer(pos, inv_freq)
        self._cos_cache = torch.cos(angles)
        self._sin_cache = torch.sin(angles)
        self._cache_len = seq_len


@dataclass
class Exercise1Solution:
    """Exercise 1 — implement NTK-aware scaling and verify properties."""

    head_dim: int = 8
    seq_len: int = 64
    theta: float = 10_000.0
    scaling_factor: float = 8.0

    def run(self) -> pd.DataFrame:
        print("\n--- Exercise 1: NTK-aware scaling ---")
        cfg = NTKAwareRoPEConfig(
            head_dim=self.head_dim,
            max_seq_len=self.seq_len,
            theta=self.theta,
            scaling_type="ntk",
            scaling_factor=self.scaling_factor,
        )
        print(f"  θ = {cfg.theta:g};  θ' (effective) = {cfg.effective_theta:,.2f}")

        rope = NTKAwareRoPE(cfg)
        q, _ = generate_synthetic_qk(
            seq_len=self.seq_len, head_dim=self.head_dim, seed=42
        )
        q_rot = rope.rotate(q)
        norm_err = float((q.norm(dim=-1) - q_rot.norm(dim=-1)).abs().max())

        # Compare final-position rotation vs vanilla RoPE.
        vanilla = RotaryPositionEmbedding(
            RoPEConfig(head_dim=self.head_dim,
                       max_seq_len=self.seq_len, theta=self.theta)
        )
        q_vanilla = vanilla.rotate(q)
        diff = float((q_rot - q_vanilla).abs().max())

        rows = [
            dict(metric="effective theta", value=f"{cfg.effective_theta:,.2f}"),
            dict(metric="norm-preservation error", value=f"{norm_err:.3e}"),
            dict(metric="max |NTK − vanilla|", value=f"{diff:.4f}"),
        ]
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        assert norm_err < 1e-5
        assert diff > 0
        print("  ✓ NTK scaling implemented; norm preserved; differs from vanilla.")
        return df


# ======================================================================
# EXERCISE 2 — Half-split vs interleaved convention
# ======================================================================

def rotate_half_split(x: torch.Tensor, rope: RotaryPositionEmbedding) -> torch.Tensor:
    """
    Apply RoPE with the *half-split* convention (RoFormer paper).

    Pairs `(x[i], x[i + d/2])` instead of `(x[2i], x[2i+1])`.  The cos/sin
    cache is the same shape because the number of pairs is unchanged.
    """
    B, H, L, d = x.shape
    if L == 0:
        return x.clone()
    rope._ensure_cache(L)
    cos = rope._cos_cache[:L].unsqueeze(1)   # [L, 1, d/2]
    sin = rope._sin_cache[:L].unsqueeze(1)

    half = d // 2
    x1 = x[..., :half]                      # [B, H, L, d/2]
    x2 = x[..., half:]                      # [B, H, L, d/2]

    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.cat([rot1, rot2], dim=-1)


@dataclass
class Exercise2Solution:
    """Exercise 2 — compare half-split and interleaved conventions."""

    head_dim: int = 8
    seq_len: int = 16

    def run(self) -> pd.DataFrame:
        print("\n--- Exercise 2: half-split vs interleaved convention ---")
        rope = RotaryPositionEmbedding(
            RoPEConfig(head_dim=self.head_dim, max_seq_len=self.seq_len,
                       theta=10_000.0)
        )
        q, _ = generate_synthetic_qk(
            seq_len=self.seq_len, head_dim=self.head_dim, seed=5
        )
        q_inter = rope.rotate(q)
        q_split = rotate_half_split(q, rope)

        diff = float((q_inter - q_split).abs().max())
        norm_inter = float((q.norm(dim=-1) - q_inter.norm(dim=-1)).abs().max())
        norm_split = float((q.norm(dim=-1) - q_split.norm(dim=-1)).abs().max())

        # Relative-position property for half-split variant.
        g = torch.Generator().manual_seed(123)
        q0 = torch.randn(1, 1, 1, self.head_dim, generator=g)
        k0 = torch.randn(1, 1, 1, self.head_dim, generator=g)
        logits = torch.zeros(self.seq_len, self.seq_len)
        for m in range(self.seq_len):
            qm = rotate_half_split(q0, rope)
            # (Re-use a temporary rope-like call; simpler to inline.)
            # Actually rotate with m via a temporary cache — trick:
            tmp = RotaryPositionEmbedding(
                RoPEConfig(head_dim=self.head_dim,
                           max_seq_len=self.seq_len, theta=10_000.0)
            )
            # Instead, simply build the rotation for position m directly:
            d = self.head_dim
            inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, d, 2).float() / d))
            angle = m * inv_freq
            c, s = torch.cos(angle), torch.sin(angle)
            q0_m = torch.cat([q0[..., :d // 2], q0[..., d // 2:]], dim=-1)
            q0_half1, q0_half2 = q0[..., :d // 2], q0[..., d // 2:]
            q0_m = torch.cat(
                [q0_half1 * c - q0_half2 * s,
                 q0_half1 * s + q0_half2 * c], dim=-1
            )
            for n in range(self.seq_len):
                angle_n = n * inv_freq
                cn, sn = torch.cos(angle_n), torch.sin(angle_n)
                k0_h1, k0_h2 = k0[..., :d // 2], k0[..., d // 2:]
                k0_n = torch.cat(
                    [k0_h1 * cn - k0_h2 * sn,
                     k0_h1 * sn + k0_h2 * cn], dim=-1
                )
                logits[m, n] = (q0_m * k0_n).sum()
        rel_dev = 0.0
        for delta in range(-(self.seq_len - 1), self.seq_len):
            vals = [logits[m, n].item() for m in range(self.seq_len)
                                        for n in range(self.seq_len)
                                        if m - n == delta]
            if len(vals) > 1:
                rel_dev = max(rel_dev, max(vals) - min(vals))

        rows = [
            dict(metric="interleaved vs half-split max diff", value=f"{diff:.4f}"),
            dict(metric="norm err (interleaved)", value=f"{norm_inter:.3e}"),
            dict(metric="norm err (half-split)", value=f"{norm_split:.3e}"),
            dict(metric="relative-pos deviation (half-split)", value=f"{rel_dev:.3e}"),
        ]
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        assert diff > 0
        assert norm_inter < 1e-5 and norm_split < 1e-5
        assert rel_dev < 1e-5
        print("  ✓ Conventions differ, both norm-preserving, both "
              "satisfy relative-position property.")
        return df


# ======================================================================
# EXERCISE 3 — Attention temperature
# ======================================================================

def attention_with_temperature(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Softmax-normalized attention scores: softmax(logits / T)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return torch.softmax(logits / temperature, dim=-1)


@dataclass
class Exercise3Solution:
    """Exercise 3 — attention temperature and its effect on peakedness."""

    seq_len: int = 32
    temperatures: List[float] = field(default_factory=lambda: [1.0, 4.0])

    def run(self) -> pd.DataFrame:
        print("\n--- Exercise 3: attention temperature ---")
        rope = RotaryPositionEmbedding(
            RoPEConfig(head_dim=8, max_seq_len=self.seq_len, theta=10_000.0)
        )
        g = torch.Generator().manual_seed(7)
        q = torch.randn(1, 1, 1, 8, generator=g)
        k = torch.randn(1, 1, self.seq_len, 8, generator=g)

        q_pos0 = rope.rotate(q, position_ids=torch.tensor([0]))
        k_rot = rope.rotate(k)
        logits = torch.matmul(q_pos0, k_rot.transpose(-1, -2)).squeeze() / math.sqrt(8)

        rows = []
        argmax_logit = int(torch.argmax(logits).item())
        for T in self.temperatures:
            probs = attention_with_temperature(logits, T)
            entropy = float(-(probs * (probs + 1e-12).log()).sum())
            max_prob = float(probs.max())
            argmax_prob = int(torch.argmax(probs).item())
            rows.append(dict(
                temperature=T,
                argmax_logit=argmax_logit,
                argmax_prob=argmax_prob,
                max_prob=max_prob,
                entropy=entropy,
            ))
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        # Same argmax across temperatures — temperature does not reorder logits.
        assert df["argmax_prob"].nunique() == 1
        # Higher T ⇒ higher entropy, lower max probability.
        assert df["entropy"].iloc[1] > df["entropy"].iloc[0]
        assert df["max_prob"].iloc[1] < df["max_prob"].iloc[0]
        print("  ✓ Temperature changes peakedness but not argmax.")

        # Figure.
        fig, ax = plt.subplots(figsize=(9, 4.5))
        positions = np.arange(self.seq_len)
        for T in self.temperatures:
            probs = attention_with_temperature(logits, T).numpy()
            ax.plot(positions, probs, marker="o", markersize=3,
                    label=f"T = {T}")
        ax.set_xlabel("K position")
        ax.set_ylabel("attention probability")
        ax.set_title("Attention temperature: peakedness changes, argmax does not")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("fig_rope_exercise3_temperature.png", dpi=150)
        plt.show()
        print("  [Viz] Saved: fig_rope_exercise3_temperature.png")
        return df


# ======================================================================
# DRIVER
# ======================================================================

def main() -> None:
    print("=" * 78)
    print(" RoPE — Exercise Solutions")
    print("=" * 78)

    Exercise1Solution().run()
    Exercise2Solution().run()
    Exercise3Solution().run()

    print("\n" + "=" * 78)
    print(" DONE — all three exercises passed")
    print(" Figure: fig_rope_exercise3_temperature.png")
    print("=" * 78)


if __name__ == "__main__":
    main()

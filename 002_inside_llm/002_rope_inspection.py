"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

ELIM5
Imagine a wheel with a pointer drawn on it. When the wheel is at position 0 (the start), the pointer points straight up — nothing has changed. But as the wheel rolls forward to position 5, 10, 25, the pointer rotates more and more. RoPE (Rotary Positional Embedding) does the same thing to vectors inside the LLM: it "rotates" the vector based on its position in the sentence. The clever bit is that different parts of the vector rotate at different speeds — like having many wheels of different sizes all rolling together. This exercise lets you see the rotation happen by feeding RoPE a totally flat vector [1, 1, 1, ...] and watching it spin at different positions

Packages to install
pip3 install numpy>=1.24, pandas>=2.0, matplotlib>=3.7, scikit-learn>=1.3


Input
A flat vector v = np.ones(32) of dtype float32, inspected at positions [0, 5, 10, 25], with theta=10000, max_seq_len=64.

Outcome
At position 0: the rotated vector equals the input exactly (max_abs_diff = 0). All cos(0) = 1, all sin(0) = 0, so rotation is identity.
At position 5, 10, 25: the vector changes; the later feature pairs rotate faster because their inverse frequency inv_freq[i] = theta^(-2i/d) is larger.
Two saved PNG files:

    fig_rope_overlapping_bars.png — overlapping bar chart of all four positions.
    fig_rope_rotation_angle.png — per-feature rotation angle (in radians) vs. feature index for each position.


3_rope_inspection.py
==============================
Inspect and visualize what Rotary Positional Embeddings (RoPE) do to a vector.

Educational objectives:
    - Verify that RoPE at position 0 is the identity (no rotation).
    - Visualize that rotation grows with position.
    - Visualize that LATER feature pairs rotate FASTER (higher inv_freq).
    - Build intuition for WHY RoPE gives *relative* position encoding:
      rotating Q_m and K_n by angles (a*m) and (a*n) makes their dot
      product depend on (m - n).

Design principles:
    - Single Responsibility: RoPEInspector computes; RoPEPlotter renders.
    - DRY: reuses the RoPE class from chapter3_llm_internals.
    - Reproducibility: every random op seeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from chapter3_llm_internals import RoPE


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class RoPEInspectionConfig:
    """All parameters for the RoPE inspection exercise.

    Attributes:
        vector_dim:   Dimensionality of the inspected vector (must be even).
        positions:    Token positions at which to apply RoPE.
        max_seq_len:  RoPE precompute length (must exceed max(positions)).
        theta:        RoPE base frequency (10000 in Llama).
    """
    vector_dim: int = 32
    positions: List[int] = field(default_factory=lambda: [0, 5, 10, 25])
    max_seq_len: int = 64
    theta: float = 10000.0


# ============================================================================
# 2. INSPECTOR
# ============================================================================

class RoPEInspector:
    """Applies RoPE to a fixed vector at multiple positions and validates behavior."""

    def __init__(self, config: RoPEInspectionConfig):
        self.config = config
        # The flat input vector — a perfect baseline to *see* rotation.
        self.input_vector = np.ones(config.vector_dim, dtype=np.float32)
        # Build the RoPE engine once.
        self.rope = RoPE(config.vector_dim, config.max_seq_len, config.theta)

    def inspect(self) -> dict:
        """Apply RoPE at every configured position; return rotated vectors."""
        rotated_vectors = {}
        for pos in self.config.positions:
            # RoPE.apply expects shape (..., seq_len, hidden_dim).
            # We feed a single token at the chosen position by slicing cos/sin
            # starting at offset=pos. To do that, we pass a (1, hidden_dim)
            # tensor and set seq_len_offset = pos.
            v = self.input_vector.reshape(1, -1)        # shape (1, 32)
            rotated = self.rope.apply(v, seq_len_offset=pos)
            rotated_vectors[pos] = rotated[0]           # back to (32,)
        return rotated_vectors

    def verify_position_zero_identity(self, rotated_vectors: dict) -> dict:
        """At position 0, cos=1 and sin=0 -> output MUST equal input."""
        v0 = rotated_vectors.get(0)
        if v0 is None:
            return {"test": "position 0 identity",
                    "status": "SKIPPED: position 0 not in inspection set"}
        diff = float(np.max(np.abs(v0 - self.input_vector)))
        return {
            "test": "position 0 identity",
            "max_abs_diff": diff,
            "tolerance": 1e-6,
            "passed": diff < 1e-6,
        }

    def compute_rotation_angles(self) -> dict:
        """For each position, compute the rotation angle (radians) per feature pair.

        Returns:
            Dict mapping position -> 1-D array of length hidden_dim/2 with
            angles (radians).
        """
        half = self.config.vector_dim // 2
        # inv_freq[i] = theta^(-2i/d)  (matches RoPE.__init__)
        inv_freq = 1.0 / (self.config.theta ** (
            np.arange(0, half, dtype=np.float32) / self.config.vector_dim
        ))
        angles = {}
        for pos in self.config.positions:
            angles[pos] = pos * inv_freq   # angle[i] = pos * inv_freq[i]
        return angles


# ============================================================================
# 3. PLOTTER
# ============================================================================

class RoPEPlotter:
    """Renders the inspector's results into two figures."""

    COLORS = ["steelblue", "seagreen", "darkorange", "crimson", "purple"]

    @classmethod
    def plot_overlapping_bars(cls, input_vector: np.ndarray,
                              rotated_vectors: dict,
                              save_path: str = "fig_rope_overlapping_bars.png"
                              ) -> None:
        """Overlay rotated vectors for every position on a single bar chart."""
        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(input_vector))
        width = 0.18

        # Plot the original (input) vector as a dashed reference.
        ax.bar(x - 2 * width, input_vector, width * 0.8,
               color="lightgray", label="Original (input)",
               alpha=0.6, hatch="//")

        # Plot rotated vectors side by side at each feature index.
        for i, (pos, vec) in enumerate(sorted(rotated_vectors.items())):
            ax.bar(x + (i - 1) * width, vec, width,
                   color=cls.COLORS[i % len(cls.COLORS)],
                   alpha=0.85, label=f"Position {pos}")

        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Feature index (pairs rotate at increasing speed →)")
        ax.set_ylabel("Vector value after RoPE rotation")
        ax.set_title("RoPE Applied to a Flat Vector at Different Positions\n"
                     "(Later feature pairs rotate faster)")
        ax.legend(loc="upper right", ncol=3, fontsize=9)
        ax.set_xticks(x[::2])
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def plot_rotation_angles(angles: dict, save_path: str =
                             "fig_rope_rotation_angle.png") -> None:
        """Plot per-feature-pair rotation angle (radians) for each position."""
        fig, ax = plt.subplots(figsize=(10, 5))
        for pos, angle_arr in sorted(angles.items()):
            ax.plot(np.arange(len(angle_arr)), angle_arr,
                    marker="o", label=f"Position {pos}")
        ax.set_xlabel("Feature pair index i (later pairs rotate faster →)")
        ax.set_ylabel("Rotation angle (radians)")
        ax.set_title("Per-Pair Rotation Angle vs. Position\n"
                     "angle_i = position × theta^(-2i/d)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()


# ============================================================================
# 4. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 3 — RoPE Inspection")
    print("=" * 70)

    cfg = RoPEInspectionConfig()
    inspector = RoPEInspector(cfg)
    rotated = inspector.inspect()
    angles = inspector.compute_rotation_angles()

    # Verify position-0 identity.
    result = inspector.verify_position_zero_identity(rotated)
    status = " PASS" if result.get("passed") else "✗ FAIL"
    print(f"\n[{status}] {result['test']}")
    print(f"    max_abs_diff = {result.get('max_abs_diff')} "
          f"(tolerance {result.get('tolerance')})")

    # Print a small numeric summary so learners can sanity-check by eye.
    print("\nNumeric summary of rotated vectors (first 8 features):")
    print(f"  {'Position':<10} " + " ".join(f"f{i:<2d}" for i in range(8)))
    for pos in sorted(rotated.keys()):
        vals = " ".join(f"{v:+.2f}" for v in rotated[pos][:8])
        print(f"  {pos:<10} {vals}")

    print("\nRotation angles (radians) per pair (first 8 pairs):")
    print(f"  {'Position':<10} " + " ".join(f"p{i:<2d}" for i in range(8)))
    for pos in sorted(angles.keys()):
        vals = " ".join(f"{a:+.3f}" for a in angles[pos][:8])
        print(f"  {pos:<10} {vals}")

    # Render visualizations.
    RoPEPlotter.plot_overlapping_bars(inspector.input_vector, rotated)
    RoPEPlotter.plot_rotation_angles(angles)

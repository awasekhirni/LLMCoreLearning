"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

ELIM5
Imagine you're taking a test and the teacher puts a piece of paper over the rest of the test so you can only read one question at a time — you cannot peek at question 5 while answering question 1. Causal masking is that piece of paper for the LLM: when the model is on token 3, it's allowed to "see" tokens 1, 2, 3 but NOT 4, 5, 6, ... This forces the model to read left-to-right, which is what makes it generative — it can only predict what comes next, never look ahead. The trick: we build a triangular "no-peeking" matrix full of very-negative numbers (-1e9) in the upper-right corner, then add it to the attention scores before softmax. Because softmax(exp(-1e9)) ≈ 0, future tokens get exactly zero attention weight — guaranteed.

input
An integer seq_len (default 8), a neg_inf_value of -1e9, and a seeded random attention-score matrix of shape (seq_len, seq_len) to demonstrate the effect on softmax.


outcome
Mask shape is exactly (seq_len, seq_len) (a square matrix).
Lower triangle (including the diagonal) is exactly 0.0 — every token may attend to itself and to all past tokens.
Strict upper triangle is exactly neg_inf_value (-1e9) — no token may attend to a future token.
After adding the mask to attention scores and applying softmax, every upper-triangular position is exactly 0.0 (within 1e-6).
After adding the mask + softmax, every row's remaining (lower-triangle) probabilities sum to 1.0.
Autoregressive property: for row i, the non-zero columns are exactly {0, 1, ..., i}.
The -1e9 substitute is numerically safe: it produces finite zeros. Using actual -inf would propagate NaN through downstream operations (e.g., a row of all -inf becomes 0/0 = NaN).


causal_masking.py
=================================
Inspect and visualize Causal Masking.

Educational objectives:
    - See the triangular structure of the causal mask.
    - Verify that the lower triangle (incl. diagonal) is 0 and the strict
      upper triangle is a very-negative number.
    - Verify that masking + softmax makes future attention weights exactly 0.
    - Verify the autoregressive property: row i may attend to columns 0..i.
    - Empirically demonstrate WHY we use -1e9 instead of -inf:
        -1e9  -> softmax -> exactly 0, finite, safe for downstream ops.
        -inf  -> can produce NaN when a whole row is -inf (e.g., position 0
                 when no past exists, but in pure causal masking position 0
                 still attends to itself so it stays finite; the danger is
                 when masking is combined with padding masks).

Design principles:
    - Single Responsibility: CausalMaskInspector computes & verifies;
      CausalMaskPlotter renders.
    - DRY: one CausalMask.build reused across all tests.
    - Reproducibility: every random op seeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. SOFTMAX (reused from 10.1; included here for self-containment)
# ============================================================================

class Softmax:
    """Numerically stable softmax along the last axis."""

    @staticmethod
    def forward(x: np.ndarray, axis: int = -1) -> np.ndarray:
        if x.size == 0:
            raise ValueError("Softmax received empty array")
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================================================
# 2. CAUSAL MASK IMPLEMENTATION
# ============================================================================

class CausalMask:
    """Upper-triangular -inf mask used in autoregressive attention.

    Why -inf (or -1e9)?
        Because softmax(exp(-inf)) = 0 exactly, future tokens get ZERO
        attention weight, guaranteeing the autoregressive property.
        Using -1e9 is a numerically safe approximation that avoids actual
        inf values that propagate badly.

    Input contract:
        seq_len: int >= 1.
        neg_inf_value: float, large negative number (default -1e9).

    Result/outcome expected:
        np.ndarray of shape (seq_len, seq_len), dtype float32.
        Lower triangle (incl. diagonal) = 0.0.
        Strict upper triangle = neg_inf_value.

    Where used in this chapter:
        Added to attention scores BEFORE softmax inside
        GroupedQueryAttention.forward.
    """

    @staticmethod
    def build(seq_len: int, neg_inf_value: float = -1e9) -> np.ndarray:
        """Stable mask using -1e9 (the recommended default)."""
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        # np.triu(k=1) gives the strict upper triangle (above diagonal).
        mask = np.triu(
            np.full((seq_len, seq_len), neg_inf_value, dtype=np.float32),
            k=1,
        )
        return mask

    @staticmethod
    def build_naive_inf(seq_len: int) -> np.ndarray:
        """DEMONSTRATION-ONLY: mask using actual -inf.

        DO NOT USE IN PRODUCTION. This exists only so learners can see
        why we substitute -1e9.
        """
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        mask = np.triu(
            np.full((seq_len, seq_len), float("-inf"), dtype=np.float64),
            k=1,
        )
        return mask


# ============================================================================
# 3. CONFIGURATION
# ============================================================================

@dataclass
class CausalMaskInspectionConfig:
    """All parameters for the causal mask inspection exercise.

    Attributes:
        seq_len:        Length of the demo sequence (mask will be seq_len x seq_len).
        neg_inf_value:  Large negative number used in the strict upper triangle.
        score_scale:    Std-dev of the random attention scores used for the demo.
        seed:           RNG seed for reproducibility.
    """
    seq_len: int = 8
    neg_inf_value: float = -1e9
    score_scale: float = 2.0
    seed: int = 42


# ============================================================================
# 4. INSPECTOR
# ============================================================================

class CausalMaskInspector:
    """Builds a causal mask and verifies its mathematical properties."""

    def __init__(self, config: CausalMaskInspectionConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # Build the mask ONCE and reuse it everywhere (DRY).
        self.mask = CausalMask.build(config.seq_len, config.neg_inf_value)
        # Generate a deterministic demo attention-score matrix.
        self.raw_scores = (self.rng.standard_normal(
            (config.seq_len, config.seq_len)
        ) * config.score_scale).astype(np.float32)

    # ----- the six mathematical guarantees -----

    def verify_shape(self) -> dict:
        """Property (1): mask is a square (seq_len, seq_len) matrix."""
        expected = (self.config.seq_len, self.config.seq_len)
        return {
            "test": "mask shape is square",
            "shape": tuple(self.mask.shape),
            "expected_shape": expected,
            "passed": bool(tuple(self.mask.shape) == expected),
        }

    def verify_lower_triangle_is_zero(self) -> dict:
        """Property (2): lower triangle (incl. diagonal) is exactly 0."""
        lower = np.tril(self.mask)  # includes diagonal
        max_abs = float(np.max(np.abs(lower)))
        return {
            "test": "lower triangle (incl. diagonal) is 0",
            "max_abs_value_in_lower_triangle": max_abs,
            "tolerance": 0.0,
            "passed": bool(max_abs == 0.0),
        }

    def verify_upper_triangle_is_neg_inf(self) -> dict:
        """Property (3): strict upper triangle equals neg_inf_value exactly."""
        strict_upper = np.triu(self.mask, k=1)
        # All strict-upper entries should equal neg_inf_value.
        all_match = bool(np.all(strict_upper == self.config.neg_inf_value))
        # Count how many entries are in the strict upper triangle.
        n_upper = int(np.sum(np.triu(np.ones_like(self.mask), k=1) != 0))
        return {
            "test": "strict upper triangle equals neg_inf_value",
            "neg_inf_value": self.config.neg_inf_value,
            "n_strict_upper_entries": n_upper,
            "all_match": all_match,
            "passed": all_match,
        }

    def verify_diagonal_is_zero(self) -> dict:
        """Property (4): diagonal is 0 (a token may attend to ITSELF)."""
        diag = np.diag(self.mask)
        max_abs = float(np.max(np.abs(diag)))
        return {
            "test": "diagonal is 0 (token attends to itself)",
            "diagonal_values": diag.tolist(),
            "max_abs_value": max_abs,
            "passed": bool(max_abs == 0.0),
        }

    def verify_softmax_zeros_future(self) -> dict:
        """Property (5): after mask + softmax, strict upper triangle is exactly 0."""
        masked = self.raw_scores + self.mask
        weights = Softmax.forward(masked, axis=-1)
        strict_upper_weights = np.triu(weights, k=1)
        max_upper = float(np.max(strict_upper_weights))
        # Also verify the lower-triangle weights still sum to 1 per row.
        lower_weights = np.tril(weights)  # incl. diagonal
        row_sums = lower_weights.sum(axis=-1)
        return {
            "test": "softmax(masked scores) zeros out future positions",
            "max_future_attention_weight": max_upper,
            "tolerance": 1e-6,
            "row_sums_lower_triangle": row_sums.tolist(),
            "all_row_sums_are_one": bool(np.allclose(row_sums, 1.0, atol=1e-6)),
            "passed": bool(max_upper < 1e-6 and
                            np.allclose(row_sums, 1.0, atol=1e-6)),
        }

    def verify_autoregressive_property(self) -> dict:
        """Property (6): row i has non-zero columns exactly {0, 1, ..., i}.

        This is the formal statement of "left-to-right reading":
        position i may attend to itself and any earlier position, but no later.
        """
        masked = self.raw_scores + self.mask
        weights = Softmax.forward(masked, axis=-1)
        # For each row i, find the indices with non-zero weight.
        allowed_per_row = []
        passed = True
        for i in range(self.config.seq_len):
            nonzero_cols = np.where(weights[i] > 1e-6)[0].tolist()
            expected = list(range(i + 1))
            allowed_per_row.append(nonzero_cols)
            if nonzero_cols != expected:
                passed = False
        return {
            "test": "autoregressive property (row i attends to cols 0..i)",
            "allowed_columns_per_row": allowed_per_row,
            "passed": passed,
        }

    def verify_numerical_safety_vs_inf(self) -> dict:
        """Property (7): -1e9 produces finite 0s; -inf can produce NaN.

        Demonstrates why we use -1e9 instead of actual -inf.
        """
        # Stable path with -1e9.
        stable_mask = self.mask
        stable_weights = Softmax.forward(self.raw_scores + stable_mask, axis=-1)
        stable_all_finite = bool(np.all(np.isfinite(stable_weights)))
        # Naive path with -inf.
        inf_mask = CausalMask.build_naive_inf(self.config.seq_len)
        # Use float64 for the -inf path so we can actually observe NaN
        # (float32 -inf + finite may behave differently across platforms).
        inf_weights = Softmax.forward(
            self.raw_scores.astype(np.float64) + inf_mask, axis=-1
        )
        inf_all_finite = bool(np.all(np.isfinite(inf_weights)))
        return {
            "test": "numerical safety (-1e9 vs. -inf)",
            "stable_neg1e9_all_finite": stable_all_finite,
            "naive_neg_inf_all_finite": inf_all_finite,
            "passed": bool(stable_all_finite and not inf_all_finite
                            or (stable_all_finite and inf_all_finite)),
            "note": ("With pure causal masking (no padding), -inf is technically "
                     "still safe because each row has at least one non-masked entry. "
                     "The danger arises when -inf combines with padding masks that "
                     "zero out entire rows. -1e9 is the universally safe choice."),
        }

    def run_all_verifications(self) -> dict:
        """Run all six property checks against the standard inputs."""
        return {
            "mask": self.mask,
            "raw_scores": self.raw_scores,
            "tests": [
                self.verify_shape(),
                self.verify_lower_triangle_is_zero(),
                self.verify_upper_triangle_is_neg_inf(),
                self.verify_diagonal_is_zero(),
                self.verify_softmax_zeros_future(),
                self.verify_autoregressive_property(),
                self.verify_numerical_safety_vs_inf(),
            ],
        }


# ============================================================================
# 5. PLOTTER
# ============================================================================

class CausalMaskPlotter:
    """Renders the inspector's results into three figures."""

    @staticmethod
    def plot_mask_matrix(mask: np.ndarray,
                          save_path: str = "fig_causal_mask_matrix.png") -> None:
        """Heatmap of the mask itself, showing the triangular structure."""
        fig, ax = plt.subplots(figsize=(7, 6))
        # Use a diverging colormap because the mask has both 0 and large-negative.
        im = ax.imshow(mask, cmap="RdBu", vmin=-1e9, vmax=0)
        ax.set_title("Causal Mask Matrix\n"
                     "Blue (0) = may attend   |   Red (-1e9) = blocked (future)")
        ax.set_xlabel("Key position (attends TO)")
        ax.set_ylabel("Query position (attends FROM)")
        # Annotate each cell with its value (0 or -1e9).
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                val = mask[i, j]
                label = "0" if val == 0 else "-1e9"
                color = "white" if val < -1 else "black"
                ax.text(j, i, label, ha="center", va="center",
                        color=color, fontsize=8, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Mask value")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def plot_effect_on_softmax(raw_scores: np.ndarray, mask: np.ndarray,
                                save_path: str =
                                "fig_causal_mask_effect_on_softmax.png") -> None:
        """Three-panel: raw scores -> masked scores -> softmax weights."""
        masked = raw_scores + mask
        weights = Softmax.forward(masked, axis=-1)
        seq_len = raw_scores.shape[0]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        panels = [
            (axes[0], raw_scores, "Raw attention scores",
             "viridis", None),
            (axes[1], masked, "Masked scores (raw + causal mask)",
             "viridis", mask),
            (axes[2], weights,
             "Softmax weights (future = 0 exactly)",
             "magma", None),
        ]
        for ax, data, title, cmap, overlay in panels:
            vmin = float(data.min()) if title.startswith("Raw") else None
            vmax = float(data.max()) if title.startswith("Raw") else None
            if "Softmax" in title:
                vmin, vmax = 0.0, 1.0
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Causal Masking Effect on Softmax",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def plot_allowed_attentions(mask: np.ndarray,
                                  save_path: str =
                                  "fig_causal_mask_allowed_attentions.png") -> None:
        """Bar chart: how many tokens each query position may attend to."""
        seq_len = mask.shape[0]
        # Position i may attend to i+1 tokens (positions 0..i).
        allowed_counts = np.arange(1, seq_len + 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(np.arange(seq_len), allowed_counts,
                       color="steelblue", edgecolor="black", linewidth=0.5)
        # Annotate each bar with its count.
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{allowed_counts[i]}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_title("Autoregressive Property: How Many Tokens Each Position May Attend To\n"
                     "Position 0 sees 1 (itself).  Position N-1 sees all N.")
        ax.set_xlabel("Query position (attends FROM)")
        ax.set_ylabel("Number of allowed key positions")
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(1, seq_len + 1))
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()


# ============================================================================
# 6. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Reference Card 10.3 — Causal Masking Inspection")
    print("=" * 70)

    cfg = CausalMaskInspectionConfig()
    inspector = CausalMaskInspector(cfg)
    results = inspector.run_all_verifications()

    # Print the mask matrix so learners can sanity-check by eye.
    print(f"\nCausal mask (shape {results['mask'].shape}, "
          f"neg_inf_value = {cfg.neg_inf_value}):")
    print(results['mask'])

    # Print a few rows of the raw attention scores.
    print(f"\nRaw attention scores (shape {results['raw_scores'].shape}):")
    print(results['raw_scores'].round(2))

    # Run & pretty-print every property test.
    print("\n" + "-" * 70)
    print("Property verification tests")
    print("-" * 70)
    for t in results["tests"]:
        status = "✓ PASS" if t.get("passed") else "✗ FAIL"
        print(f"\n[{status}] {t['test']}")
        for k, v in t.items():
            if k in ("test", "passed"):
                continue
            # Truncate long nested lists for readability.
            if isinstance(v, list) and len(v) > 6:
                print(f"    {k}: [{v[0]}, {v[1]}, {v[2]}, ... , {v[-1]}] "
                      f"({len(v)} entries)")
            else:
                print(f"    {k}: {v}")

    # Render visualizations.
    print("\n" + "-" * 70)
    print("Generating visualizations...")
    print("-" * 70)
    CausalMaskPlotter.plot_mask_matrix(results["mask"])
    CausalMaskPlotter.plot_effect_on_softmax(
        results["raw_scores"], results["mask"]
    )
    CausalMaskPlotter.plot_allowed_attentions(results["mask"])

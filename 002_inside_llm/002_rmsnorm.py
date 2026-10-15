"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

ELIM5
Suppose your friend tells you a number,but they always shout.
To compare their numbers fairly, you divide each one by how loud they were.
RMSNorm does this for vectors: it shrinks every vector so its "average loudness" (root mean square, or RMS) becomes 1,
then lets the model re-scale each feature with a learnable weight. The trick: it skips the mean-subtraction step that LayerNorm does,
which saves compute , and at 32+ layers deep, that adds up. Even better, RMSNorm has a magical property called scale invariance:
 if you whisper the same message 3× louder, RMSNorm gives you back the exact same normalized answer, only the sign of the message can flip the output, never the loudness.


input
A 2-D NumPy array of shape (batch, vector_dim). Default: batch=4, vector_dim=16, eps=1e-6, seed=42. Stress-tested with zero inputs (numerical stability) and scaled inputs (scale invariance).

outcome
Output shape matches input shape exactly — RMSNorm preserves dimensions.
Before the learnable weight is applied, every row's RMS equals 1.0 (within 1e-5).
RMSNorm does NOT subtract the mean: the output's mean is generally non-zero (unlike LayerNorm where mean ≈ 0).
The learnable weight independently scales each feature: output[i, j] = normalized[i, j] * weight[j].
Scale invariance: for any positive scalar c, f(c * x) == f(x) exactly (within 1e-5).
Numerical stability: for an all-zeros input, the eps term prevents division by zero — output is finite (zeros).



rmsnorm.py
==========================
Inspect and visualize RMSNorm (Root Mean Square Normalization).

Educational objectives:
    - See, step by step, how a vector is magnitude-normalized by RMSNorm.
    - Verify the mathematical properties of RMSNorm:
        (1) shape preservation
        (2) RMS of normalized output = 1 (before weight)
        (3) NO mean subtraction (unlike LayerNorm)
        (4) weight independently scales each feature
        (5) scale invariance: f(c * x) = f(x) for c > 0
        (6) numerical stability: eps prevents division by zero
    - Empirically compare RMSNorm vs. LayerNorm to see WHY RMSNorm is
      cheaper: it skips the mean-subtraction reduction entirely.

Design principles:
    - Single Responsibility: RMSNormInspector computes & verifies;
      RMSNormPlotter renders.
    - DRY: one RMSNorm implementation reused across all tests.
    - Reproducibility: every random op seeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. RMSNORM IMPLEMENTATION
# ============================================================================

class RMSNorm:
    """Root Mean Square Normalization (Llama-style, no mean subtraction).

    Why not LayerNorm?
        RMSNorm skips the mean-centering step (one full tensor reduction
        along the feature axis). Empirically matches LayerNorm quality at
        lower compute -- important when you stack 32+ transformer blocks.

    Input contract:
        x: np.ndarray, last dim == self.weight.shape[0].
        Returns: np.ndarray of same shape, RMS-normalized then scaled.

    Result/outcome expected:
        Same-shape array. For each row, BEFORE applying self.weight:
            sqrt(mean(x_normalized^2)) ≈ 1.

    Mathematical properties:
        (1) Shape-preserving: out.shape == x.shape.
        (2) Scale-invariant for c > 0: f(c * x) == f(x).
        (3) No mean subtraction: mean(out) need not be 0.

    Where used in this chapter:
        As the pre-norm before BOTH the attention and FFN sub-layers
        inside TransformerBlock.
    """

    def __init__(self, dim: int, eps: float = 1e-6,
                 rng: Optional[np.random.Generator] = None):
        # Learnable scale; initialized to ones (PyTorch convention).
        self.weight = np.ones(dim, dtype=np.float32)
        self.eps = eps
        self.dim = dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply RMSNorm: out = (x / RMS(x)) * weight."""
        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"RMSNorm expected last dim {self.weight.shape[0]}, "
                f"got {x.shape[-1]}"
            )
        # Compute mean square along feature axis, then take reciprocal sqrt.
        ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
        norm = 1.0 / np.sqrt(ms + self.eps)
        return (x * norm) * self.weight

    def forward_without_weight(self, x: np.ndarray) -> np.ndarray:
        """Apply RMSNorm WITHOUT the learnable weight -- for property checks.

        Returns just (x / RMS(x)) so we can verify the RMS becomes 1.
        """
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"RMSNorm expected last dim {self.dim}, got {x.shape[-1]}"
            )
        ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
        norm = 1.0 / np.sqrt(ms + self.eps)
        return x * norm


# ============================================================================
# 2. LAYERNORM (reference implementation for comparison only)
# ============================================================================

class LayerNorm:
    """Reference LayerNorm implementation -- for COMPARISON ONLY.

    LayerNorm subtracts the mean AND divides by the standard deviation,
    then applies a learnable weight + bias. RMSNorm skips the mean
    subtraction entirely, which is what makes it cheaper.

    Input contract & result/outcome expected: same shape as input;
    mean(output) ≈ 0 and std(output) ≈ 1 (before weight & bias).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = np.ones(dim, dtype=np.float32)
        self.bias = np.zeros(dim, dtype=np.float32)
        self.eps = eps
        self.dim = dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"LayerNorm expected last dim {self.dim}, got {x.shape[-1]}"
            )
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + self.eps)
        return normed * self.weight + self.bias

    def forward_without_weight(self, x: np.ndarray) -> np.ndarray:
        """Apply LayerNorm WITHOUT weight & bias -- for fair comparison."""
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"LayerNorm expected last dim {self.dim}, got {x.shape[-1]}"
            )
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps)


# ============================================================================
# 3. CONFIGURATION
# ============================================================================

@dataclass
class RMSNormInspectionConfig:
    """All parameters for the RMSNorm inspection exercise.

    Attributes:
        vector_dim:    Dimensionality of each input vector (must be even).
        batch_size:    Number of samples in the demo batch.
        eps:           Epsilon for numerical stability (default 1e-6).
        scale_test_c:  Positive scalar for the scale-invariance test.
        zero_test:     Whether to also run the all-zeros stability test.
        seed:          RNG seed for reproducibility.
    """
    vector_dim: int = 16
    batch_size: int = 4
    eps: float = 1e-6
    scale_test_c: float = 3.0
    zero_test: bool = True
    seed: int = 42


# ============================================================================
# 4. INSPECTOR
# ============================================================================

class RMSNormInspector:
    """Applies RMSNorm to inputs and verifies its mathematical properties."""

    def __init__(self, config: RMSNormInspectionConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # Build the RMSNorm layer ONCE and reuse it everywhere (DRY).
        self.rmsnorm = RMSNorm(config.vector_dim, eps=config.eps)
        # Also build a LayerNorm for comparison.
        self.layernorm = LayerNorm(config.vector_dim, eps=config.eps)
        # Generate a deterministic demo batch with NONZERO mean so we can
        # visibly see that RMSNorm does NOT center.
        self.demo_batch = (self.rng.standard_normal(
            (config.batch_size, config.vector_dim)
        ) * 2.0 + 5.0).astype(np.float32)  # +5 to create a clear nonzero mean
        # A single sample for the scale-invariance test.
        self.sample = (self.rng.standard_normal(config.vector_dim)
                       * 2.0).astype(np.float32)

    # ----- the seven mathematical guarantees -----

    def verify_shape_preservation(self) -> dict:
        """Property (1): output shape == input shape."""
        out = self.rmsnorm.forward(self.demo_batch)
        return {
            "test": "shape preservation",
            "input_shape": tuple(self.demo_batch.shape),
            "output_shape": tuple(out.shape),
            "passed": bool(tuple(out.shape) == tuple(self.demo_batch.shape)),
        }

    def verify_rms_equals_one_before_weight(self) -> dict:
        """Property (2): before weight, each row's RMS is approximately 1."""
        normed = self.rmsnorm.forward_without_weight(self.demo_batch)
        # RMS per row.
        rms_per_row = np.sqrt(np.mean(normed ** 2, axis=-1))
        max_dev = float(np.max(np.abs(rms_per_row - 1.0)))
        return {
            "test": "RMS of normalized output = 1 (before weight)",
            "rms_per_row": rms_per_row.tolist(),
            "max_abs_deviation_from_1": max_dev,
            "tolerance": 1e-3,  # float32 precision + eps slack
            "passed": bool(max_dev < 1e-3),
        }

    def verify_no_mean_subtraction(self) -> dict:
        """Property (3): RMSNorm does NOT subtract the mean.

        Contrast with LayerNorm, which DOES center the mean to ~0.
        """
        rms_out = self.rmsnorm.forward_without_weight(self.demo_batch)
        ln_out = self.layernorm.forward_without_weight(self.demo_batch)
        # Per-row mean of the raw input (should be far from 0 since we added 5).
        input_mean_per_row = np.mean(self.demo_batch, axis=-1)
        # Per-row mean of RMSNorm output: should be CLOSE to input_mean / RMS.
        rms_mean_per_row = np.mean(rms_out, axis=-1)
        # Per-row mean of LayerNorm output: should be ~0.
        ln_mean_per_row = np.mean(ln_out, axis=-1)
        max_abs_ln_mean = float(np.max(np.abs(ln_mean_per_row)))
        return {
            "test": "no mean subtraction (vs. LayerNorm)",
            "input_mean_per_row": input_mean_per_row.tolist(),
            "rmsnorm_output_mean_per_row": rms_mean_per_row.tolist(),
            "layernorm_output_mean_per_row": ln_mean_per_row.tolist(),
            "max_abs_layernorm_mean": max_abs_ln_mean,
            "layernorm_centers_mean_to_zero": bool(max_abs_ln_mean < 1e-3),
            "passed": bool(max_abs_ln_mean < 1e-3),  # LN centers; RMS doesn't
        }

    def verify_weight_scaling(self) -> dict:
        """Property (4): weight independently scales each feature."""
        # Temporarily set weight to a known pattern (e.g., 1, 2, 3, ...).
        original_weight = self.rmsnorm.weight.copy()
        try:
            test_weight = np.arange(1, self.config.vector_dim + 1,
                                     dtype=np.float32)
            self.rmsnorm.weight = test_weight
            out = self.rmsnorm.forward(self.sample)
            # Compute what the output SHOULD be: (sample / RMS(sample)) * weight.
            normed = self.rmsnorm.forward_without_weight(self.sample)
            expected = normed * test_weight
            diff = float(np.max(np.abs(out - expected)))
        finally:
            self.rmsnorm.weight = original_weight
        return {
            "test": "weight independently scales each feature",
            "max_abs_diff_vs_expected": diff,
            "tolerance": 1e-5,
            "passed": bool(diff < 1e-5),
        }

    def verify_scale_invariance(self) -> dict:
        """Property (5): f(c * x) == f(x) for any positive scalar c.

        This is the defining property of RMS-based normalization:
            RMS(c * x) = c * RMS(x)   for c > 0
            =>  f(c * x) = (c * x) / (c * RMS(x)) * w = f(x)
        """
        c = self.config.scale_test_c
        original_weight = self.rmsnorm.weight.copy()
        try:
            # Use weight = ones so we isolate the normalization behavior.
            self.rmsnorm.weight = np.ones(self.config.vector_dim, dtype=np.float32)
            f_x = self.rmsnorm.forward(self.sample)
            f_cx = self.rmsnorm.forward(c * self.sample)
            diff = float(np.max(np.abs(f_cx - f_x)))
        finally:
            self.rmsnorm.weight = original_weight
        return {
            "test": f"scale invariance (f({c} * x) = f(x))",
            "max_abs_diff": diff,
            "tolerance": 1e-4,  # float32 precision
            "passed": bool(diff < 1e-4),
        }

    def verify_negative_scale_flips_sign(self) -> dict:
        """Property (5b): f(-x) = -f(x)  (sign flips, magnitude preserved)."""
        original_weight = self.rmsnorm.weight.copy()
        try:
            self.rmsnorm.weight = np.ones(self.config.vector_dim, dtype=np.float32)
            f_x = self.rmsnorm.forward(self.sample)
            f_neg_x = self.rmsnorm.forward(-self.sample)
            diff = float(np.max(np.abs(f_neg_x + f_x)))  # f(-x) - (-f(x)) = 0
        finally:
            self.rmsnorm.weight = original_weight
        return {
            "test": "negative scaling flips sign (f(-x) = -f(x))",
            "max_abs_diff_vs_neg_f_x": diff,
            "tolerance": 1e-4,
            "passed": bool(diff < 1e-4),
        }

    def verify_numerical_stability_on_zeros(self) -> dict:
        """Property (6): eps prevents division by zero on an all-zeros input."""
        zeros = np.zeros((1, self.config.vector_dim), dtype=np.float32)
        out = self.rmsnorm.forward(zeros)
        all_finite = bool(np.all(np.isfinite(out)))
        # With weight=ones, output should be ~0 (since input is 0).
        max_abs = float(np.max(np.abs(out)))
        return {
            "test": "numerical stability on all-zeros input",
            "all_outputs_finite": all_finite,
            "max_abs_output": max_abs,
            "expected_max_abs": 0.0,  # 0 / sqrt(eps) * weight(=1) = 0
            "tolerance": 1e-3,
            "passed": bool(all_finite and max_abs < 1e-3),
        }

    def run_all_verifications(self) -> dict:
        """Run all seven property checks against the standard inputs."""
        out = self.rmsnorm.forward(self.demo_batch)
        return {
            "demo_batch": self.demo_batch,
            "demo_output": out,
            "sample": self.sample,
            "tests": [
                self.verify_shape_preservation(),
                self.verify_rms_equals_one_before_weight(),
                self.verify_no_mean_subtraction(),
                self.verify_weight_scaling(),
                self.verify_scale_invariance(),
                self.verify_negative_scale_flips_sign(),
                self.verify_numerical_stability_on_zeros(),
            ],
        }


# ============================================================================
# 5. PLOTTER
# ============================================================================

class RMSNormPlotter:
    """Renders the inspector's results into three figures."""

    @staticmethod
    def plot_io_transformation(demo_batch: np.ndarray, demo_output: np.ndarray,
                                save_path: str =
                                "fig_rmsnorm_io_transformation.png") -> None:
        """Two-panel plot: input vector -> output vector for one sample."""
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
        sample_idx = 0
        x = np.arange(demo_batch.shape[1])

        axes[0].bar(x, demo_batch[sample_idx], color="steelblue",
                    edgecolor="black", linewidth=0.5)
        axes[0].axhline(0, color="black", linewidth=0.7)
        axes[0].axhline(np.mean(demo_batch[sample_idx]), color="red",
                        linestyle="--", linewidth=1.5,
                        label=f"mean = {np.mean(demo_batch[sample_idx]):.2f}")
        axes[0].set_title(f"RMSNorm — Input vector (sample {sample_idx})\n"
                          f"RMS = {np.sqrt(np.mean(demo_batch[sample_idx]**2)):.2f}, "
                          f"mean = {np.mean(demo_batch[sample_idx]):.2f} (NOT zero!)")
        axes[0].set_xlabel("Feature index")
        axes[0].set_ylabel("Input value")
        axes[0].legend(loc="upper right")
        axes[0].grid(alpha=0.3, axis="y")

        axes[1].bar(x, demo_output[sample_idx], color="seagreen",
                    edgecolor="black", linewidth=0.5)
        axes[1].axhline(0, color="black", linewidth=0.7)
        axes[1].axhline(np.mean(demo_output[sample_idx]), color="red",
                        linestyle="--", linewidth=1.5,
                        label=f"mean = {np.mean(demo_output[sample_idx]):.2f}")
        axes[1].set_title(f"RMSNorm — Output vector (after x / RMS(x) * weight)\n"
                          f"RMS ≈ 1.0 (magnitude normalized), "
                          f"mean = {np.mean(demo_output[sample_idx]):.2f} (still NOT zero!)")
        axes[1].set_xlabel("Feature index")
        axes[1].set_ylabel("Output value")
        axes[1].legend(loc="upper right")
        axes[1].grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def plot_rmsnorm_vs_layernorm(demo_batch: np.ndarray,
                                    rmsnorm: RMSNorm,
                                    layernorm: LayerNorm,
                                    save_path: str =
                                    "fig_rmsnorm_vs_layernorm.png") -> None:
        """Three-panel: raw input, RMSNorm output (mean != 0), LayerNorm output (mean ~ 0)."""
        sample_idx = 0
        sample = demo_batch[sample_idx]
        rms_out = rmsnorm.forward_without_weight(sample)
        ln_out = layernorm.forward_without_weight(sample)
        x = np.arange(len(sample))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        panels = [
            (axes[0], sample, "Raw input\n(mean ≠ 0)", "steelblue"),
            (axes[1], rms_out,
             f"RMSNorm output\n(mean = {np.mean(rms_out):.3f}, still ≠ 0)",
             "seagreen"),
            (axes[2], ln_out,
             f"LayerNorm output\n(mean = {np.mean(ln_out):.3f} ≈ 0, centered!)",
             "darkorange"),
        ]
        for ax, vec, title, color in panels:
            ax.bar(x, vec, color=color, edgecolor="black", linewidth=0.5)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.axhline(np.mean(vec), color="red", linestyle="--", linewidth=1.5,
                       label=f"mean = {np.mean(vec):.3f}")
            ax.set_title(title)
            ax.set_xlabel("Feature index")
            ax.set_ylabel("Value")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle("RMSNorm vs. LayerNorm: LayerNorm centers the mean; RMSNorm does NOT",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def plot_scale_invariance(sample: np.ndarray, rmsnorm: RMSNorm,
                                save_path: str =
                                "fig_rmsnorm_scale_invariance.png") -> None:
        """Three overlapping bar charts: f(x), f(2x), f(3x) should be identical."""
        original_weight = rmsnorm.weight.copy()
        try:
            rmsnorm.weight = np.ones(rmsnorm.dim, dtype=np.float32)
            f_x = rmsnorm.forward(sample)
            f_2x = rmsnorm.forward(2.0 * sample)
            f_3x = rmsnorm.forward(3.0 * sample)
            diff_2x = float(np.max(np.abs(f_2x - f_x)))
            diff_3x = float(np.max(np.abs(f_3x - f_x)))
        finally:
            rmsnorm.weight = original_weight

        x = np.arange(len(sample))
        width = 0.27
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width, f_x, width, color="steelblue",
               edgecolor="black", linewidth=0.5,
               label=f"f(x)  (RMS={np.sqrt(np.mean(sample**2)):.2f})")
        ax.bar(x, f_2x, width, color="seagreen",
               edgecolor="black", linewidth=0.5,
               label=f"f(2x)  (RMS={np.sqrt(np.mean((2*sample)**2)):.2f})")
        ax.bar(x + width, f_3x, width, color="darkorange",
               edgecolor="black", linewidth=0.5,
               label=f"f(3x)  (RMS={np.sqrt(np.mean((3*sample)**2)):.2f})")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(
            f"RMSNorm Scale Invariance: f(x) = f(2x) = f(3x)\n"
            f"max|f(2x) - f(x)| = {diff_2x:.2e},  max|f(3x) - f(x)| = {diff_3x:.2e}"
        )
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Normalized output value")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3, axis="y")
        # Annotate the invariance verdict.
        if diff_2x < 1e-4 and diff_3x < 1e-4:
            ax.text(0.5, 0.95, "INVARIANT ✓",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=14, color="green", fontweight="bold")
        else:
            ax.text(0.5, 0.95, "NOT INVARIANT ✗",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=14, color="red", fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()


# ============================================================================
# 6. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Reference Card 10.4 — RMSNorm Inspection")
    print("=" * 70)

    cfg = RMSNormInspectionConfig()
    inspector = RMSNormInspector(cfg)
    results = inspector.run_all_verifications()

    # Print the input/output for the first sample.
    print(f"\nDemo batch shape: {results['demo_batch'].shape}")
    print(f"Demo output shape: {results['demo_output'].shape}")
    print(f"\nSample 0 input (first 8 of {cfg.vector_dim} features):")
    print(f"  {results['demo_batch'][0][:8].round(3)} ...")
    print(f"  mean = {np.mean(results['demo_batch'][0]):.3f}, "
          f"RMS = {np.sqrt(np.mean(results['demo_batch'][0]**2)):.3f}")
    print(f"\nSample 0 output (first 8 of {cfg.vector_dim} features):")
    print(f"  {results['demo_output'][0][:8].round(3)} ...")
    print(f"  mean = {np.mean(results['demo_output'][0]):.3f} (still non-zero!), "
          f"RMS = {np.sqrt(np.mean(results['demo_output'][0]**2)):.3f}")

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
            # Truncate long lists for readability.
            if isinstance(v, list) and len(v) > 4:
                print(f"    {k}: [{v[0]:.4f}, {v[1]:.4f}, ... , "
                      f"{v[-1]:.4f}] ({len(v)} entries)")
            else:
                print(f"    {k}: {v}")

    # Render visualizations.
    print("\n" + "-" * 70)
    print("Generating visualizations...")
    print("-" * 70)
    RMSNormPlotter.plot_io_transformation(
        results["demo_batch"], results["demo_output"]
    )
    RMSNormPlotter.plot_rmsnorm_vs_layernorm(
        results["demo_batch"], inspector.rmsnorm, inspector.layernorm
    )
    RMSNormPlotter.plot_scale_invariance(
        results["sample"], inspector.rmsnorm
    )

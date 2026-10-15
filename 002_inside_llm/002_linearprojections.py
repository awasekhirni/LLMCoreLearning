"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

ELIM5
A linear projection is like a translator at a conference: someone walks up to the translator speaking one language (a vector of size in), and the translator outputs the same idea in a different language (a vector of size out). The translator's "dictionary" is just a big table of numbers (W), and the act of translating is matrix multiplication. Optionally the translator can also add a personal bias (+ b). The two magic rules every such translator obeys are: (1) translating two sentences separately and adding the results equals translating their sum — f(a + b) = f(a) + f(b) (called linearity), and (2) translating a louder version of one sentence equals the louder version of the translation — f(c * a) = c * f(a) (called homogeneity). These two rules are why we call it "linear" — and they are why gradients can flow backward through the whole LLM during training.


input
A 2-D NumPy array of shape (batch, in_features). Default: batch=4, in_features=8, out_features=5, use_bias=True, generated with seed=42. Also stress-tested with zero inputs (to probe bias) and summed inputs (to probe linearity).

outcome:

     Output shape is (batch, out_features) — dimensionality changes from in_features to out_features.
     Linearity: f(a + b) == f(a) + f(b) to within 1e-5 tolerance.
     Homogeneity: f(c * a) == c * f(a) to within 1e-5 tolerance.
     Bias behavior: with bias, f(zeros) == bias broadcast across batch; without bias, f(zeros) == zeros.
     Transpose convention: y = x @ W.T (with W of shape (out, in)) gives the correct output; the non-transposed version x @ W produces a different (and wrong, unless W is square) result.

linear_projection.py
====================================
Inspect and visualize Linear Projections (matrix multiplication).

Educational objectives:
    - See, step by step, how a vector changes dimensionality under y = x @ W.T + b.
    - Verify the two defining properties of LINEARITY:
        (1) f(a + b) = f(a) + f(b)            (additivity)
        (2) f(c * a) = c * f(a)               (homogeneity)
    - Verify the shape transformation: (batch, in_features) -> (batch, out_features).
    - Verify the bias behavior: f(0) = bias (when bias is used).
    - Empirically demonstrate WHY we transpose W:
        y = x @ W.T  with W of shape (out, in)  ->  correct (matches PyTorch).
        y = x @ W    without transposing        ->  wrong (unless W is square).

Design principles:
    - Single Responsibility: LinearProjectionInspector computes & verifies;
      LinearProjectionPlotter renders.
    - DRY: one LinearProjection implementation reused across all tests.
    - Reproducibility: every random op seeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. LINEAR PROJECTION IMPLEMENTATION
# ============================================================================

class LinearProjection:
    """y = x @ W.T + b  (the matrix-multiply primitive).

    Why W.T?
        PyTorch stores Linear.weight with shape (out_features, in_features).
        To compute y = x @ W^T cleanly as a batched matmul, we transpose
        inside forward. This matches PyTorch semantics so learners can
        switch mentally between NumPy and PyTorch code.

    Input contract:
        x: np.ndarray of shape (batch, in_features) or (in_features,).
        Returns: np.ndarray of shape (batch, out_features).

    Result/outcome expected:
        Same-batch output, last dim changed to out_features.
        Linear in x: f(a + b) = f(a) + f(b),  f(c * a) = c * f(a).

    Where used in this chapter:
        - Q/K/V/O projections in GroupedQueryAttention.
        - All three projections (w1, w2, w3) in SwiGLU.
        - The LM head's hidden_dim -> vocab_size projection.
    """

    def __init__(self, in_features: int, out_features: int,
                 use_bias: bool, rng: np.random.Generator):
        # Kaiming-style init scaled for toy sizes.
        std = np.sqrt(2.0 / (in_features + out_features))
        # Weight: (out_features, in_features) -- PyTorch convention.
        self.weight = (rng.standard_normal((out_features, in_features)) * std
                       ).astype(np.float32)
        self.bias = (rng.standard_normal(out_features) * std
                     ).astype(np.float32) if use_bias else None
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Correct projection: y = x @ W.T (+ b)."""
        if x.ndim < 2:
            # Promote 1-D vector to (1, d) so matmul works uniformly.
            x = x[None, :]
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward_naive_no_transpose(self, x: np.ndarray) -> np.ndarray:
        """DEMONSTRATION-ONLY: y = x @ W (no transpose, no bias).

        Used to show WHY the transpose convention matters.
        Will raise a shape error unless W is square (in == out).
        """
        if x.ndim < 2:
            x = x[None, :]
        # x: (batch, in), W: (out, in) -> x @ W fails because shapes don't align.
        # We deliberately do NOT transpose to trigger the failure visibly.
        return x @ self.weight  # may raise ValueError on non-square W


# ============================================================================
# 2. CONFIGURATION
# ============================================================================

@dataclass
class LinearProjectionInspectionConfig:
    """All parameters for the linear projection inspection exercise.

    Attributes:
        in_features:   Dimensionality of the input vectors.
        out_features:  Dimensionality of the output vectors (DIFFERENT from
                       in_features so the dimension change is visible).
        batch_size:    Number of samples in the demo batch.
        use_bias:      Whether to add a learnable bias term.
        scale:         Std-dev of the random input & weight init.
        homogeneity_c: Scalar multiplier for the homogeneity test.
        seed:          RNG seed for reproducibility.
    """
    in_features: int = 8
    out_features: int = 5
    batch_size: int = 4
    use_bias: bool = True
    scale: float = 1.0
    homogeneity_c: float = 2.5
    seed: int = 42


# ============================================================================
# 3. INSPECTOR
# ============================================================================

class LinearProjectionInspector:
    """Applies a linear projection to inputs and verifies its mathematical properties."""

    def __init__(self, config: LinearProjectionInspectionConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # Build the projection layer ONCE and reuse it everywhere (DRY).
        self.layer = LinearProjection(
            in_features=config.in_features,
            out_features=config.out_features,
            use_bias=config.use_bias,
            rng=self.rng,
        )
        # Generate a deterministic demo batch.
        self.demo_batch = (self.rng.standard_normal(
            (config.batch_size, config.in_features)
        ) * config.scale).astype(np.float32)
        # Two single samples for the linearity test.
        self.sample_a = (self.rng.standard_normal(config.in_features)
                         * config.scale).astype(np.float32)
        self.sample_b = (self.rng.standard_normal(config.in_features)
                         * config.scale).astype(np.float32)

    # ----- the five mathematical guarantees -----

    def verify_shape_transformation(self) -> dict:
        """Property (1): output shape is (batch, out_features)."""
        out = self.layer.forward(self.demo_batch)
        expected = (self.config.batch_size, self.config.out_features)
        return {
            "test": "shape transformation",
            "input_shape": tuple(self.demo_batch.shape),
            "output_shape": tuple(out.shape),
            "expected_output_shape": expected,
            "passed": bool(tuple(out.shape) == expected),
        }

    def verify_additivity(self) -> dict:
        """Property (2): f(a + b) == f(a) + f(b)."""
        fa = self.layer.forward(self.sample_a)
        fb = self.layer.forward(self.sample_b)
        f_ab = self.layer.forward(self.sample_a + self.sample_b)
        diff = float(np.max(np.abs(f_ab - (fa + fb))))
        return {
            "test": "additivity (f(a+b) = f(a) + f(b))",
            "max_abs_diff": diff,
            "tolerance": 1e-5,
            "passed": bool(diff < 1e-5),
        }

    def verify_homogeneity(self) -> dict:
        """Property (3): f(c * a) == c * f(a)."""
        c = self.config.homogeneity_c
        fa = self.layer.forward(self.sample_a)
        f_ca = self.layer.forward(c * self.sample_a)
        diff = float(np.max(np.abs(f_ca - c * fa)))
        return {
            "test": f"homogeneity (f({c} * a) = {c} * f(a))",
            "max_abs_diff": diff,
            "tolerance": 1e-5,
            "passed": bool(diff < 1e-5),
        }

    def verify_bias_behavior(self) -> dict:
        """Property (4): f(zeros) == bias (with bias) or zeros (without bias)."""
        zeros = np.zeros((1, self.config.in_features), dtype=np.float32)
        out = self.layer.forward(zeros)
        if self.config.use_bias:
            expected = self.layer.bias
            label = "f(0) = bias (use_bias=True)"
        else:
            expected = np.zeros(self.config.out_features, dtype=np.float32)
            label = "f(0) = zeros (use_bias=False)"
        diff = float(np.max(np.abs(out[0] - expected)))
        return {
            "test": "bias behavior on zero input",
            "label": label,
            "f_zeros": out[0].tolist(),
            "expected": (expected.tolist() if isinstance(expected, np.ndarray)
                         else expected),
            "max_abs_diff": diff,
            "tolerance": 1e-6,
            "passed": bool(diff < 1e-6),
        }

    def verify_transpose_convention(self) -> dict:
        """Property (5): y = x @ W.T (with W shape (out, in)) is the correct form.

        Demonstrates that the non-transposed version either errors (non-square W)
        or gives different (wrong) results.
        """
        result = {
            "test": "transpose convention (y = x @ W.T)",
            "weight_shape": tuple(self.layer.weight.shape),
            "input_last_dim": self.config.in_features,
            "output_last_dim": self.config.out_features,
        }
        # Correct path: always works.
        correct = self.layer.forward(self.sample_a)
        result["correct_output_shape"] = tuple(correct.shape)
        # Naive path: try without transpose.
        try:
            wrong = self.layer.forward_naive_no_transpose(self.sample_a)
            # If we got here, W must be square; check the values differ
            # unless by absurd coincidence.
            diff = float(np.max(np.abs(wrong - correct)))
            result["naive_output_shape"] = tuple(wrong.shape)
            result["naive_max_abs_diff_from_correct"] = diff
            result["naive_status"] = ("ran but produced different values "
                                      f"(diff={diff:.4f})")
            result["passed"] = bool(diff > 1e-5)
        except ValueError as e:
            # This is the EXPECTED outcome when in != out: shape mismatch.
            result["naive_status"] = f"correctly FAILED with ValueError: {e}"
            result["passed"] = True
        return result

    def run_all_verifications(self) -> dict:
        """Run all five property checks against the standard inputs."""
        out = self.layer.forward(self.demo_batch)
        return {
            "demo_batch": self.demo_batch,
            "demo_output": out,
            "sample_a": self.sample_a,
            "sample_b": self.sample_b,
            "tests": [
                self.verify_shape_transformation(),
                self.verify_additivity(),
                self.verify_homogeneity(),
                self.verify_bias_behavior(),
                self.verify_transpose_convention(),
            ],
        }


# ============================================================================
# 4. PLOTTER
# ============================================================================

class LinearProjectionPlotter:
    """Renders the inspector's results into three figures."""

    @staticmethod
    def plot_io_transformation(demo_batch: np.ndarray, demo_output: np.ndarray,
                                save_path: str =
                                "fig_linear_projection_io_transformation.png"
                                ) -> None:
        """Two-panel plot: one sample's input (in_features) -> output (out_features)."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        sample_idx = 0  # just show the first sample for clarity
        x_in = np.arange(demo_batch.shape[1])
        x_out = np.arange(demo_output.shape[1])

        axes[0].bar(x_in, demo_batch[sample_idx], color="steelblue",
                    edgecolor="black", linewidth=0.5)
        axes[0].axhline(0, color="black", linewidth=0.7)
        axes[0].set_title(f"Linear Projection — Input vector (sample {sample_idx})\n"
                          f"Dimensionality = {demo_batch.shape[1]} (in_features)")
        axes[0].set_xlabel("Feature index")
        axes[0].set_ylabel("Input value")
        axes[0].set_xticks(x_in)
        axes[0].grid(alpha=0.3, axis="y")

        axes[1].bar(x_out, demo_output[sample_idx], color="seagreen",
                    edgecolor="black", linewidth=0.5)
        axes[1].axhline(0, color="black", linewidth=0.7)
        axes[1].set_title(f"Linear Projection — Output vector (after y = x @ W.T + b)\n"
                          f"Dimensionality = {demo_output.shape[1]} "
                          f"(out_features)  <-- dimension changed!")
        axes[1].set_xlabel("Feature index")
        axes[1].set_ylabel("Output value")
        axes[1].set_xticks(x_out)
        axes[1].grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def plot_linearity(sample_a: np.ndarray, sample_b: np.ndarray,
                       layer: LinearProjection,
                       save_path: str =
                       "fig_linear_projection_linearity.png") -> None:
        """Four-panel proof: f(a), f(b), f(a+b), f(a)+f(b) -- last two must match."""
        fa = layer.forward(sample_a)
        fb = layer.forward(sample_b)
        f_ab = layer.forward(sample_a + sample_b)
        fa_plus_fb = fa + fb
        diff = np.abs(f_ab - fa_plus_fb).max()

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
        x = np.arange(fa.shape[-1])
        panels = [
            (axes[0, 0], fa, "f(a)  --  apply layer to a", "steelblue"),
            (axes[0, 1], fb, "f(b)  --  apply layer to b", "darkorange"),
            (axes[1, 0], f_ab, "f(a + b)  --  apply layer to (a + b)", "seagreen"),
            (axes[1, 1], fa_plus_fb, "f(a) + f(b)  --  add the two outputs",
             "crimson"),
        ]
        for ax, vec, title, color in panels:
            ax.bar(x, vec, color=color, edgecolor="black", linewidth=0.5)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_title(title)
            ax.set_xlabel("Output feature index")
            ax.set_ylabel("Output value")
            ax.set_xticks(x)
            ax.grid(alpha=0.3, axis="y")
        # Mark that the bottom two panels MUST match (linearity proof).
        axes[1, 1].set_title(
            axes[1, 1].get_title() +
            f"\n^^ must equal f(a + b) above (max diff = {diff:.2e})"
        )
        if diff < 1e-5:
            axes[1, 1].text(
                0.5, 0.95, "MATCH ✓",
                transform=axes[1, 1].transAxes, ha="center", va="top",
                fontsize=14, color="green", fontweight="bold",
            )
        else:
            axes[1, 1].text(
                0.5, 0.95, "MISMATCH ✗",
                transform=axes[1, 1].transAxes, ha="center", va="top",
                fontsize=14, color="red", fontweight="bold",
            )
        fig.suptitle("Linearity Proof: f(a + b) = f(a) + f(b)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def plot_bias_effect(layer: LinearProjection,
                          in_features: int,
                          save_path: str =
                          "fig_linear_projection_bias_effect.png") -> None:
        """Compare f(zeros) with bias vs. without bias (rebuild a no-bias layer)."""
        zeros = np.zeros((1, in_features), dtype=np.float32)
        # Output WITH bias (from the supplied layer).
        out_with_bias = layer.forward(zeros)[0]
        # Output WITHOUT bias: zero out the bias temporarily.
        original_bias = layer.bias
        layer.bias = None
        out_no_bias = layer.forward(zeros)[0]
        layer.bias = original_bias  # restore

        x = np.arange(out_with_bias.shape[0])
        width = 0.4
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, out_no_bias, width, color="lightgray",
               edgecolor="black", linewidth=0.5,
               label="Without bias: f(0) = 0  (zeros)")
        ax.bar(x + width / 2, out_with_bias, width, color="indianred",
               edgecolor="black", linewidth=0.5,
               label="With bias: f(0) = b  (the bias vector)")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title("Bias Effect on Zero Input\n"
                     "The bias is exactly what f(zeros) returns.")
        ax.set_xlabel("Output feature index")
        ax.set_ylabel("Output value")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()


# ============================================================================
# 5. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Reference Card 10.2 — Linear Projection Inspection")
    print("=" * 70)

    cfg = LinearProjectionInspectionConfig()
    inspector = LinearProjectionInspector(cfg)
    results = inspector.run_all_verifications()

    # Print the input/output for the first demo sample.
    print(f"\nDemo batch shape: {results['demo_batch'].shape}")
    print(f"Demo output shape: {results['demo_output'].shape}")
    print(f"\nSample 0 input  (in_features={cfg.in_features}):")
    print(f"  {results['demo_batch'][0].round(3)}")
    print(f"\nSample 0 output (out_features={cfg.out_features}):")
    print(f"  {results['demo_output'][0].round(3)}")
    if cfg.use_bias:
        print(f"\nBias vector (out_features={cfg.out_features}):")
        print(f"  {inspector.layer.bias.round(3)}")
    print(f"\nWeight matrix shape: {inspector.layer.weight.shape} "
          f"(out_features, in_features)")

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
            print(f"    {k}: {v}")

    # Render visualizations.
    print("\n" + "-" * 70)
    print("Generating visualizations...")
    print("-" * 70)
    LinearProjectionPlotter.plot_io_transformation(
        results["demo_batch"], results["demo_output"]
    )
    LinearProjectionPlotter.plot_linearity(
        results["sample_a"], results["sample_b"], inspector.layer
    )
    LinearProjectionPlotter.plot_bias_effect(
        inspector.layer, cfg.in_features
    )

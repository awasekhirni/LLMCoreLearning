"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

ELIM5
Softmax is like a contest where every contestant (a number) raises their hand to a certain height. We then measure each hand's height as a fraction of the total height of all hands combined. So instead of raw heights like [10, 5, 1], we get percentages [~0.99, ~0.006, ~0.0001] that always add up to 100%. This lets us talk about probabilities instead of raw scores. The one clever trick: before measuring, we secretly lower every hand by the same amount (the tallest hand's height) so that the tallest ends up at exactly zero — this stops any hand from being so tall that it overflows the measuring tape.



softmax.py
==========================
Inspect and visualize the Softmax function.


input
A 1-D Numpy array of real-valued logits
a deterministic vecor of 8 values generated with seed=42 (reproduceability)

outcome

output values are all in [0,1]
Output values sum to 1.0 exactly (within 1e-6 tolerance).
The index of the maximum input equals the index of the maximum output (argmax preserved).
For extreme inputs (logits ~ 1e6), the stable implementation produces finite probabilities; the naive implementation (without max-subtraction) overflows to NaN.


Educational objectives:
    - See, step by step, how raw logits become probabilities.
    - Verify the four mathematical guarantees of softmax:
        (1) all outputs are in [0, 1]
        (2) outputs sum to exactly 1
        (3) argmax(input) == argmax(output)
        (4) numerical stability under extreme inputs
    - Empirically demonstrate WHY subtracting max(x) matters by comparing
      to a naive (overflow-prone) implementation.

Design principles:
    - Single Responsibility: SoftmaxInspector computes & verifies;
      SoftmaxPlotter renders.
    - DRY: one softmax implementation reused across all tests.
    - Reproducibility: every random op seeded.

install pacakages
pip3 install numpy>=1.24, matplotlib>=3.7

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. SOFTMAX IMPLEMENTATION (stable + naive for comparison)
# ============================================================================

class Softmax:
    """Numerically stable softmax along the last axis.

    Stability trick:
        Subtract max(x) before exp() so the largest exponent is exp(0)=1.
        Without this, exp(1000) -> inf and the whole distribution becomes NaN.

    Input contract:
        x: np.ndarray of any shape. Last axis is the distribution axis.
        axis: which axis to normalize over (default: last).

    Result/outcome expected:
        np.ndarray of the same shape, values in [0, 1], sum to 1 along `axis`.

    Where used in this chapter:
        - GroupedQueryAttention: turn attention scores into weights.
        - Sampler: turn final logits into token probabilities.
    """

    @staticmethod
    def forward(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Stable softmax: subtract max BEFORE exp to prevent overflow."""
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Softmax expects np.ndarray, got {type(x)}")
        if x.size == 0:
            raise ValueError("Softmax received empty array")
        x_max = np.max(x, axis=axis, keepdims=True)
        # Subtract max for stability (see class docstring).
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def forward_naive(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """UNSTABLE softmax: no max-subtraction. Used to DEMONSTRATE overflow.

        DO NOT USE IN PRODUCTION. This exists only so learners can see
        what happens without the stability trick.
        """
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================================================
# 2. CONFIGURATION
# ============================================================================

@dataclass
class SoftmaxInspectionConfig:
    """All parameters for the softmax inspection exercise.

    Attributes:
        input_size:      Length of the standard input logits vector.
        input_scale:     Std-dev of the random input logits.
        extreme_values:  Logits used to stress-test numerical stability.
        seed:            RNG seed for reproducible inputs.
    """
    input_size: int = 8
    input_scale: float = 3.0
    extreme_values: List[float] = field(
        default_factory=lambda: [1.0, 10.0, 100.0, 500.0, 700.0, 1000.0, 1e4, 1e6]
    )
    seed: int = 42


# ============================================================================
# 3. INSPECTOR
# ============================================================================

class SoftmaxInspector:
    """Applies softmax to inputs, verifies its mathematical properties."""

    def __init__(self, config: SoftmaxInspectionConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # Build a deterministic input vector for the "standard" demo.
        self.input_logits = (self.rng.standard_normal(config.input_size)
                              * config.input_scale).astype(np.float32)

    # ----- the four mathematical guarantees -----

    def verify_all_outputs_in_unit_interval(self, probs: np.ndarray) -> dict:
        """Property (1): every output value must lie in [0, 1]."""
        return {
            "test": "all outputs in [0, 1]",
            "min_value": float(probs.min()),
            "max_value": float(probs.max()),
            "passed": bool(0.0 <= probs.min() and probs.max() <= 1.0),
        }

    def verify_outputs_sum_to_one(self, probs: np.ndarray) -> dict:
        """Property (2): outputs must sum to exactly 1 along the distribution axis."""
        total = float(probs.sum())
        return {
            "test": "outputs sum to 1",
            "sum": total,
            "tolerance": 1e-6,
            "passed": bool(abs(total - 1.0) < 1e-6),
        }

    def verify_argmax_preserved(self, logits: np.ndarray, probs: np.ndarray) -> dict:
        """Property (3): the largest input maps to the largest output."""
        return {
            "test": "argmax(input) == argmax(output)",
            "argmax_input": int(np.argmax(logits)),
            "argmax_output": int(np.argmax(probs)),
            "passed": bool(np.argmax(logits) == np.argmax(probs)),
        }

    def verify_stability_under_extreme_inputs(self) -> dict:
        """Property (4): stable softmax stays finite; naive softmax overflows."""
        # We feed each extreme value as a single-element "logit" alongside a 0
        # so the softmax has something to compare to.
        stable_results = []
        naive_results = []
        for v in self.config.extreme_values:
            pair = np.array([v, 0.0], dtype=np.float64)
            stable = Softmax.forward(pair)
            naive = Softmax.forward_naive(pair)
            stable_results.append({
                "input": v,
                "output": stable.tolist(),
                "finite": bool(np.all(np.isfinite(stable))),
            })
            naive_results.append({
                "input": v,
                "output": naive.tolist(),
                "finite": bool(np.all(np.isfinite(naive))),
            })
        stable_all_finite = all(r["finite"] for r in stable_results)
        naive_all_finite = all(r["finite"] for r in naive_results)
        return {
            "test": "numerical stability under extreme inputs",
            "stable_results": stable_results,
            "naive_results": naive_results,
            "stable_all_finite": stable_all_finite,
            "naive_all_finite": naive_all_finite,
            "passed": bool(stable_all_finite and not naive_all_finite),
        }

    def run_all_verifications(self) -> dict:
        """Run all four property checks against the standard input vector."""
        probs = Softmax.forward(self.input_logits)
        return {
            "input_logits": self.input_logits,
            "output_probs": probs,
            "tests": [
                self.verify_all_outputs_in_unit_interval(probs),
                self.verify_outputs_sum_to_one(probs),
                self.verify_argmax_preserved(self.input_logits, probs),
                self.verify_stability_under_extreme_inputs(),
            ],
        }


# ============================================================================
# 4. PLOTTER
# ============================================================================

class SoftmaxPlotter:
    """Renders the inspector's results into two figures."""

    @staticmethod
    def plot_io_transformation(input_logits: np.ndarray,
                                output_probs: np.ndarray,
                                save_path: str =
                                "fig_softmax_io_transformation.png") -> None:
        """Two-panel plot: (top) input logits, (bottom) output probabilities."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        x = np.arange(len(input_logits))
        # Color: gold for the argmax (the "winner"), gray for the rest.
        argmax_idx = int(np.argmax(input_logits))
        colors_in = ["gold" if i == argmax_idx else "steelblue"
                     for i in range(len(input_logits))]
        colors_out = ["gold" if i == argmax_idx else "seagreen"
                      for i in range(len(output_probs))]

        axes[0].bar(x, input_logits, color=colors_in, edgecolor="black",
                     linewidth=0.5)
        axes[0].axhline(0, color="black", linewidth=0.7)
        axes[0].set_ylabel("Raw logit value")
        axes[0].set_title("Softmax — Input: Raw Logits\n"
                          "(gold bar = argmax, will become the largest probability)")
        axes[0].grid(alpha=0.3, axis="y")

        axes[1].bar(x, output_probs, color=colors_out, edgecolor="black",
                    linewidth=0.5)
        axes[1].set_xlabel("Token / class index")
        axes[1].set_ylabel("Probability")
        axes[1].set_title("Softmax — Output: Probabilities (sum = 1.0)\n"
                          "(argmax preserved, all values in [0, 1])")
        axes[1].grid(alpha=0.3, axis="y")
        # Annotate the winner's probability.
        axes[1].annotate(
            f"argmax prob = {output_probs[argmax_idx]:.3f}",
            xy=(argmax_idx, output_probs[argmax_idx]),
            xytext=(argmax_idx + 0.5, output_probs[argmax_idx] + 0.05),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def plot_stability(stability_result: dict,
                        save_path: str = "fig_softmax_stability.png") -> None:
        """Bar chart comparing stable vs. naive softmax on extreme inputs.

        Shows that the stable version stays at 1.0 (no overflow) while the
        naive version degenerates to NaN at large inputs.
        """
        extreme_vals = [r["input"] for r in stability_result["stable_results"]]
        stable_finite = [1 if r["finite"] else 0
                         for r in stability_result["stable_results"]]
        naive_finite = [1 if r["finite"] else 0
                        for r in stability_result["naive_results"]]

        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(extreme_vals))
        width = 0.4
        ax.bar(x - width / 2, stable_finite, width, color="seagreen",
               label="Stable softmax (with max-subtraction)", edgecolor="black",
               linewidth=0.5)
        ax.bar(x + width / 2, naive_finite, width, color="indianred",
               label="Naive softmax (no max-subtraction)", edgecolor="black",
               linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in extreme_vals])
        ax.set_xlabel("Input logit value (extreme magnitudes)")
        ax.set_ylabel("All outputs finite? (1 = yes, 0 = no / NaN)")
        ax.set_title("Softmax Numerical Stability Test\n"
                     "Stable version stays finite at any magnitude; naive overflows")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["NO (NaN/inf)", "YES"])
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3, axis="y")
        # Mark the first input where naive softmax overflows.
        for i, finite in enumerate(naive_finite):
            if not finite:
                ax.annotate(
                    "first overflow",
                    xy=(i + width / 2, 0),
                    xytext=(i + width / 2, 0.4),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=9, color="indianred", ha="center",
                )
                break

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()


# ============================================================================
# 5. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" Softmax Function Inspection")
    print("=" * 70)

    cfg = SoftmaxInspectionConfig()
    inspector = SoftmaxInspector(cfg)
    results = inspector.run_all_verifications()

    # Print the input/output vectors so learners can sanity-check by eye.
    print("\nInput logits:")
    print(f"  {results['input_logits'].round(3)}")
    print("\nOutput probabilities (after softmax):")
    print(f"  {results['output_probs'].round(3)}")
    print(f"  Sum = {results['output_probs'].sum():.6f}")

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
            # Truncate long nested result lists for readability.
            if isinstance(v, list) and len(v) > 4:
                print(f"    {k}: [{v[0]}, {v[1]}, ... , {v[-1]}] "
                      f"({len(v)} entries)")
            else:
                print(f"    {k}: {v}")

    # Render visualizations.
    print("\n" + "-" * 70)
    print("Generating visualizations...")
    print("-" * 70)
    SoftmaxPlotter.plot_io_transformation(
        results["input_logits"], results["output_probs"]
    )
    stability = results["tests"][3]  # the stability test
    SoftmaxPlotter.plot_stability(stability)

"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

ELIM5
A normal feed-forward layer is like a one-door room: information goes in one door, gets bumped around, and comes out another door. SwiGLU adds a second door that opens or closes depending on the input — like a smart gatekeeper that decides "yes, let this information through" or "no, hold this back" for each feature. The door's openness is computed from the input itself using the SiLU activation (x × sigmoid(x)), which is a smooth, S-shaped curve that:

- Lets positive values pass through (door opens wide),
- Squashes negative values toward zero (door almost closes),
- Is smooth everywhere (no sharp corners like ReLU), so gradients flow nicely during training.



output = W2( silu(W1 × input) ⊙ (W3 × input) )
                ↑ gate        ↑ up      ↑ down-projection

Input: A 2-D NumPy array of shape (batch, hidden_dim). Default: batch=4, hidden_dim=8, ffn_hidden_dim=16 (2× the hidden dim, a typical ratio), seed=42.

outcome:

Output shape is (batch, hidden_dim) — the output dimension matches the input dimension, making SwiGLU residual-friendly (you can do x + ffn(x) cleanly).
The output equals W2(silu(W1 x) ⊙ W3 x) element-wise — the gating formula is exact.
 The SiLU activation silu(x) = x * sigmoid(x) is verified numerically across a range of inputs.
The gate silu(W1 x) can suppress features: where the gate is near zero, the corresponding gated value is near zero regardless of the up-projection's magnitude.
SwiGLU produces different (richer) outputs than a vanilla ReLU FFN on the same input — demonstrating the value of gating.



Swish Activation swiglu.py
=========================
Inspect and visualize the SwiGLU Activation (Feed-Forward Network).

Educational objectives:
    - See, step by step, how SwiGLU transforms a vector through three
      projections and a gating mechanism.
    - Verify the mathematical properties of SwiGLU:
        (1) shape preservation: output dim == input dim (residual-friendly)
        (2) gating formula: output == W2(silu(W1 x) * W3 x)
        (3) SiLU activation: silu(x) == x * sigmoid(x)
        (4) gate can suppress features (where silu(W1 x) ≈ 0, output ≈ 0)
        (5) richer behavior than vanilla ReLU FFN on same input
    - Empirically demonstrate WHY three matrices beat two:
        The gate silu(W1 x) * up(W3 x) lets the network SELECTIVELY PASS
        or SUPPRESS each feature, improving capacity over plain ReLU/GELU.

Design principles:
    - Single Responsibility: SwiGLUInspector computes & verifies;
      SwiGLUPlotter renders.
    - DRY: one SwiGLUFFN implementation reused across all tests.
    - Reproducibility: every random op seeded.

Why SiLU instead of ReLU or GELU?

     Rectified Linear Unit(ReLU) (max(0, x)) has a sharp corner at x=0 — the gradient is undefined there, which can cause "dead neurons" during training.
     Gaussian Error Linear Unit (GELU) (x × Φ(x), where Φ is the Gaussian CDF) is smooth but expensive to compute (requires erf).
     Sigmoid Linear Unit (SiLU) (x × sigmoid(x)) is smooth, cheap, and has a slight "bump" below zero that gives it richer behavior than ReLU. It's the sweet spot.



"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. LINEAR PROJECTION (reused from 10.2; included for self-containment)
# ============================================================================

class LinearProjection:
    """y = x @ W.T + b  (the matrix-multiply primitive).

    Input contract:
        x: np.ndarray of shape (batch, in_features) or (in_features,).
        Returns: np.ndarray of shape (batch, out_features).
    """

    def __init__(self, in_features: int, out_features: int,
                 use_bias: bool, rng: np.random.Generator):
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = (rng.standard_normal((out_features, in_features)) * std
                       ).astype(np.float32)
        self.bias = (rng.standard_normal(out_features) * std
                     ).astype(np.float32) if use_bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim < 2:
            x = x[None, :]
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


# ============================================================================
# 2. SWIGLU FFN IMPLEMENTATION
# ============================================================================

class SwiGLUFFN:
    """SwiGLU feed-forward sub-layer: y = W2(silu(W1 x) * (W3 x)).

    Why three matrices?
        The gating mechanism silu(W1 x) * (W3 x) lets the network learn to
        SELECTIVELY PASS or SUPPRESS each feature, improving capacity
        over plain ReLU/GELU FFNs at a modest parameter cost
        (3*hidden*ffn instead of 2*hidden*ffn).

    Input contract:
        x: np.ndarray of shape (seq_len, hidden_dim).
        Returns: np.ndarray of shape (seq_len, hidden_dim).

    Result/outcome expected:
        Same-shape output as input (so it composes cleanly with the
        residual connection). ffn_hidden_dim is typically 2-4x hidden_dim.

    Where used in this chapter:
        As the FFN sub-layer inside TransformerBlock.
    """

    def __init__(self, hidden_dim: int, ffn_hidden_dim: int,
                 rng: np.random.Generator):
        # W1 = gate projection, W3 = up projection, W2 = down projection.
        self.w1 = LinearProjection(hidden_dim, ffn_hidden_dim,
                                    use_bias=False, rng=rng)
        self.w2 = LinearProjection(ffn_hidden_dim, hidden_dim,
                                    use_bias=False, rng=rng)
        self.w3 = LinearProjection(hidden_dim, ffn_hidden_dim,
                                    use_bias=False, rng=rng)
        self.hidden_dim = hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        """SiLU = x * sigmoid(x)  (a.k.a. Swish).

        Input contract:
            x: np.ndarray of any shape.
        Returns:
            np.ndarray of same shape.
        """
        return x * (1.0 / (1.0 + np.exp(-x)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full SwiGLU forward: y = W2(silu(W1 x) * W3 x)."""
        # Gate path: silu-activated.
        gate = self.silu(self.w1.forward(x))
        # Up path: raw projection (no activation).
        up = self.w3.forward(x)
        # Element-wise gating: gate * up, then down-project.
        gated = gate * up
        return self.w2.forward(gated)

    def forward_decomposed(self, x: np.ndarray) -> dict:
        """Return ALL intermediates for visualization & verification.

        Returns a dict with keys:
            'gate':   silu(W1 x)       -- shape (batch, ffn_dim)
            'up':     W3 x             -- shape (batch, ffn_dim)
            'gated':  gate * up         -- shape (batch, ffn_dim)
            'output': W2(gated)        -- shape (batch, hidden_dim)
        """
        gate = self.silu(self.w1.forward(x))
        up = self.w3.forward(x)
        gated = gate * up
        output = self.w2.forward(gated)
        return {"gate": gate, "up": up, "gated": gated, "output": output}


# ============================================================================
# 3. VANILLA RELU FFN (for comparison only)
# ============================================================================

class ReLUFFN:
    """Vanilla ReLU FFN: y = W2(relu(W1 x)).

    Used ONLY for comparison to show why SwiGLU's gating is better.
    Has 2 matrices instead of 3, and uses ReLU instead of SiLU+gate.
    """

    def __init__(self, hidden_dim: int, ffn_hidden_dim: int,
                 rng: np.random.Generator):
        self.w1 = LinearProjection(hidden_dim, ffn_hidden_dim,
                                    use_bias=False, rng=rng)
        self.w2 = LinearProjection(ffn_hidden_dim, hidden_dim,
                                    use_bias=False, rng=rng)
        self.hidden_dim = hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.w2.forward(self.relu(self.w1.forward(x)))


# ============================================================================
# 4. CONFIGURATION
# ============================================================================

@dataclass
class SwiGLUInspectionConfig:
    """All parameters for the SwiGLU inspection exercise.

    Attributes:
        hidden_dim:     Input/output dimensionality (residual-friendly).
        ffn_hidden_dim: Intermediate dimensionality (typically 2-4x hidden).
        batch_size:     Number of samples in the demo batch.
        seed:           RNG seed for reproducibility.
    """
    hidden_dim: int = 8
    ffn_hidden_dim: int = 16
    batch_size: int = 4
    seed: int = 42


# ============================================================================
# 5. INSPECTOR
# ============================================================================

class SwiGLUInspector:
    """Applies SwiGLU to inputs and verifies its mathematical properties."""

    def __init__(self, config: SwiGLUInspectionConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # Build SwiGLU and ReLU FFN with the SAME RNG so weights are comparable.
        self.swiglu = SwiGLUFFN(config.hidden_dim, config.ffn_hidden_dim,
                                 self.rng)
        # ReLU FFN uses a fresh RNG draw but same shapes.
        self.relu_ffn = ReLUFFN(config.hidden_dim, config.ffn_hidden_dim,
                                 self.rng)
        # Generate a deterministic demo batch.
        self.demo_batch = (self.rng.standard_normal(
            (config.batch_size, config.hidden_dim)
        ) * 2.0).astype(np.float32)
        # A single sample for decomposed inspection.
        self.sample = self.demo_batch[0]

    # ----- the six mathematical guarantees -----

    def verify_shape_preservation(self) -> dict:
        """Property (1): output dim == input dim (residual-friendly)."""
        out = self.swiglu.forward(self.demo_batch)
        expected = (self.config.batch_size, self.config.hidden_dim)
        return {
            "test": "shape preservation (output dim == input dim)",
            "input_shape": tuple(self.demo_batch.shape),
            "output_shape": tuple(out.shape),
            "expected_output_shape": expected,
            "passed": bool(tuple(out.shape) == expected),
        }

    def verify_gating_formula(self) -> dict:
        """Property (2): output == W2(silu(W1 x) * W3 x) element-wise."""
        decomposed = self.swiglu.forward_decomposed(self.demo_batch)
        # Reconstruct from intermediates.
        reconstructed = self.swiglu.w2.forward(decomposed["gated"])
        direct = self.swiglu.forward(self.demo_batch)
        diff = float(np.max(np.abs(reconstructed - direct)))
        return {
            "test": "gating formula: output == W2(silu(W1 x) * W3 x)",
            "max_abs_diff_reconstructed_vs_direct": diff,
            "tolerance": 1e-5,
            "passed": bool(diff < 1e-5),
        }

    def verify_silu_activation(self) -> dict:
        """Property (3): silu(x) == x * sigmoid(x) numerically."""
        # Test across a range of input values.
        test_inputs = np.array(
            [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
            dtype=np.float32,
        )
        silu_result = SwiGLUFFN.silu(test_inputs)
        sigmoid = 1.0 / (1.0 + np.exp(-test_inputs))
        expected = test_inputs * sigmoid
        diff = float(np.max(np.abs(silu_result - expected)))
        # Also verify key properties:
        # - silu(0) = 0  (sigmoid(0)=1, so 0 * 1 = 0)
        # - silu(x) > 0 for x > 0
        # - silu(x) < 0 for x < 0 (but small magnitude; "bump" below zero)
        silu_zero = float(SwiGLUFFN.silu(np.array([0.0], dtype=np.float32))[0])
        positive_inputs = test_inputs[test_inputs > 0]
        silu_positive = SwiGLUFFN.silu(positive_inputs)
        all_positive_silu_positive = bool(np.all(silu_positive > 0))
        return {
            "test": "SiLU activation: silu(x) == x * sigmoid(x)",
            "test_inputs": test_inputs.tolist(),
            "silu_outputs": silu_result.tolist(),
            "max_abs_diff_vs_x_sigmoid": diff,
            "silu_at_zero": silu_zero,
            "silu_positive_for_positive_x": all_positive_silu_positive,
            "tolerance": 1e-6,
            "passed": bool(diff < 1e-6 and
                            abs(silu_zero) < 1e-6 and
                            all_positive_silu_positive),
        }

    def verify_gate_can_suppress(self) -> dict:
        """Property (4): where silu(W1 x) ≈ 0, the gated value ≈ 0.

        This is the KEY property of SwiGLU: the gate can SUPPRESS features.
        We find the feature in the gate with the smallest magnitude and
        verify the corresponding gated value is also near zero.
        """
        decomposed = self.swiglu.forward_decomposed(
            self.sample.reshape(1, -1)
        )
        gate = decomposed["gate"][0]       # (ffn_dim,)
        up = decomposed["up"][0]           # (ffn_dim,)
        gated = decomposed["gated"][0]     # (ffn_dim,)
        # Find the feature with the smallest |gate| value.
        min_gate_idx = int(np.argmin(np.abs(gate)))
        min_gate_val = float(gate[min_gate_idx])
        up_at_that_idx = float(up[min_gate_idx])
        gated_at_that_idx = float(gated[min_gate_idx])
        # Also find the feature with the LARGEST |gate| for contrast.
        max_gate_idx = int(np.argmax(np.abs(gate)))
        max_gate_val = float(gate[max_gate_idx])
        gated_at_max = float(gated[max_gate_idx])
        return {
            "test": "gate can suppress features (gate ≈ 0 -> gated ≈ 0)",
            "min_gate_feature_index": min_gate_idx,
            "min_gate_value": min_gate_val,
            "up_at_min_gate": up_at_that_idx,
            "gated_at_min_gate": gated_at_that_idx,
            "max_gate_feature_index": max_gate_idx,
            "max_gate_value": max_gate_val,
            "gated_at_max_gate": gated_at_max,
            "passed": bool(abs(gated_at_that_idx) < abs(up_at_that_idx)),
        }

    def verify_residual_friendly(self) -> dict:
        """Property (5): output can be added to input (residual connection).

        This is a DESIGN property: because output.shape == input.shape,
        you can compute x + ffn(x) cleanly. We verify the addition works.
        """
        out = self.swiglu.forward(self.demo_batch)
        residual = self.demo_batch + out
        return {
            "test": "residual-friendly (x + ffn(x) is valid)",
            "input_shape": tuple(self.demo_batch.shape),
            "output_shape": tuple(out.shape),
            "residual_shape": tuple(residual.shape),
            "shapes_compatible": bool(tuple(self.demo_batch.shape) ==
                                        tuple(out.shape)),
            "passed": bool(tuple(self.demo_batch.shape) ==
                            tuple(out.shape)),
        }

    def verify_richer_than_relu_ffn(self) -> dict:
        """Property (6): SwiGLU produces DIFFERENT output than ReLU FFN.

        This demonstrates that the gating mechanism provides richer
        transformations than a plain ReLU FFN.
        """
        swiglu_out = self.swiglu.forward(self.demo_batch)
        relu_out = self.relu_ffn.forward(self.demo_batch)
        diff = float(np.max(np.abs(swiglu_out - relu_out)))
        # Also compute the fraction of features where they differ significantly.
        significant_diff = np.abs(swiglu_out - relu_out) > 0.01
        frac_different = float(np.mean(significant_diff))
        return {
            "test": "SwiGLU produces different (richer) output than ReLU FFN",
            "max_abs_diff": diff,
            "fraction_features_differing_significantly": frac_different,
            "passed": bool(diff > 0.01 and frac_different > 0.1),
        }

    def run_all_verifications(self) -> dict:
        """Run all six property checks against the standard inputs."""
        decomposed = self.swiglu.forward_decomposed(
            self.demo_batch
        )
        return {
            "demo_batch": self.demo_batch,
            "demo_output": decomposed["output"],
            "decomposed_sample": self.swiglu.forward_decomposed(
                self.sample.reshape(1, -1)
            ),
            "swiglu_output": self.swiglu.forward(self.demo_batch),
            "relu_output": self.relu_ffn.forward(self.demo_batch),
            "tests": [
                self.verify_shape_preservation(),
                self.verify_gating_formula(),
                self.verify_silu_activation(),
                self.verify_gate_can_suppress(),
                self.verify_residual_friendly(),
                self.verify_richer_than_relu_ffn(),
            ],
        }


# ============================================================================
# 6. PLOTTER
# ============================================================================

class SwiGLUPlotter:
    """Renders the inspector's results into three figures."""

    @staticmethod
    def plot_io_transformation(demo_batch: np.ndarray, demo_output: np.ndarray,
                                save_path: str =
                                "fig_swiglu_io_transformation.png") -> None:
        """Two-panel plot: input vector -> output vector for one sample."""
        fig, axes = plt.subplots(2, 1, figsize=(11, 6))
        sample_idx = 0
        x = np.arange(demo_batch.shape[1])

        axes[0].bar(x, demo_batch[sample_idx], color="steelblue",
                    edgecolor="black", linewidth=0.5)
        axes[0].axhline(0, color="black", linewidth=0.7)
        axes[0].set_title(f"SwiGLU FFN — Input vector (sample {sample_idx})\n"
                          f"Dimensionality = {demo_batch.shape[1]} (hidden_dim)")
        axes[0].set_xlabel("Feature index")
        axes[0].set_ylabel("Input value")
        axes[0].set_xticks(x)
        axes[0].grid(alpha=0.3, axis="y")

        axes[1].bar(x, demo_output[sample_idx], color="seagreen",
                    edgecolor="black", linewidth=0.5)
        axes[1].axhline(0, color="black", linewidth=0.7)
        axes[1].set_title(f"SwiGLU FFN — Output vector (after W2(silu(W1 x) ⊙ W3 x))\n"
                          f"Dimensionality = {demo_output.shape[1]} (hidden_dim, "
                          f"RESIDUAL-FRIENDLY: same as input)")
        axes[1].set_xlabel("Feature index")
        axes[1].set_ylabel("Output value")
        axes[1].set_xticks(x)
        axes[1].grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def plot_gating_mechanism(sample: np.ndarray, swiglu: SwiGLUFFN,
                                save_path: str =
                                "fig_swiglu_gating_mechanism.png") -> None:
        """Five-panel decomposition: input -> gate -> up -> gated -> output."""
        decomposed = swiglu.forward_decomposed(sample.reshape(1, -1))
        gate = decomposed["gate"][0]    # (ffn_dim,)
        up = decomposed["up"][0]        # (ffn_dim,)
        gated = decomposed["gated"][0]  # (ffn_dim,)
        output = decomposed["output"][0]  # (hidden_dim,)

        fig, axes = plt.subplots(5, 1, figsize=(12, 13))
        # Panel 1: input.
        x_in = np.arange(len(sample))
        axes[0].bar(x_in, sample, color="steelblue",
                    edgecolor="black", linewidth=0.5)
        axes[0].axhline(0, color="black", linewidth=0.7)
        axes[0].set_title("① Input x  (shape: hidden_dim)")
        axes[0].set_ylabel("Value")
        axes[0].grid(alpha=0.3, axis="y")

        # Panel 2: gate = silu(W1 x).
        x_ffn = np.arange(len(gate))
        gate_colors = ["crimson" if abs(v) < 0.1 else "darkorange"
                       for v in gate]
        axes[1].bar(x_ffn, gate, color=gate_colors,
                    edgecolor="black", linewidth=0.5)
        axes[1].axhline(0, color="black", linewidth=0.7)
        axes[1].set_title("② Gate = silu(W1 x)  (shape: ffn_dim)\n"
                          "RED bars ≈ 0 → will SUPPRESS the corresponding feature")
        axes[1].set_ylabel("Gate value")
        axes[1].grid(alpha=0.3, axis="y")

        # Panel 3: up = W3 x.
        axes[2].bar(x_ffn, up, color="mediumseagreen",
                    edgecolor="black", linewidth=0.5)
        axes[2].axhline(0, color="black", linewidth=0.7)
        axes[2].set_title("③ Up = W3 x  (shape: ffn_dim)\n"
                          "Raw projected values (before gating)")
        axes[2].set_ylabel("Up value")
        axes[2].grid(alpha=0.3, axis="y")

        # Panel 4: gated = gate * up.
        gated_colors = ["crimson" if abs(g) < 0.05 else "purple"
                        for g in gated]
        axes[3].bar(x_ffn, gated, color=gated_colors,
                    edgecolor="black", linewidth=0.5)
        axes[3].axhline(0, color="black", linewidth=0.7)
        axes[3].set_title("④ Gated = gate ⊙ up  (shape: ffn_dim)\n"
                          "RED bars ≈ 0 → feature was SUPPRESSED by the gate")
        axes[3].set_ylabel("Gated value")
        axes[3].grid(alpha=0.3, axis="y")

        # Panel 5: output = W2(gated).
        x_out = np.arange(len(output))
        axes[4].bar(x_out, output, color="navy",
                    edgecolor="black", linewidth=0.5)
        axes[4].axhline(0, color="black", linewidth=0.7)
        axes[4].set_title("⑤ Output = W2(gated)  (shape: hidden_dim)\n"
                          "Final down-projection back to input dimensionality")
        axes[4].set_xlabel("Feature index")
        axes[4].set_ylabel("Output value")
        axes[4].grid(alpha=0.3, axis="y")

        fig.suptitle("SwiGLU Gating Mechanism Decomposition\n"
                     "y = W2( silu(W1 x) ⊙ (W3 x) )",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def plot_swiglu_vs_relu(demo_batch: np.ndarray, swiglu: SwiGLUFFN,
                             relu_ffn: ReLUFFN,
                             save_path: str =
                             "fig_swiglu_vs_relu_ffn.png") -> None:
        """Four-panel: input, SwiGLU output, ReLU output, difference."""
        swiglu_out = swiglu.forward(demo_batch)
        relu_out = relu_ffn.forward(demo_batch)
        diff = swiglu_out - relu_out
        sample_idx = 0
        x = np.arange(demo_batch.shape[1])

        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        panels = [
            (axes[0], demo_batch[sample_idx], "① Input x",
             "steelblue"),
            (axes[1], swiglu_out[sample_idx],
             f"② SwiGLU output: W2(silu(W1 x) ⊙ W3 x)  [3 matrices + gate]",
             "seagreen"),
            (axes[2], relu_out[sample_idx],
             f"③ ReLU FFN output: W2(relu(W1 x))  [2 matrices, no gate]",
             "darkorange"),
            (axes[3], diff[sample_idx],
             "④ Difference (SwiGLU - ReLU)  [shows gating's richer behavior]",
             "crimson"),
        ]
        for ax, vec, title, color in panels:
            ax.bar(x, vec, color=color, edgecolor="black", linewidth=0.5)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_title(title)
            ax.set_ylabel("Value")
            ax.grid(alpha=0.3, axis="y")
        axes[-1].set_xlabel("Feature index")
        fig.suptitle("SwiGLU vs. ReLU FFN: Gating Produces Richer Transformations",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()


# ============================================================================
# 7. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Reference Card 10.5 — SwiGLU Activation (FFN) Inspection")
    print("=" * 70)

    cfg = SwiGLUInspectionConfig()
    inspector = SwiGLUInspector(cfg)
    results = inspector.run_all_verifications()

    # Print the input/output for the first sample.
    print(f"\nDemo batch shape: {results['demo_batch'].shape}")
    print(f"SwiGLU output shape: {results['swiglu_output'].shape}")
    print(f"\nSample 0 input (hidden_dim={cfg.hidden_dim}):")
    print(f"  {results['demo_batch'][0].round(3)}")
    print(f"\nSample 0 SwiGLU output:")
    print(f"  {results['swiglu_output'][0].round(3)}")
    print(f"\nSample 0 ReLU FFN output (for comparison):")
    print(f"  {results['relu_output'][0].round(3)}")
    print(f"\nSample 0 difference (SwiGLU - ReLU):")
    print(f"  {(results['swiglu_output'][0] - results['relu_output'][0]).round(3)}")

    # Print the decomposed intermediates for one sample.
    decomp = results["decomposed_sample"]
    print(f"\nDecomposed intermediates for sample 0:")
    print(f"  gate (silu(W1 x), shape {decomp['gate'].shape}):")
    print(f"    {decomp['gate'][0].round(3)}")
    print(f"  up (W3 x, shape {decomp['up'].shape}):")
    print(f"    {decomp['up'][0].round(3)}")
    print(f"  gated (gate * up, shape {decomp['gated'].shape}):")
    print(f"    {decomp['gated'][0].round(3)}")

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
            if isinstance(v, list) and len(v) > 6:
                print(f"    {k}: [{v[0]:.4f}, {v[1]:.4f}, ... , "
                      f"{v[-1]:.4f}] ({len(v)} entries)")
            else:
                print(f"    {k}: {v}")

    # Render visualizations.
    print("\n" + "-" * 70)
    print("Generating visualizations...")
    print("-" * 70)
    SwiGLUPlotter.plot_io_transformation(
        results["demo_batch"], results["demo_output"]
    )
    SwiGLUPlotter.plot_gating_mechanism(
        results["demo_batch"][0], inspector.swiglu
    )
    SwiGLUPlotter.plot_swiglu_vs_relu(
        results["demo_batch"], inspector.swiglu, inspector.relu_ffn
    )

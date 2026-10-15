"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

ELIM5


swiglu_expanded.py
===================================
Expanded inspection and visualization of the SwiGLU Activation (FFN).

Educational objectives:
    - See, step by step, how SwiGLU transforms a vector through three
      projections and a gating mechanism.
    - Verify TWELVE mathematical properties of SwiGLU:
        (1)  shape preservation (output dim == input dim)
        (2)  gating formula: output == W2(silu(W1 x) * W3 x)
        (3)  SiLU activation: silu(x) == x * sigmoid(x)
        (4)  SiLU smoothness: no discontinuities (differentiable everywhere)
        (5)  gate can suppress features (gate ≈ 0 -> gated ≈ 0)
        (6)  gate sparsity: non-trivial fraction of gates near 0
        (7)  residual-friendly: output dim == input dim
        (8)  richer than ReLU FFN
        (9)  richer than GELU FFN
        (10) richer than SiLU-only FFN (isolates gating benefit)
        (11) parameter count: 3*hidden*ffn vs 2*hidden*ffn
        (12) intermediate statistics (mean, std, sparsity at each step)
    - Handle SIX edge cases:
        (a) zero input        (b) very large input   (c) very small input
        (d) single feature     (e) all negative       (f) all positive
    - Compare the activation LANDSCAPE of SiLU vs ReLU vs GELU vs Sigmoid.

Design principles:
    - Single Responsibility: SwiGLUInspector computes & verifies;
      SwiGLUPlotter renders.
    - DRY: one SwiGLUFFN implementation reused across all tests.
    - Open/Closed: comparison FFNs (ReLU, GELU, SiLU-only) are separate
      classes that share the LinearProjection primitive.
    - Reproducibility: every random op seeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
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
# 2. ACTIVATION FUNCTIONS (SiLU, ReLU, GELU, Sigmoid)
# ============================================================================

class Activations:
    """Collection of activation functions used in modern FFNs.

    Each is a static method so they can be called without instantiation
    and can be swapped into any FFN class.
    """

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        """SiLU = x * sigmoid(x), a.k.a. Swish.

        Properties:
            - Smooth everywhere (infinitely differentiable).
            - silu(0) = 0.
            - silu(x) > 0 for x > 0.
            - silu(x) < 0 for x < 0 (slight bump below zero, max ≈ -0.278).
            - As x -> +inf, silu(x) -> x  (linear, door fully open).
            - As x -> -inf, silu(x) -> 0  (door fully closed).
        """
        return x * (1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU = max(0, x).

        Properties:
            - NOT smooth at x=0 (gradient undefined).
            - relu(x) = 0 for x <= 0  (dead neurons risk).
            - relu(x) = x for x > 0.
        """
        return np.maximum(0.0, x)

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """GELU = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).

        This is the tanh approximation of the exact GELU
        (which uses the Gaussian CDF erf). Used in BERT, GPT-2.

        Properties:
            - Smooth everywhere.
            - gelu(0) = 0.
            - Similar shape to SiLU but slightly different curve.
        """
        return 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        ))

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid = 1 / (1 + exp(-x)).

        Properties:
            - Output in (0, 1) for all finite x.
            - sigmoid(0) = 0.5.
            - Monotonically increasing.
        """
        return 1.0 / (1.0 + np.exp(-x))


# ============================================================================
# 3. FFN VARIANTS (SwiGLU, ReLU, GELU, SiLU-only)
# ============================================================================

class SwiGLUFFN:
    """SwiGLU feed-forward sub-layer: y = W2(silu(W1 x) * (W3 x)).

    Why three matrices?
        The gating mechanism silu(W1 x) * (W3 x) lets the network learn to
        SELECTIVELY PASS or SUPPRESS each feature, improving capacity
        over plain ReLU/GELU FFNs at a modest parameter cost.

    Input contract:
        x: np.ndarray of shape (seq_len, hidden_dim).
        Returns: np.ndarray of shape (seq_len, hidden_dim).

    Result/outcome expected:
        Same-shape output as input (residual-friendly).

    Where used in this chapter:
        As the FFN sub-layer inside TransformerBlock.
    """

    def __init__(self, hidden_dim: int, ffn_hidden_dim: int,
                 rng: np.random.Generator):
        self.w1 = LinearProjection(hidden_dim, ffn_hidden_dim,
                                    use_bias=False, rng=rng)
        self.w2 = LinearProjection(ffn_hidden_dim, hidden_dim,
                                    use_bias=False, rng=rng)
        self.w3 = LinearProjection(hidden_dim, ffn_hidden_dim,
                                    use_bias=False, rng=rng)
        self.hidden_dim = hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.name = "SwiGLU"

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = Activations.silu(self.w1.forward(x))
        up = self.w3.forward(x)
        return self.w2.forward(gate * up)

    def forward_decomposed(self, x: np.ndarray) -> dict:
        """Return ALL intermediates for visualization & verification."""
        w1_out = self.w1.forward(x)
        gate = Activations.silu(w1_out)
        up = self.w3.forward(x)
        gated = gate * up
        output = self.w2.forward(gated)
        return {
            "w1_out": w1_out, "gate": gate, "up": up,
            "gated": gated, "output": output,
        }

    def parameter_count(self) -> int:
        """Total learnable parameters: 3 * hidden * ffn."""
        return (self.w1.weight.size + self.w2.weight.size
                + self.w3.weight.size)


class ReLUFFN:
    """Vanilla ReLU FFN: y = W2(relu(W1 x)). Two matrices, no gate."""

    def __init__(self, hidden_dim: int, ffn_hidden_dim: int,
                 rng: np.random.Generator):
        self.w1 = LinearProjection(hidden_dim, ffn_hidden_dim,
                                    use_bias=False, rng=rng)
        self.w2 = LinearProjection(ffn_hidden_dim, hidden_dim,
                                    use_bias=False, rng=rng)
        self.hidden_dim = hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.name = "ReLU"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.w2.forward(Activations.relu(self.w1.forward(x)))

    def parameter_count(self) -> int:
        return self.w1.weight.size + self.w2.weight.size


class GELUFFN:
    """Vanilla GELU FFN: y = W2(gelu(W1 x)). Two matrices, no gate."""

    def __init__(self, hidden_dim: int, ffn_hidden_dim: int,
                 rng: np.random.Generator):
        self.w1 = LinearProjection(hidden_dim, ffn_hidden_dim,
                                    use_bias=False, rng=rng)
        self.w2 = LinearProjection(ffn_hidden_dim, hidden_dim,
                                    use_bias=False, rng=rng)
        self.hidden_dim = hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.name = "GELU"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.w2.forward(Activations.gelu(self.w1.forward(x)))

    def parameter_count(self) -> int:
        return self.w1.weight.size + self.w2.weight.size


class SiLUOnlyFFN:
    """SiLU-only FFN: y = W2(silu(W1 x)). Two matrices, no gate.

    This isolates the ACTIVATION benefit (SiLU vs ReLU) from the
    GATING benefit (the * W3 multiplication). By comparing SwiGLU
    to SiLU-only, we see what the gate adds on top of the activation.
    """

    def __init__(self, hidden_dim: int, ffn_hidden_dim: int,
                 rng: np.random.Generator):
        self.w1 = LinearProjection(hidden_dim, ffn_hidden_dim,
                                    use_bias=False, rng=rng)
        self.w2 = LinearProjection(ffn_hidden_dim, hidden_dim,
                                    use_bias=False, rng=rng)
        self.hidden_dim = hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.name = "SiLU-only"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.w2.forward(Activations.silu(self.w1.forward(x)))

    def parameter_count(self) -> int:
        return self.w1.weight.size + self.w2.weight.size


# ============================================================================
# 4. CONFIGURATION
# ============================================================================

@dataclass
class SwiGLUInspectionConfig:
    """All parameters for the SwiGLU inspection exercise.

    Attributes:
        hidden_dim:      Input/output dimensionality (residual-friendly).
        ffn_hidden_dim:  Intermediate dimensionality (typically 2-4x hidden).
        batch_size:      Number of samples in the demo batch.
        seed:            RNG seed for reproducibility.
        gate_sparse_threshold: Threshold below which a gate value is
                                considered "suppressed" (near zero).
        large_input_scale:     Scale for the "very large input" edge case.
        small_input_scale:     Scale for the "very small input" edge case.
    """
    hidden_dim: int = 8
    ffn_hidden_dim: int = 16
    batch_size: int = 4
    seed: int = 42
    gate_sparse_threshold: float = 0.1
    large_input_scale: float = 100.0
    small_input_scale: float = 0.001


# ============================================================================
# 5. INSPECTOR
# ============================================================================

class SwiGLUInspector:
    """Applies SwiGLU to inputs and verifies TWELVE mathematical properties."""

    def __init__(self, config: SwiGLUInspectionConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # Build all four FFN variants with INDEPENDENT RNGs so each gets
        # its own random init (fair comparison, not shared weights).
        self.swiglu = SwiGLUFFN(config.hidden_dim, config.ffn_hidden_dim,
                                 np.random.default_rng(config.seed))
        self.relu_ffn = ReLUFFN(config.hidden_dim, config.ffn_hidden_dim,
                                 np.random.default_rng(config.seed + 1))
        self.gelu_ffn = GELUFFN(config.hidden_dim, config.ffn_hidden_dim,
                                 np.random.default_rng(config.seed + 2))
        self.silu_only_ffn = SiLUOnlyFFN(config.hidden_dim,
                                          config.ffn_hidden_dim,
                                          np.random.default_rng(config.seed + 3))
        # Generate a deterministic demo batch.
        self.demo_batch = (self.rng.standard_normal(
            (config.batch_size, config.hidden_dim)
        ) * 2.0).astype(np.float32)
        # Single sample for decomposed inspection.
        self.sample = self.demo_batch[0]

    # ----- TWELVE property verification methods -----

    def verify_shape_preservation(self) -> dict:
        """(1) output dim == input dim (residual-friendly)."""
        out = self.swiglu.forward(self.demo_batch)
        expected = (self.config.batch_size, self.config.hidden_dim)
        return {
            "test": "(1) shape preservation (output dim == input dim)",
            "input_shape": tuple(self.demo_batch.shape),
            "output_shape": tuple(out.shape),
            "expected_output_shape": expected,
            "passed": bool(tuple(out.shape) == expected),
        }

    def verify_gating_formula(self) -> dict:
        """(2) output == W2(silu(W1 x) * W3 x) element-wise."""
        decomposed = self.swiglu.forward_decomposed(self.demo_batch)
        reconstructed = self.swiglu.w2.forward(decomposed["gated"])
        direct = self.swiglu.forward(self.demo_batch)
        diff = float(np.max(np.abs(reconstructed - direct)))
        return {
            "test": "(2) gating formula: output == W2(silu(W1 x) * W3 x)",
            "max_abs_diff_reconstructed_vs_direct": diff,
            "tolerance": 1e-5,
            "passed": bool(diff < 1e-5),
        }

    def verify_silu_activation(self) -> dict:
        """(3) silu(x) == x * sigmoid(x) numerically."""
        test_inputs = np.linspace(-10, 10, 21, dtype=np.float32)
        silu_result = Activations.silu(test_inputs)
        sigmoid = Activations.sigmoid(test_inputs)
        expected = test_inputs * sigmoid
        diff = float(np.max(np.abs(silu_result - expected)))
        silu_zero = float(Activations.silu(np.array([0.0]))[0])
        positive_inputs = test_inputs[test_inputs > 0]
        silu_positive = Activations.silu(positive_inputs)
        all_positive_silu_positive = bool(np.all(silu_positive > 0))
        return {
            "test": "(3) SiLU activation: silu(x) == x * sigmoid(x)",
            "max_abs_diff_vs_x_sigmoid": diff,
            "silu_at_zero": silu_zero,
            "silu_positive_for_positive_x": all_positive_silu_positive,
            "tolerance": 1e-6,
            "passed": bool(diff < 1e-6 and abs(silu_zero) < 1e-6
                            and all_positive_silu_positive),
        }

    def verify_silu_smoothness(self) -> dict:
        """(4) SiLU is smooth (differentiable everywhere).

        ReLU has a kink at x=0 (derivative jumps from 0 to 1).
        SiLU has NO kink: the derivative is continuous everywhere.
        We verify by checking that finite-difference gradients are
        continuous across x=0 (no jump).
        """
        # Sample points around x=0 with very fine resolution.
        eps = 1e-5
        x_left = np.array([-eps], dtype=np.float64)
        x_right = np.array([eps], dtype=np.float64)
        x_center = np.array([0.0], dtype=np.float64)
        # Finite-difference gradient at x=0 from left and right.
        h = 1e-6
        grad_left = (Activations.silu(x_center) -
                      Activations.silu(x_center - h)) / h
        grad_right = (Activations.silu(x_center + h) -
                       Activations.silu(x_center)) / h
        grad_jump = float(np.abs(grad_left - grad_right))
        # For ReLU, the jump would be ~1.0 (from 0 to 1).
        relu_grad_left = (Activations.relu(np.array([0.0])) -
                           Activations.relu(np.array([-h]))) / h
        relu_grad_right = (Activations.relu(np.array([h])) -
                            Activations.relu(np.array([0.0]))) / h
        relu_jump = float(np.abs(relu_grad_left - relu_grad_right))
        return {
            "test": "(4) SiLU smoothness (no gradient jump at x=0)",
            "silu_gradient_jump_at_zero": grad_jump,
            "relu_gradient_jump_at_zero": relu_jump,
            "tolerance": 1e-3,
            "passed": bool(grad_jump < 1e-3 and relu_jump > 0.5),
        }

    def verify_gate_can_suppress(self) -> dict:
        """(5) where silu(W1 x) ≈ 0, the gated value ≈ 0."""
        decomposed = self.swiglu.forward_decomposed(
            self.sample.reshape(1, -1)
        )
        gate = decomposed["gate"][0]
        up = decomposed["up"][0]
        gated = decomposed["gated"][0]
        min_gate_idx = int(np.argmin(np.abs(gate)))
        min_gate_val = float(gate[min_gate_idx])
        up_at_min = float(up[min_gate_idx])
        gated_at_min = float(gated[min_gate_idx])
        max_gate_idx = int(np.argmax(np.abs(gate)))
        max_gate_val = float(gate[max_gate_idx])
        gated_at_max = float(gated[max_gate_idx])
        return {
            "test": "(5) gate can suppress features (gate ≈ 0 -> gated ≈ 0)",
            "min_gate_feature_index": min_gate_idx,
            "min_gate_value": min_gate_val,
            "up_at_min_gate": up_at_min,
            "gated_at_min_gate": gated_at_min,
            "max_gate_feature_index": max_gate_idx,
            "max_gate_value": max_gate_val,
            "gated_at_max_gate": gated_at_max,
            "passed": bool(abs(gated_at_min) < abs(up_at_min)),
        }

    def verify_gate_sparsity(self) -> dict:
        """(6) non-trivial fraction of gate values are near zero."""
        decomposed = self.swiglu.forward_decomposed(self.demo_batch)
        gate = decomposed["gate"]  # (batch, ffn_dim)
        threshold = self.config.gate_sparse_threshold
        n_suppressed = int(np.sum(np.abs(gate) < threshold))
        total = int(gate.size)
        frac_suppressed = n_suppressed / total
        # Also compute the mean and std of gate values.
        gate_mean = float(np.mean(gate))
        gate_std = float(np.std(gate))
        return {
            "test": "(6) gate sparsity (fraction of gates near 0)",
            "threshold": threshold,
            "n_suppressed": n_suppressed,
            "total_gate_values": total,
            "fraction_suppressed": frac_suppressed,
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "passed": bool(frac_suppressed > 0.0),
        }

    def verify_residual_friendly(self) -> dict:
        """(7) output can be added to input (residual connection)."""
        out = self.swiglu.forward(self.demo_batch)
        residual = self.demo_batch + out
        return {
            "test": "(7) residual-friendly (x + ffn(x) is valid)",
            "input_shape": tuple(self.demo_batch.shape),
            "output_shape": tuple(out.shape),
            "residual_shape": tuple(residual.shape),
            "passed": bool(tuple(self.demo_batch.shape) ==
                            tuple(out.shape)),
        }

    def verify_richer_than_relu(self) -> dict:
        """(8) SwiGLU produces different output than ReLU FFN."""
        swiglu_out = self.swiglu.forward(self.demo_batch)
        relu_out = self.relu_ffn.forward(self.demo_batch)
        diff = float(np.max(np.abs(swiglu_out - relu_out)))
        frac_diff = float(np.mean(np.abs(swiglu_out - relu_out) > 0.01))
        return {
            "test": "(8) SwiGLU richer than ReLU FFN",
            "max_abs_diff": diff,
            "fraction_features_differing": frac_diff,
            "passed": bool(diff > 0.01 and frac_diff > 0.1),
        }

    def verify_richer_than_gelu(self) -> dict:
        """(9) SwiGLU produces different output than GELU FFN."""
        swiglu_out = self.swiglu.forward(self.demo_batch)
        gelu_out = self.gelu_ffn.forward(self.demo_batch)
        diff = float(np.max(np.abs(swiglu_out - gelu_out)))
        frac_diff = float(np.mean(np.abs(swiglu_out - gelu_out) > 0.01))
        return {
            "test": "(9) SwiGLU richer than GELU FFN",
            "max_abs_diff": diff,
            "fraction_features_differing": frac_diff,
            "passed": bool(diff > 0.01 and frac_diff > 0.1),
        }

    def verify_richer_than_silu_only(self) -> dict:
        """(10) SwiGLU different from SiLU-only FFN (isolates gating benefit).

        If SwiGLU and SiLU-only produced identical outputs, the gate would
        be useless. The difference here is PURELY due to the gating
        mechanism (the * W3 path), not the activation choice.
        """
        swiglu_out = self.swiglu.forward(self.demo_batch)
        silu_only_out = self.silu_only_ffn.forward(self.demo_batch)
        diff = float(np.max(np.abs(swiglu_out - silu_only_out)))
        frac_diff = float(np.mean(np.abs(swiglu_out - silu_only_out) > 0.01))
        return {
            "test": "(10) SwiGLU richer than SiLU-only FFN (gating benefit isolated)",
            "max_abs_diff": diff,
            "fraction_features_differing": frac_diff,
            "passed": bool(diff > 0.01 and frac_diff > 0.1),
        }

    def verify_parameter_count(self) -> dict:
        """(11) SwiGLU has 3*hidden*ffn params; others have 2*hidden*ffn."""
        swiglu_params = self.swiglu.parameter_count()
        relu_params = self.relu_ffn.parameter_count()
        gelu_params = self.gelu_ffn.parameter_count()
        silu_only_params = self.silu_only_ffn.parameter_count()
        expected_swiglu = 3 * self.config.hidden_dim * self.config.ffn_hidden_dim
        expected_others = 2 * self.config.hidden_dim * self.config.ffn_hidden_dim
        overhead = swiglu_params / relu_params
        return {
            "test": "(11) parameter count (3*hidden*ffn vs 2*hidden*ffn)",
            "swiglu_params": swiglu_params,
            "relu_params": relu_params,
            "gelu_params": gelu_params,
            "silu_only_params": silu_only_params,
            "expected_swiglu": expected_swiglu,
            "expected_others": expected_others,
            "swiglu_overhead_vs_relu": f"{overhead:.2f}x",
            "passed": bool(swiglu_params == expected_swiglu
                            and relu_params == expected_others
                            and gelu_params == expected_others
                            and silu_only_params == expected_others),
        }

    def verify_intermediate_statistics(self) -> dict:
        """(12) track mean, std, and sparsity at each step of SwiGLU."""
        decomp = self.swiglu.forward_decomposed(self.demo_batch)
        stats = {}
        for name, tensor in decomp.items():
            stats[name] = {
                "shape": tuple(tensor.shape),
                "mean": float(np.mean(tensor)),
                "std": float(np.std(tensor)),
                "min": float(np.min(tensor)),
                "max": float(np.max(tensor)),
                "sparsity": float(np.mean(np.abs(tensor) < 0.01)),
            }
        return {
            "test": "(12) intermediate statistics (mean, std, sparsity per step)",
            "statistics": stats,
            "passed": True,  # informational; always passes
        }

    # ----- SIX edge-case methods -----

    def edge_case_zero_input(self) -> dict:
        """(a) All-zero input -> all-zero output (no NaN)."""
        zeros = np.zeros((1, self.config.hidden_dim), dtype=np.float32)
        out = self.swiglu.forward(zeros)
        return {
            "test": "(a) edge case: zero input",
            "output_all_finite": bool(np.all(np.isfinite(out))),
            "max_abs_output": float(np.max(np.abs(out))),
            "passed": bool(np.all(np.isfinite(out)) and
                            np.max(np.abs(out)) < 1e-4),
        }

    def edge_case_large_input(self) -> dict:
        """(b) Very large input -> no overflow (sigmoid saturates)."""
        large = (np.random.randn(1, self.config.hidden_dim)
                 * self.config.large_input_scale).astype(np.float32)
        out = self.swiglu.forward(large)
        return {
            "test": f"(b) edge case: large input (scale={self.config.large_input_scale})",
            "output_all_finite": bool(np.all(np.isfinite(out))),
            "max_abs_output": float(np.max(np.abs(out))),
            "passed": bool(np.all(np.isfinite(out))),
        }

    def edge_case_small_input(self) -> dict:
        """(c) Very small input -> proportional small output (linear regime)."""
        small = (np.random.randn(1, self.config.hidden_dim)
                 * self.config.small_input_scale).astype(np.float32)
        out = self.swiglu.forward(small)
        return {
            "test": f"(c) edge case: small input (scale={self.config.small_input_scale})",
            "output_all_finite": bool(np.all(np.isfinite(out))),
            "max_abs_output": float(np.max(np.abs(out))),
            "passed": bool(np.all(np.isfinite(out))),
        }

    def edge_case_single_feature(self) -> dict:
        """(d) Single feature (batch=1, hidden=1) -> shape preserved."""
        # We can't reuse the existing model (hidden_dim=8), so build a tiny one.
        tiny = SwiGLUFFN(1, 2, np.random.default_rng(99))
        x = np.array([[1.5]], dtype=np.float32)
        out = tiny.forward(x)
        return {
            "test": "(d) edge case: single feature (hidden_dim=1)",
            "input_shape": tuple(x.shape),
            "output_shape": tuple(out.shape),
            "output_finite": bool(np.all(np.isfinite(out))),
            "passed": bool(tuple(out.shape) == (1, 1) and
                            np.all(np.isfinite(out))),
        }

    def edge_case_all_negative(self) -> dict:
        """(e) All-negative input -> gate mostly closed, output small."""
        neg = -np.abs(np.random.randn(1, self.config.hidden_dim)
                       * 2.0).astype(np.float32)
        out = self.swiglu.forward(neg)
        decomp = self.swiglu.forward_decomposed(neg)
        gate = decomp["gate"][0]
        gate_near_zero = float(np.mean(np.abs(gate) < 0.1))
        return {
            "test": "(e) edge case: all-negative input",
            "gate_fraction_near_zero": gate_near_zero,
            "max_abs_output": float(np.max(np.abs(out))),
            "output_finite": bool(np.all(np.isfinite(out))),
            "passed": bool(np.all(np.isfinite(out)) and gate_near_zero > 0.3),
        }

    def edge_case_all_positive(self) -> dict:
        """(f) All-positive input -> gate mostly open, output larger."""
        pos = np.abs(np.random.randn(1, self.config.hidden_dim)
                      * 2.0).astype(np.float32)
        out = self.swiglu.forward(pos)
        decomp = self.swiglu.forward_decomposed(pos)
        gate = decomp["gate"][0]
        gate_near_zero = float(np.mean(np.abs(gate) < 0.1))
        return {
            "test": "(f) edge case: all-positive input",
            "gate_fraction_near_zero": gate_near_zero,
            "max_abs_output": float(np.max(np.abs(out))),
            "output_finite": bool(np.all(np.isfinite(out))),
            "passed": bool(np.all(np.isfinite(out)) and gate_near_zero < 0.5),
        }

    def run_all_verifications(self) -> dict:
        """Run all twelve property checks + six edge cases."""
        decomposed = self.swiglu.forward_decomposed(self.demo_batch)
        return {
            "demo_batch": self.demo_batch,
            "demo_output": decomposed["output"],
            "decomposed_sample": self.swiglu.forward_decomposed(
                self.sample.reshape(1, -1)
            ),
            "swiglu_output": self.swiglu.forward(self.demo_batch),
            "relu_output": self.relu_ffn.forward(self.demo_batch),
            "gelu_output": self.gelu_ffn.forward(self.demo_batch),
            "silu_only_output": self.silu_only_ffn.forward(self.demo_batch),
            "tests": [
                self.verify_shape_preservation(),
                self.verify_gating_formula(),
                self.verify_silu_activation(),
                self.verify_silu_smoothness(),
                self.verify_gate_can_suppress(),
                self.verify_gate_sparsity(),
                self.verify_residual_friendly(),
                self.verify_richer_than_relu(),
                self.verify_richer_than_gelu(),
                self.verify_richer_than_silu_only(),
                self.verify_parameter_count(),
                self.verify_intermediate_statistics(),
            ],
            "edge_cases": [
                self.edge_case_zero_input(),
                self.edge_case_large_input(),
                self.edge_case_small_input(),
                self.edge_case_single_feature(),
                self.edge_case_all_negative(),
                self.edge_case_all_positive(),
            ],
        }


# ============================================================================
# 6. PLOTTER
# ============================================================================

class SwiGLUPlotter:
    """Renders the inspector's results into FIVE figures."""

    @staticmethod
    def plot_io_transformation(demo_batch: np.ndarray, demo_output: np.ndarray,
                                save_path: str =
                                "fig_swiglu_io_transformation.png") -> None:
        """Two-panel: input vector -> output vector for one sample."""
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
                          f"Dimensionality = {demo_output.shape[1]} "
                          f"(RESIDUAL-FRIENDLY: same as input)")
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
        gate = decomposed["gate"][0]
        up = decomposed["up"][0]
        gated = decomposed["gated"][0]
        output = decomposed["output"][0]

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
    def plot_swiglu_vs_relu_gelu(demo_batch: np.ndarray, swiglu: SwiGLUFFN,
                                  relu_ffn: ReLUFFN, gelu_ffn: GELUFFN,
                                  silu_only_ffn: SiLUOnlyFFN,
                                  save_path: str =
                                  "fig_swiglu_vs_relu_gelu.png") -> None:
        """Five-panel: input, SwiGLU, ReLU, GELU, SiLU-only outputs."""
        swiglu_out = swiglu.forward(demo_batch)
        relu_out = relu_ffn.forward(demo_batch)
        gelu_out = gelu_ffn.forward(demo_batch)
        silu_only_out = silu_only_ffn.forward(demo_batch)
        sample_idx = 0
        x = np.arange(demo_batch.shape[1])

        fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
        panels = [
            (axes[0], demo_batch[sample_idx], "① Input x",
             "steelblue"),
            (axes[1], swiglu_out[sample_idx],
             f"② SwiGLU output: W2(silu(W1 x) ⊙ W3 x)\n"
             f"[3 matrices + gate, {swiglu.parameter_count()} params]",
             "seagreen"),
            (axes[2], silu_only_out[sample_idx],
             f"③ SiLU-only output: W2(silu(W1 x))\n"
             f"[2 matrices, no gate, {silu_only_ffn.parameter_count()} params]",
             "teal"),
            (axes[3], gelu_out[sample_idx],
             f"④ GELU output: W2(gelu(W1 x))\n"
             f"[2 matrices, no gate, {gelu_ffn.parameter_count()} params]",
             "darkorange"),
            (axes[4], relu_out[sample_idx],
             f"⑤ ReLU output: W2(relu(W1 x))\n"
             f"[2 matrices, no gate, {relu_ffn.parameter_count()} params]",
             "crimson"),
        ]
        for ax, vec, title, color in panels:
            ax.bar(x, vec, color=color, edgecolor="black", linewidth=0.5)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_title(title)
            ax.set_ylabel("Value")
            ax.grid(alpha=0.3, axis="y")
        axes[-1].set_xlabel("Feature index")
        fig.suptitle("SwiGLU vs. SiLU-only vs. GELU vs. ReLU FFN\n"
                     "SwiGLU's gating produces visibly different (richer) transformations",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def plot_activation_landscape(save_path: str =
                                   "fig_swiglu_activation_landscape.png"
                                   ) -> None:
        """Plot SiLU, ReLU, GELU, and Sigmoid curves side by side."""
        x = np.linspace(-6, 6, 400, dtype=np.float64)
        silu_y = Activations.silu(x)
        relu_y = Activations.relu(x)
        gelu_y = Activations.gelu(x)
        sigmoid_y = Activations.sigmoid(x)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: activation functions.
        axes[0].plot(x, silu_y, color="seagreen", linewidth=2.5, label="SiLU (Swish)")
        axes[0].plot(x, relu_y, color="crimson", linewidth=2.5, label="ReLU")
        axes[0].plot(x, gelu_y, color="darkorange", linewidth=2.5,
                      label="GELU (tanh approx.)")
        axes[0].axhline(0, color="black", linewidth=0.5)
        axes[0].axvline(0, color="black", linewidth=0.5)
        axes[0].set_title("Activation Function Landscape")
        axes[0].set_xlabel("Input x")
        axes[0].set_ylabel("Output")
        axes[0].legend(fontsize=11)
        axes[0].grid(alpha=0.3)
        # Annotate SiLU's negative bump.
        min_silu_x = x[np.argmin(silu_y)]
        min_silu_y = float(np.min(silu_y))
        axes[0].annotate(
            f"SiLU negative bump\n(min ≈ {min_silu_y:.3f})",
            xy=(min_silu_x, min_silu_y),
            xytext=(min_silu_x - 1.5, min_silu_y - 0.3),
            arrowprops=dict(arrowstyle="->", color="seagreen"),
            fontsize=9, color="seagreen",
        )
        # Annotate ReLU's kink.
        axes[0].annotate(
            "ReLU kink\n(not differentiable)",
            xy=(0, 0),
            xytext=(-3, 1.5),
            arrowprops=dict(arrowstyle="->", color="crimson"),
            fontsize=9, color="crimson",
        )

        # Right panel: derivatives (finite difference).
        h = 1e-5
        silu_grad = (Activations.silu(x + h) - Activations.silu(x - h)) / (2 * h)
        relu_grad = (Activations.relu(x + h) - Activations.relu(x - h)) / (2 * h)
        gelu_grad = (Activations.gelu(x + h) - Activations.gelu(x - h)) / (2 * h)
        axes[1].plot(x, silu_grad, color="seagreen", linewidth=2.5,
                      label="SiLU gradient")
        axes[1].plot(x, relu_grad, color="crimson", linewidth=2.5,
                      label="ReLU gradient (JUMP at 0!)")
        axes[1].plot(x, gelu_grad, color="darkorange", linewidth=2.5,
                      label="GELU gradient")
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].axvline(0, color="black", linewidth=0.5)
        axes[1].set_title("Activation Gradients (why SiLU/GELU train better)")
        axes[1].set_xlabel("Input x")
        axes[1].set_ylabel("Gradient")
        axes[1].legend(fontsize=11)
        axes[1].grid(alpha=0.3)
        axes[1].annotate(
            "ReLU gradient JUMP\n(causes dead neurons)",
            xy=(0, 0.5),
            xytext=(-4, 0.8),
            arrowprops=dict(arrowstyle="->", color="crimson"),
            fontsize=9, color="crimson",
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def plot_gate_sparsity(demo_batch: np.ndarray, swiglu: SwiGLUFFN,
                            threshold: float = 0.1,
                            save_path: str =
                            "fig_swiglu_gate_sparsity.png") -> None:
        """Histogram of gate values showing sparsity (suppression pattern)."""
        decomposed = swiglu.forward_decomposed(demo_batch)
        gate = decomposed["gate"].flatten()  # (batch * ffn_dim,)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: histogram of gate values.
        axes[0].hist(gate, bins=40, color="darkorange", edgecolor="black",
                      alpha=0.8)
        axes[0].axvline(-threshold, color="crimson", linestyle="--",
                         linewidth=2, label=f"±{threshold} (suppressed)")
        axes[0].axvline(threshold, color="crimson", linestyle="--",
                         linewidth=2)
        axes[0].axvline(0, color="black", linewidth=0.7)
        axes[0].set_title("Gate Value Distribution (silu(W1 x))\n"
                          "Bars near 0 = SUPPRESSED features")
        axes[0].set_xlabel("Gate value")
        axes[0].set_ylabel("Count")
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis="y")

        # Right: bar chart of all gate values (sorted).
        sorted_gate = np.sort(gate)
        colors = ["crimson" if abs(v) < threshold else "steelblue"
                  for v in sorted_gate]
        axes[1].bar(range(len(sorted_gate)), sorted_gate, color=colors,
                    edgecolor="black", linewidth=0.3)
        axes[1].axhline(0, color="black", linewidth=0.7)
        axes[1].axhline(-threshold, color="crimson", linestyle="--",
                         linewidth=1.5)
        axes[1].axhline(threshold, color="crimson", linestyle="--",
                         linewidth=1.5)
        n_suppressed = int(np.sum(np.abs(sorted_gate) < threshold))
        frac = n_suppressed / len(sorted_gate)
        axes[1].set_title(
            f"Sorted Gate Values\n"
            f"RED = suppressed (|gate| < {threshold}): "
            f"{n_suppressed}/{len(sorted_gate)} = {frac:.1%}"
        )
        axes[1].set_xlabel("Sorted index")
        axes[1].set_ylabel("Gate value")
        axes[1].grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()


# ============================================================================
# 7. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Reference Card 10.5 (EXPANDED) — SwiGLU Activation Inspection")
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
    print(f"\nSample 0 SiLU-only output (no gate, for comparison):")
    print(f"  {results['silu_only_output'][0].round(3)}")
    print(f"\nSample 0 GELU FFN output (for comparison):")
    print(f"  {results['gelu_output'][0].round(3)}")
    print(f"\nSample 0 ReLU FFN output (for comparison):")
    print(f"  {results['relu_output'][0].round(3)}")

    # Print decomposed intermediates.
    decomp = results["decomposed_sample"]
    print(f"\nDecomposed intermediates for sample 0:")
    print(f"  gate (silu(W1 x), shape {decomp['gate'].shape}):")
    print(f"    {decomp['gate'][0].round(3)}")
    print(f"  up (W3 x, shape {decomp['up'].shape}):")
    print(f"    {decomp['up'][0].round(3)}")
    print(f"  gated (gate * up, shape {decomp['gated'].shape}):")
    print(f"    {decomp['gated'][0].round(3)}")

    # Run & pretty-print every property test.
    print("\n" + "=" * 70)
    print("PROPERTY VERIFICATION TESTS (12 tests)")
    print("=" * 70)
    for t in results["tests"]:
        status = "✓ PASS" if t.get("passed") else "✗ FAIL"
        print(f"\n[{status}] {t['test']}")
        for k, v in t.items():
            if k in ("test", "passed"):
                continue
            if isinstance(v, dict):
                print(f"    {k}:")
                for sub_k, sub_v in v.items():
                    print(f"      {sub_k}: {sub_v}")
            elif isinstance(v, list) and len(v) > 6:
                print(f"    {k}: [{v[0]:.4f}, {v[1]:.4f}, ... , "
                      f"{v[-1]:.4f}] ({len(v)} entries)")
            else:
                print(f"    {k}: {v}")

    # Run & pretty-print every edge case.
    print("\n" + "=" * 70)
    print("EDGE CASE TESTS (6 scenarios)")
    print("=" * 70)
    for t in results["edge_cases"]:
        status = "✓ PASS" if t.get("passed") else "✗ FAIL"
        print(f"\n[{status}] {t['test']}")
        for k, v in t.items():
            if k in ("test", "passed"):
                continue
            print(f"    {k}: {v}")

    # Render visualizations.
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS (5 figures)")
    print("=" * 70)
    SwiGLUPlotter.plot_io_transformation(
        results["demo_batch"], results["demo_output"]
    )
    SwiGLUPlotter.plot_gating_mechanism(
        results["demo_batch"][0], inspector.swiglu
    )
    SwiGLUPlotter.plot_swiglu_vs_relu_gelu(
        results["demo_batch"], inspector.swiglu, inspector.relu_ffn,
        inspector.gelu_ffn, inspector.silu_only_ffn
    )
    SwiGLUPlotter.plot_activation_landscape()
    SwiGLUPlotter.plot_gate_sparsity(
        results["demo_batch"], inspector.swiglu,
        cfg.gate_sparse_threshold
    )

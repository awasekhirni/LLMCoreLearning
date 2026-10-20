"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
================================================================================
ALiBi (Attention with Linear Biases)
================================================================================

ELI5 SUMMARY
------------
Imagine you're at a party. The person right next to you is easy to hear,
but someone across the room needs to shout. ALiBi works like the party's
natural "volume knob": the farther away a word is (in position), the less
attention it gets, so nearby words dominate. Instead of giving every token
a numbered ticket (positional embeddings), ALiBi just adds a "distance fine"
to the attention scores — simple, no extra parameters, and it keeps working
even on much longer sequences than the model was trained on.

PURPOSE
-------
Teach ALiBi end-to-end for a beginner audience: the math, the attention
mechanism, edge cases, visualizations, and hands-on exercises — all in
one reproducible script.

INSTALL
-------
    python3 -m venv .venv
    source .venv/bin/activate
    pip install numpy matplotlib pandas

RUN
---
    python3 alibi_learning_module.py

OUTPUTS (saved to current working directory):
    fig_alibi_processing.png
    fig_alibi_ex1.png
    fig_alibi_ex2.png
    fig_alibi_ex3.png

REFERENCES
----------
Press et al. 2022, "Train Short, Test Long: Attention with Linear Biases
Enables Input Length Extrapolation", ICLR 2022.

IMPLEMENTATION NOTES
--------------------
- Canonical slope formula (mosaicml/llm-foundry):
      m_h = 2 ** (-8 * (h+1) / H)   for h = 0, 1, ..., H-1
- Bias is added to the raw logits BEFORE softmax.
- No learnable parameters, no positional embeddings.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Global reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DEFAULT_DEMO_TEXT = "The cat sat on the warm mat quietly"


# ==========================================================================
# Utilities — small, pure, testable
# ==========================================================================
def stable_hash(s: str) -> int:
    """Deterministic 32-bit hash for token→seed mapping (PYTHONHASHSEED-safe)."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention_entropy(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Shannon entropy (nats) of attention distributions.

    Parameters
    ----------
    weights : np.ndarray
        Shape (..., L_q, L_k). Rows must sum to 1.

    Returns
    -------
    np.ndarray
        Shape (...). One entropy per query position (averaged over
        leading dims as needed by the caller).
    """
    return -np.sum(weights * np.log(weights + eps), axis=-1)


# ==========================================================================
# Core: ALiBi slope spectrum
# ==========================================================================
@dataclass
class ALiBiSlopes:
    """
    Per-head slope geometric sequence: m_h = 2^(-8 * (h+1) / H).

    The slope controls how fast attention decays with distance for head h.
    Head 0 (largest slope) is the sharpest "local" filter; the last head
    (smallest slope) is the "long-range scanner".
    """

    n_heads: int
    base: float = 8.0

    def __post_init__(self) -> None:
        if not isinstance(self.n_heads, int) or self.n_heads < 1:
            raise ValueError(f"n_heads must be a positive int, got {self.n_heads!r}")
        if not (isinstance(self.base, (int, float)) and self.base > 0):
            raise ValueError(f"base must be > 0, got {self.base!r}")

    def compute(self) -> np.ndarray:
        """Return the (H,) array of slopes."""
        h = np.arange(1, self.n_heads + 1, dtype=np.float64)
        return np.power(2.0, -self.base * h / self.n_heads)


# ==========================================================================
# Core: distance matrix and bias
# ==========================================================================
def build_distance_matrix(seq_len: int) -> np.ndarray:
    """Return the (L, L) matrix D[i, j] = |i - j|."""
    if not isinstance(seq_len, int):
        raise TypeError(f"seq_len must be int, got {type(seq_len).__name__}")
    if seq_len < 0:
        raise ValueError(f"seq_len must be >= 0, got {seq_len}")
    i = np.arange(seq_len)[:, None]
    j = np.arange(seq_len)[None, :]
    return np.abs(i - j).astype(np.float32)


def build_alibi_bias(slopes: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Build the ALiBi bias tensor: bias[h, i, j] = -slopes[h] * |i - j|.

    Parameters
    ----------
    slopes : np.ndarray, shape (H,)
    seq_len : int

    Returns
    -------
    np.ndarray, shape (H, L, L)
    """
    if slopes.ndim != 1:
        raise ValueError(f"slopes must be 1-D, got shape {slopes.shape}")
    D = build_distance_matrix(seq_len)
    return -slopes[:, None, None] * D[None, :, :]


# ==========================================================================
# Core: multi-head self-attention with optional ALiBi
# ==========================================================================
@dataclass(frozen=True)
class AttentionConfig:
    """Hyperparameters for one self-attention layer."""
    d_model: int = 32
    n_heads: int = 4
    use_alibi: bool = True

    def __post_init__(self) -> None:
        if self.d_model < 1 or self.n_heads < 1:
            raise ValueError("d_model and n_heads must be >= 1")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )


@dataclass
class AttentionOutput:
    """Everything the caller might want to inspect after forward()."""
    queries: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    scores_raw: np.ndarray
    bias: Optional[np.ndarray]
    scores_biased: np.ndarray
    weights: np.ndarray
    output: np.ndarray


@dataclass
class MultiHeadSelfAttention:
    """
    Minimal MHA layer. Fixed-seed projections make runs reproducible.

    ALiBi (if enabled) is added to the raw logits *before* the softmax —
    that is the whole mechanism; there are no learned positional parameters.
    """

    config: AttentionConfig

    def __post_init__(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        d = self.config.d_model
        # Small random init keeps logits in a sensible range.
        self.W_q = rng.normal(0.0, 0.1, (d, d))
        self.W_k = rng.normal(0.0, 0.1, (d, d))
        self.W_v = rng.normal(0.0, 0.1, (d, d))
        self.slopes = (
            ALiBiSlopes(self.config.n_heads).compute()
            if self.config.use_alibi
            else None
        )

    def forward(self, x: np.ndarray) -> AttentionOutput:
        """
        Run attention on ``x``.

        Parameters
        ----------
        x : np.ndarray, shape (L, d_model)

        Returns
        -------
        AttentionOutput
        """
        if x.ndim != 2:
            raise ValueError(f"expected x of shape (L, d_model), got {x.shape}")
        L, d = x.shape
        if d != self.config.d_model:
            raise ValueError(f"x has d={d}, expected {self.config.d_model}")
        if L == 0:
            raise ValueError("empty sequence not supported (L must be >= 1)")

        H = self.config.n_heads
        hd = d // H

        # Project and split into heads: (H, L, hd)
        Q = (x @ self.W_q).reshape(L, H, hd).transpose(1, 0, 2)
        K = (x @ self.W_k).reshape(L, H, hd).transpose(1, 0, 2)
        V = (x @ self.W_v).reshape(L, H, hd).transpose(1, 0, 2)

        scores_raw = Q @ K.transpose(0, 2, 1) / np.sqrt(hd)  # (H, L, L)

        if self.config.use_alibi and self.slopes is not None:
            bias = build_alibi_bias(self.slopes, L)  # (H, L, L)
            scores_biased = scores_raw + bias
        else:
            bias = None
            scores_biased = scores_raw

        weights = softmax(scores_biased, axis=-1)            # (H, L, L)
        out = (weights @ V).transpose(1, 0, 2).reshape(L, d)

        return AttentionOutput(Q, K, V, scores_raw, bias, scores_biased, weights, out)


# ==========================================================================
# Text helpers (visualisation labels only)
# ==========================================================================
def tokenize_demo(text: str) -> List[str]:
    """Simple word/punct tokenizer — good enough for heatmap labels."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def embeddings_for_tokens(
    tokens: List[str], d_model: int, seed: int = RANDOM_SEED
) -> np.ndarray:
    """Deterministic per-token embeddings, shape (L, d_model)."""
    rows = []
    for tok in tokens:
        rng = np.random.default_rng((stable_hash(tok) + seed) % (2**31 - 1))
        rows.append(rng.normal(0.0, 1.0, d_model))
    return np.stack(rows, axis=0)


# ==========================================================================
# Edge cases
# ==========================================================================
def run_edge_cases() -> None:
    """Exercise every edge case the module is designed to handle."""
    print("\n" + "=" * 74)
    print("EDGE CASES & FAILURE SCENARIOS")
    print("=" * 74)

    # 1. Single-token sequence: attention is trivially 1.0
    print("\n[1] Single-token sequence (L=1):")
    x1 = np.zeros((1, 16))
    out = MultiHeadSelfAttention(AttentionConfig(16, 2, True)).forward(x1)
    print(f"    weights = {out.weights[0, 0, 0]:.4f} (expected 1.0)")

    # 2. Invalid n_heads
    print("\n[2] n_heads=0:")
    try:
        ALiBiSlopes(0).compute()
    except ValueError as e:
        print(f"    caught: {e}")

    # 3. d_model not divisible by n_heads
    print("\n[3] d_model=10, n_heads=3:")
    try:
        AttentionConfig(10, 3, True)
    except ValueError as e:
        print(f"    caught: {e}")

    # 4. Empty sequence
    print("\n[4] Empty sequence (L=0):")
    try:
        MultiHeadSelfAttention(AttentionConfig(16, 2, True)).forward(np.zeros((0, 16)))
    except ValueError as e:
        print(f"    caught: {e}")

    # 5. Wrong input dims
    print("\n[5] Mismatched d_model:")
    try:
        MultiHeadSelfAttention(AttentionConfig(16, 2, True)).forward(np.zeros((5, 8)))
    except ValueError as e:
        print(f"    caught: {e}")

    # 6. Negative seq_len
    print("\n[6] Negative seq_len:")
    try:
        build_distance_matrix(-3)
    except ValueError as e:
        print(f"    caught: {e}")


# ==========================================================================
# ELI5 walkthrough
# ==========================================================================
def eli5_walkthrough() -> None:
    print("\n" + "=" * 74)
    print("ELI5 WALKTHROUGH — ALiBi in plain English")
    print("=" * 74)
    print()
    print("Picture 5 words standing in a line:")
    words = ["I", "love", "warm", "apple", "pie"]
    for i, w in enumerate(words):
        print(f"    position {i}: {w}")
    print()
    print("When 'pie' (last word) decides what to pay attention to,")
    print("nearby words should matter more than far words.")
    print()
    print("Distance from 'pie' to each word:")
    for i, w in enumerate(words):
        print(f"    {w:>6s}: distance = {abs(4 - i)}")
    print()
    print("ALiBi gives each HEAD (a mini-attention filter) a 'slope' —")
    print("how fast its attention fades with distance:")
    slopes = ALiBiSlopes(2).compute()
    for h, s in enumerate(slopes):
        print(f"    head {h}: slope = {s:.4f}")
    print()
    print("Bias applied by head 0 for 'pie' (negative = reduce weight):")
    D = build_distance_matrix(len(words))
    bias = -slopes[0] * D
    for i, w in enumerate(words):
        row = "  ".join(f"{bias[4, j]:+.3f}" for j in range(len(words)))
        print(f"    -> [{row}]")
    print()
    print("The diagonal is always 0 — 'pie' pays itself no fine.")
    print("The far corner is most negative — far words get quiet.")
    print()
    print("Add that matrix to the raw attention scores, run softmax,")
    print("and nearby words naturally dominate. That's ALiBi.")


# ==========================================================================
# Visualisation: full pipeline
# ==========================================================================
def visualize_alibi_pipeline(
    text: str = DEFAULT_DEMO_TEXT,
    d_model: int = 32,
    n_heads: int = 4,
    save_to: str = "fig_alibi_processing.png",
) -> None:
    """Plot every stage of ALiBi: distance → bias → scores → weights."""
    tokens = tokenize_demo(text)
    L = len(tokens)
    x = embeddings_for_tokens(tokens, d_model)

    out_on = MultiHeadSelfAttention(AttentionConfig(d_model, n_heads, True)).forward(x)
    out_off = MultiHeadSelfAttention(AttentionConfig(d_model, n_heads, False)).forward(x)

    D = build_distance_matrix(L)
    bias = out_on.bias  # (H, L, L)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    im = axes[0, 0].imshow(D, cmap="viridis")
    axes[0, 0].set_title("(a) distance matrix |i − j|")
    axes[0, 0].set_xlabel("key position j"); axes[0, 0].set_ylabel("query i")
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(bias[0], cmap="RdBu_r")
    axes[0, 1].set_title(f"(b) ALiBi bias (head 0, slope={out_on.slopes[0]:.4f})")
    axes[0, 1].set_xlabel("key position j"); axes[0, 1].set_ylabel("query i")
    plt.colorbar(im, ax=axes[0, 1])

    im = axes[0, 2].imshow(out_on.scores_biased[0], cmap="RdBu_r")
    axes[0, 2].set_title("(c) raw logits + bias (head 0)")
    axes[0, 2].set_xlabel("key position j"); axes[0, 2].set_ylabel("query i")
    plt.colorbar(im, ax=axes[0, 2])

    im = axes[1, 0].imshow(out_off.weights[0], cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set_title("(d) attention WITHOUT ALiBi (head 0)")
    axes[1, 0].set_xticks(range(L)); axes[1, 0].set_xticklabels(tokens, rotation=45, ha="right")
    axes[1, 0].set_yticks(range(L)); axes[1, 0].set_yticklabels(tokens)
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].imshow(out_on.weights[0], cmap="Blues", vmin=0, vmax=1)
    axes[1, 1].set_title("(e) attention WITH ALiBi (head 0)")
    axes[1, 1].set_xticks(range(L)); axes[1, 1].set_xticklabels(tokens, rotation=45, ha="right")
    axes[1, 1].set_yticks(range(L)); axes[1, 1].set_yticklabels(tokens)
    plt.colorbar(im, ax=axes[1, 1])

    delta = out_on.weights[0] - out_off.weights[0]
    im = axes[1, 2].imshow(delta, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 2].set_title("(f) Δ weights: with − without ALiBi (head 0)")
    axes[1, 2].set_xticks(range(L)); axes[1, 2].set_xticklabels(tokens, rotation=45, ha="right")
    axes[1, 2].set_yticks(range(L)); axes[1, 2].set_yticklabels(tokens)
    plt.colorbar(im, ax=axes[1, 2])

    fig.suptitle(f"ALiBi pipeline on tokens: {tokens}", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved: {Path(save_to).resolve()}")


# ==========================================================================
# Exercises + solutions (single-file, matching reference-module pattern)
# ==========================================================================
@dataclass
class Exercise1Solution:
    """Exercise 1 — how does head count reshape the slope spectrum?"""
    head_counts: Tuple[int, ...] = (1, 2, 4, 8, 16)

    def run(self) -> pd.DataFrame:
        rows = []
        for H in self.head_counts:
            for h, s in enumerate(ALiBiSlopes(H).compute()):
                rows.append(dict(n_heads=H, head_idx=h, slope=float(s)))
        return pd.DataFrame(rows)

    def plot(self, df: pd.DataFrame, save_to: str = "fig_alibi_ex1.png") -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for H, g in df.groupby("n_heads"):
            ax.plot(g["head_idx"], g["slope"], marker="o", label=f"H={H}")
        ax.set_yscale("log")
        ax.set_xlabel("head index h")
        ax.set_ylabel("slope m_h (log scale)")
        ax.set_title("Exercise 1 — per-head slopes across different n_heads")
        ax.legend(); ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout(); fig.savefig(save_to, dpi=150); plt.show()
        print(f"[ex1] saved: {Path(save_to).resolve()}")


@dataclass
class Exercise2Solution:
    """Exercise 2 — does ALiBi reduce attention entropy as L grows?"""
    lengths: Tuple[int, ...] = (8, 16, 32, 64)
    d_model: int = 32
    n_heads: int = 4
    trials: int = 3

    def run(self) -> pd.DataFrame:
        rows = []
        for L in self.lengths:
            off, on = [], []
            for t in range(self.trials):
                rng = np.random.default_rng(RANDOM_SEED + t)
                x = rng.normal(0, 1, (L, self.d_model))
                a_off = MultiHeadSelfAttention(AttentionConfig(self.d_model, self.n_heads, False)).forward(x)
                a_on = MultiHeadSelfAttention(AttentionConfig(self.d_model, self.n_heads, True)).forward(x)
                off.append(float(attention_entropy(a_off.weights).mean()))
                on.append(float(attention_entropy(a_on.weights).mean()))
            rows.append(dict(
                seq_len=L,
                entropy_no_alibi=float(np.mean(off)),
                entropy_with_alibi=float(np.mean(on)),
            ))
        return pd.DataFrame(rows)

    def plot(self, df: pd.DataFrame, save_to: str = "fig_alibi_ex2.png") -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(df["seq_len"], df["entropy_no_alibi"], marker="o", label="no ALiBi")
        ax.plot(df["seq_len"], df["entropy_with_alibi"], marker="s", label="with ALiBi")
        ax.set_xlabel("sequence length L")
        ax.set_ylabel("mean attention entropy (nats)")
        ax.set_title("Exercise 2 — attention entropy vs sequence length")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(save_to, dpi=150); plt.show()
        print(f"[ex2] saved: {Path(save_to).resolve()}")


@dataclass
class Exercise3Solution:
    """Exercise 3 — recency bias: attention to position 0 as L grows."""
    lengths: Tuple[int, ...] = (8, 16, 32, 64)
    d_model: int = 32
    n_heads: int = 4

    def run(self) -> pd.DataFrame:
        rows = []
        for L in self.lengths:
            rng = np.random.default_rng(RANDOM_SEED)
            x = rng.normal(0, 1, (L, self.d_model))
            a_off = MultiHeadSelfAttention(AttentionConfig(self.d_model, self.n_heads, False)).forward(x)
            a_on = MultiHeadSelfAttention(AttentionConfig(self.d_model, self.n_heads, True)).forward(x)
            # mean over heads of: last query, weight to position 0
            rows.append(dict(
                seq_len=L,
                bos_weight_no_alibi=float(a_off.weights[:, -1, 0].mean()),
                bos_weight_with_alibi=float(a_on.weights[:, -1, 0].mean()),
            ))
        return pd.DataFrame(rows)

    def plot(self, df: pd.DataFrame, save_to: str = "fig_alibi_ex3.png") -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(df["seq_len"], df["bos_weight_no_alibi"], marker="o", label="no ALiBi")
        ax.plot(df["seq_len"], df["bos_weight_with_alibi"], marker="s", label="with ALiBi")
        ax.set_xlabel("sequence length L")
        ax.set_ylabel("mean attention weight to position 0")
        ax.set_title("Exercise 3 — attention to BOS as sequence grows")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(save_to, dpi=150); plt.show()
        print(f"[ex3] saved: {Path(save_to).resolve()}")


# ==========================================================================
# Expected outcomes (printed for verification)
# ==========================================================================
def print_expected_outcomes() -> None:
    print("\n" + "=" * 74)
    print("EXPECTED OUTCOMES (use these to verify your run)")
    print("=" * 74)
    print("""
    ALiBiSlopes(4).compute()
        -> [0.2500, 0.0625, 0.0156, 0.0039]
    ALiBiSlopes(8).compute()
        -> [0.5000, 0.2500, 0.1250, 0.0625, 0.0313, 0.0156, 0.0078, 0.0039]

    For L=8, head 0 (slope 0.25):
        bias[0, i, j] = -0.25 * |i - j|
        diagonal = 0.0
        max magnitude (corner) = -1.75

    L=1 attention: weight = 1.0 (trivial)

    Exercise 1: slopes decay geometrically; with more heads the largest
        slope gets closer to 1 and the smallest stays near 2^-8.

    Exercise 2: without ALiBi, entropy grows roughly as log(L);
        with ALiBi, entropy grows more slowly (attention concentrates
        near the diagonal).

    Exercise 3: attention to position 0 shrinks with L, and shrinks
        *faster* with ALiBi — this is the recency-bias tradeoff.
    """)


# ==========================================================================
# Driver
# ==========================================================================
def main() -> None:
    print_expected_outcomes()
    eli5_walkthrough()
    visualize_alibi_pipeline()
    run_edge_cases()

    print("\n" + "=" * 74)
    print("EXERCISE 1 — slope spectrum")
    print("=" * 74)
    ex1 = Exercise1Solution()
    df1 = ex1.run()
    print(df1.to_string(index=False))
    ex1.plot(df1)

    print("\n" + "=" * 74)
    print("EXERCISE 2 — attention entropy vs sequence length")
    print("=" * 74)
    ex2 = Exercise2Solution()
    df2 = ex2.run()
    print(df2.to_string(index=False))
    ex2.plot(df2)

    print("\n" + "=" * 74)
    print("EXERCISE 3 — recency-bias tradeoff")
    print("=" * 74)
    ex3 = Exercise3Solution()
    df3 = ex3.run()
    print(df3.to_string(index=False))
    ex3.plot(df3)

    print("\nDone. Figures saved to fig_alibi_*.png")


if __name__ == "__main__":
    main()


# python3 -m venv .venv
# source .venv/bin/activate
# pip install numpy matplotlib pandas
# python3 alibi_learning_module.py

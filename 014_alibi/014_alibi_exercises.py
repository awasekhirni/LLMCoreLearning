
"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
# Exercise 1 — Slope Spectrum
# Goal: For H ∈ {1, 2, 4, 8, 16}, compute the slope spectrum and plot it on a log scale.
# Questions:
#     What is the largest slope for H = 16? What about H = 1?
#     Does the smallest slope change with H? Why or why not?
#     Which H gives the most uniform spacing of slopes on log scale?
# Hints:
#     Slope formula: m_h = 2 ** (-8 * (h+1) / H). Take log2 of both sides — what is log2(m_h)?
#     The last slope is always 2 ** (-8 * H / H) = 2 ** -8 = 0.00390625, independent of H.
#     Solution template: Exercise1Solution in Section 5.
# Exercise 2 — Entropy vs Sequence Length
# Goal: For each L ∈ {8, 16, 32, 64}, compute mean attention entropy (nats) over heads, with and without ALiBi. Plot as two lines.
# Questions:
#     Without ALiBi, entropy should grow roughly as log(L) (uniform attention). Do you observe this?
#     With ALiBi, does entropy grow as fast? Why?
#     Which head contributes most to the entropy reduction? (Look at per-head entropies.)
# Hints:
#     attention_entropy(weights).mean() gives a scalar per (L, config) — no extra averaging needed.
#     To see per-head behaviour: attention_entropy(out.weights).mean(axis=-1) (shape (H,)).
#     Solution template: Exercise2Solution in Section 5.
# Exercise 3 — Recency-Bias Tradeoff
# Goal: Measure the attention weight the last token places on position 0 (the "beginning of sequence") as L grows. Compare with and without ALiBi.
# Questions:
#     Without ALiBi, how does weight[last → pos0] scale with L? (Baseline: should be ≈ 1/L for uniform attention.)
#     With ALiBi, is it higher or lower? Why?
#     Discussion: For a task that requires attending to the first token (e.g., "extract the subject of the first sentence"), is ALiBi helping or hurting? This is the recency-bias tradeoff the paper mentions.
# Hints:
#     Access: out.weights[:, -1, 0].mean() = "last query, position 0, averaged over heads."
#     Compare with the theoretical baseline 1/L to see the divergence clearly.
#     Solution template: Exercise3Solution in Section 5.


================================================================================
ALiBi Learning Module — Exercises 1–3 Solutions
================================================================================



PURPOSE
-------
Complete, self-contained, runnable solutions to Exercises 1–3 from the
ALiBi learning module. Each exercise is implemented as a `@dataclass` so
you can instantiate, inspect, and extend it. Follows the same pattern as
the reference `bpe_exercises.py`:

    -one base class (`_ALiBiExperiment`) supplies shared RNG and plotting;
    - one subclass per exercise with `.run()` -> `pd.DataFrame` and
      `.plot(df)` -> saved figure;
    - docstrings on every class/method, inline comments explaining "why".

EXERCISES
---------
    Ex1 — Slope Spectrum Geometry
          Verify that the geometric slope formula m_h = 2^(-8(h+1)/H)
          has the properties the paper relies on: monotonically decreasing,
          log-spaced, endpoint-invariant (m_last ≈ 2^-8 for all H).
          Visualise the implied "attention horizon" per head.

    Ex2 — Length Extrapolation (train short, test long)
          Simulates the paper's headline claim. A frozen learned-PE
          table is "trained" on L_train tokens; at test time on
          2×/4×/8× longer sequences its clamping failure mode destroys
          nearby attention. ALiBi, being parametric, degrades gracefully.
          Metric: "nearby attention mass" (mass on |i−j| ≤ 2).

    Ex3 — Short-Context / Long-Range Tradeoff
          Quantifies the two cons from the technical card: (a) ALiBi
          concentrates attention on recent tokens, which is good for
          short-context tasks but (b) monotonically penalises far tokens,
          so "needle at distance D" retrieval degrades as D grows.
          Shows the crossover point vs. uniform attention.

INSTALL
-------
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install numpy matplotlib pandas

RUN
---
    python3 alibi_exercises.py

OUTPUTS (saved to current working directory)
--------------------------------------------
    fig_alibi_ex1.png
    fig_alibi_ex2.png
    fig_alibi_ex3.png

SOURCE DATASETS
---------------
Every input to these exercises is generated in-code with a fixed seed;
no network access is required. This mirrors the `bpe_exercises.py`
pattern, where synthetic data keeps the exercise focused on the concept.

    +----------------------------+-------------------------------+------------+
    | Dataset                    | Generation                    | Seed       |
    +----------------------------+-------------------------------+------------+
    | slope grid (Ex1)           | `ALiBiSlopes(H).compute()`    | n/a (exact)|
    | synthetic Q/K/V emb. (Ex2) | normal(0, 1), shape (L, d)    | 42 + offset|
    | frozen "learned" PE table  | normal(0, 1), shape (T, d)    | 42         |
    | synthetic token grid (Ex3) | normal(0, 1), shape (L, d)    | 42         |
    +----------------------------+-------------------------------+------------+

    Optional external validation (NOT used here): WikiText-103
    (open license, ~500M tokens, HF `datasets` loader, split
    `wikitext-103-raw-v1`) is the paper's own benchmark for the
    "extrapolation perplexity" metric. Reproducing it end-to-end
    requires a trained transformer, which is out of scope for this
    beginner module.

REFERENCES
----------
Press et al. 2022, "Train Short, Test Long: Attention with Linear Biases
Enables Input Length Extrapolation", ICLR 2022.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Global reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)


# ==========================================================================
# Core primitives
# --------------------------------------------------------------------------
# The module file `alibi_learning_module.py` defines these as well; we
# duplicate ~50 lines here so this file runs standalone (same design
# decision as `bpe_exercises.py`, which re-implements the small BPE core).
# ==========================================================================
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def build_distance_matrix(seq_len: int) -> np.ndarray:
    """Return D[i, j] = |i − j| as float32, shape (L, L)."""
    if seq_len < 0:
        raise ValueError(f"seq_len must be >= 0, got {seq_len}")
    i = np.arange(seq_len)[:, None]
    j = np.arange(seq_len)[None, :]
    return np.abs(i - j).astype(np.float32)


def alibi_slopes(n_heads: int, base: float = 8.0) -> np.ndarray:
    """
    Canonical ALiBi slope spectrum:

        m_h = 2 ** (-base * (h+1) / n_heads),  h = 0 .. H-1

    Head 0 has the largest slope (sharpest local filter); the last
    head has the smallest slope (longest-range scanner). All slopes
    are strictly positive and strictly decreasing.
    """
    if n_heads < 1:
        raise ValueError(f"n_heads must be >= 1, got {n_heads}")
    h = np.arange(1, n_heads + 1, dtype=np.float64)
    return np.power(2.0, -base * h / n_heads)


def alibi_bias(slopes: np.ndarray, seq_len: int) -> np.ndarray:
    """Return bias[h, i, j] = -slopes[h] · |i − j|, shape (H, L, L)."""
    D = build_distance_matrix(seq_len)
    return -slopes[:, None, None] * D[None, :, :]


# ==========================================================================
# Base experiment class — shared infra (DRY)
# ==========================================================================
@dataclass
class _ALiBiExperiment:
    """
    Common infrastructure for all three exercises.

    Subclasses implement :meth:`run` (returns a DataFrame) and
    :meth:`plot` (saves a PNG). The base class supplies:
      * `_rng(offset)` — a fresh, seeded `default_rng`;
      * `_project_qkv(x)` — deterministic Q/K/V projection;
      * `_attention(x, use_alibi, use_learned_pe, train_len)` — the
        MHA forward pass used by all exercises, parameterised by which
        positional scheme is active.
    """

    d_model: int = 32
    n_heads: int = 4
    seed: int = RANDOM_SEED

    # --- shared utilities ------------------------------------------------
    def _rng(self, offset: int = 0) -> np.random.Generator:
        """A fresh, deterministic RNG. `offset` lets us vary inputs across
        repeated trials without changing the master seed."""
        return np.random.default_rng(self.seed + offset)

    def _project_qkv(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Deterministic linear projections into heads.

        Uses hash-free RNG (``default_rng(self.seed)``) so the SAME
        projection matrices are used across every trial — that isolates
        the effect of the positional scheme, which is what the
        exercises are about.
        """
        rng = self._rng(offset=0)
        d = self.d_model
        Wq = rng.normal(0.0, 0.1, (d, d))
        Wk = rng.normal(0.0, 0.1, (d, d))
        Wv = rng.normal(0.0, 0.1, (d, d))
        H, hd = self.n_heads, d // self.n_heads
        L = x.shape[0]
        Q = (x @ Wq).reshape(L, H, hd).transpose(1, 0, 2)
        K = (x @ Wk).reshape(L, H, hd).transpose(1, 0, 2)
        V = (x @ Wv).reshape(L, H, hd).transpose(1, 0, 2)
        return Q, K, V

    def _attention(
        self,
        x: np.ndarray,
        use_alibi: bool = False,
        use_learned_pe: bool = False,
        train_len: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return attention weights of shape (H, L, L).

        Parameters
        ----------
        x : (L, d_model)
        use_alibi : bool
            Add the ALiBi linear bias to logits before softmax.
        use_learned_pe : bool
            Add a frozen learned-PE table to ``x`` before projection.
            At ``L > train_len``, positions are CLAMPED to
            ``train_len - 1`` — the standard "train short, test long"
            failure mode for learned positional embeddings.
        train_len : int, optional
            Length of the learned-PE table. Required iff
            ``use_learned_pe=True``.
        """
        if use_learned_pe:
            if train_len is None:
                raise ValueError("train_len required when use_learned_pe=True")
            rng = self._rng(offset=99)      # frozen PE init
            pe_table = rng.normal(0.0, 1.0, (train_len, self.d_model))
            pos = np.minimum(np.arange(x.shape[0]), train_len - 1)
            x = x + pe_table[pos]

        Q, K, V = self._project_qkv(x)
        hd = self.d_model // self.n_heads
        logits = Q @ K.transpose(0, 2, 1) / np.sqrt(hd)    # (H, L, L)

        if use_alibi:
            slopes = alibi_slopes(self.n_heads)
            logits = logits + alibi_bias(slopes, x.shape[0])

        return softmax(logits, axis=-1)

    # --- hooks -----------------------------------------------------------
    def run(self) -> pd.DataFrame:
        raise NotImplementedError

    def plot(self, df: pd.DataFrame, save_to: str) -> None:
        raise NotImplementedError


# ==========================================================================
# Exercise 1 — Slope Spectrum Geometry
# ==========================================================================
@dataclass
class Exercise1Solution(_ALiBiExperiment):
    """
    Exercise 1 — verify the slope-spectrum properties and visualise
    the implied per-head "attention horizons".

    Key identities to prove (see `theory_checks`):
      * slopes are strictly decreasing: m_0 > m_1 > ... > m_{H-1} > 0;
      * log2(m_h) is uniformly spaced: log2(m_{h+1}) − log2(m_h) = −8/H;
      * the last slope is invariant: m_{H-1} = 2 ** (-8) for every H;
      * the first slope equals 2 ** (-8/H), which → 1 as H grows.

    "Attention horizon" of head h — the distance at which the bias first
    pulls logit below a threshold τ (τ = −1 nats, a natural "strong
    penalty" cutoff): horizon_h = τ / m_h. Sharp heads see only a few
    tokens; the last head sees the whole training window.
    """

    head_counts: Tuple[int, ...] = (1, 2, 4, 8, 16)
    horizon_threshold_nats: float = 1.0

    # ------------------------------------------------------------------
    def theory_checks(self) -> None:
        """
        Assert the four algebraic identities the paper relies on.
        Raises AssertionError on any violation; prints 'OK' on pass.
        """
        print("[Ex1] Theory checks")
        for H in self.head_counts:
            m = alibi_slopes(H)
            # (i) strictly decreasing
            assert np.all(np.diff(m) < 0), f"slopes not strictly decreasing for H={H}"
            # (ii) uniform log-spacing
            step = -8.0 / H
            log_diff = np.diff(np.log2(m))
            assert np.allclose(log_diff, step, atol=1e-10), \
                f"log-spacing violation for H={H}"
            # (iii) endpoint invariance
            assert np.isclose(m[-1], 2 ** -8, atol=1e-12), \
                f"last slope != 2^-8 for H={H}"
            # (iv) first slope matches formula
            assert np.isclose(m[0], 2 ** (-8.0 / H), atol=1e-12), \
                f"first slope mismatch for H={H}"
            print(f"    H={H:2d}:  m = [{', '.join(f'{v:.5f}' for v in m)}]")

    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Return one row per (H, h) with slope and derived horizon."""
        rows: List[dict] = []
        for H in self.head_counts:
            for h, m in enumerate(alibi_slopes(H)):
                rows.append(dict(
                    n_heads=H,
                    head_idx=h,
                    slope=float(m),
                    log2_slope=float(np.log2(m)),
                    horizon_tokens=self.horizon_threshold_nats / float(m),
                ))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def plot(self, df: pd.DataFrame, save_to: str = "fig_alibi_ex1.png") -> None:
        """Two-panel figure: (a) log-scale slope, (b) implied horizon."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Panel (a) — slope on log-y
        for H, g in df.groupby("n_heads"):
            ax1.plot(g["head_idx"], g["slope"],
                     marker="o", linewidth=1.8, label=f"H={H}")
        ax1.set_yscale("log")
        ax1.set_xlabel("head index h")
        ax1.set_ylabel("slope m_h  (log scale)")
        ax1.set_title("(a) Per-head slopes — geometric, log-spaced")
        ax1.axhline(2 ** -8, color="black", linestyle="--", linewidth=1,
                    label="m_last = 2^-8 (endpoint invariant)")
        ax1.legend(fontsize=9)
        ax1.grid(True, which="both", alpha=0.3)

        # Panel (b) — implied attention horizon, clipped at L=256 for readability
        for H, g in df.groupby("n_heads"):
            ax2.plot(g["head_idx"],
                     np.clip(g["horizon_tokens"], 1, 256),
                     marker="s", linewidth=1.8, label=f"H={H}")
        ax2.set_yscale("log")
        ax2.set_xlabel("head index h")
        ax2.set_ylabel(f"horizon (tokens until |bias| ≥ {self.horizon_threshold_nats})")
        ax2.set_title("(b) Implied attention horizon per head")
        ax2.legend(fontsize=9)
        ax2.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[Ex1] saved: {Path(save_to).resolve()}")

    # ------------------------------------------------------------------
    def answers(self, df: pd.DataFrame) -> None:
        """Print concise answers to the three exercise questions."""
        print("\n[Ex1] Answers")
        for H in (1, 16):
            m0 = df.loc[(df.n_heads == H) & (df.head_idx == 0), "slope"].iloc[0]
            print(f"    Q1  largest slope for H={H:2d}: {m0:.6f}")
        m_last = df.loc[df.head_idx == df.groupby("n_heads").head_idx.transform("max"), "slope"]
        print(f"    Q2  smallest slope across all H: {m_last.min():.6f} "
              f"(== 2^-8 — invariant, because the formula's exponent at "
              f"h=H-1 always equals -8)")
        # Uniform spacing is H-independent in absolute log2 terms only
        # when we normalise by 1/H; interpret "uniform" as "constant step
        # within each H".
        print("    Q3  On log2-scale, spacing within a given H is constant "
              "(−8/H per step). Larger H ⇒ finer spacing ⇒ more gradual "
              "multi-scale coverage. H=8 or H=16 give the smoothest "
              "spectrum; H=1 collapses to a single global filter.")


# ==========================================================================
# Exercise 2 — Length Extrapolation ("train short, test long")
# ==========================================================================
@dataclass
class Exercise2Solution(_ALiBiExperiment):
    """
    Exercise 2 — quantify the paper's headline claim on synthetic data.

    Setup
    -----
    * ``train_len``: length at which the "learned" positional embedding
      table is frozen. Positions beyond this clamp to ``train_len - 1``.
    * ``test_lengths``: multiples of ``train_len`` (2×, 4×, 8×).
    * Three configurations are compared:
        - none       : no positional encoding at all
        - learned_pe : frozen table, clamped at test time
        - alibi      : parametric bias, works at any L
    * Metric: "nearby attention mass" — average over all query positions
      of the sum of attention weights to keys within distance ≤ 2.

    Prediction
    ----------
    Learned-PE will show ~1.0 mass at L = train_len, then collapse to
    ~1/L at L = 2·train_len (all positions ≥ train_len share the same
    PE, so those queries are indistinguishable — attention to
    "nearby" becomes uniform). ALiBi will hold steady across the whole
    range. This IS the paper's extrapolation result, in miniature.
    """

    train_len: int = 32
    multiples: Tuple[int, ...] = (1, 2, 4, 8)
    window: int = 2
    trials: int = 5

    # ------------------------------------------------------------------
    def _nearby_mass(self, weights: np.ndarray, window: int) -> float:
        """
        Average, over queries, of the total attention weight assigned
        to keys within ±window of the query position.

        Note: queries 0 and L-1 have fewer neighbours; the denominator
        is ``weights.shape[-1]`` (all keys), so edge queries naturally
        have smaller mass. This is fine for a RELATIVE comparison across
        configurations at the SAME L.
        """
        L = weights.shape[-1]
        idx = np.arange(L)
        mask = np.abs(idx[:, None] - idx[None, :]) <= window      # (L, L)
        per_query = (weights * mask[None, :, :]).sum(axis=-1)   # (H, L)
        return float(per_query.mean())

    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        rows: List[dict] = []
        for mult in self.multiples:
            L = self.train_len * mult
            aggs = {"none": [], "learned_pe": [], "alibi": []}
            for t in range(self.trials):
                rng = self._rng(offset=1_000 + t)
                x = rng.normal(0.0, 1.0, (L, self.d_model))
                w_none = self._attention(x, use_alibi=False, use_learned_pe=False)
                w_lpe = self._attention(x, use_alibi=False,
                                        use_learned_pe=True,
                                        train_len=self.train_len)
                w_alb = self._attention(x, use_alibi=True, use_learned_pe=False)
                aggs["none"].append(self._nearby_mass(w_none, self.window))
                aggs["learned_pe"].append(self._nearby_mass(w_lpe, self.window))
                aggs["alibi"].append(self._nearby_mass(w_alb, self.window))
            rows.append(dict(
                test_len=L,
                multiple=mult,
                none=np.mean(aggs["none"]),
                learned_pe=np.mean(aggs["learned_pe"]),
                alibi=np.mean(aggs["alibi"]),
            ))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def plot(self, df: pd.DataFrame, save_to: str = "fig_alibi_ex2.png") -> None:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        ax.plot(df["test_len"], df["learned_pe"], marker="o",
                linewidth=2, label="learned PE (clamped)")
        ax.plot(df["test_len"], df["alibi"], marker="s",
                linewidth=2, label="ALiBi (parametric)")
        ax.plot(df["test_len"], df["none"], marker="^",
                linewidth=2, linestyle="--", label="no position info")
        ax.axvline(self.train_len, color="grey", linestyle=":",
                   linewidth=1.2, label=f"train_len = {self.train_len}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("test sequence length (log₂)")
        ax.set_ylabel(f"nearby attention mass (|i−j| ≤ {self.window})")
        ax.set_title("Exercise 2 — Length extrapolation: learned PE vs ALiBi")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[Ex2] saved: {Path(save_to).resolve()}")

    # ------------------------------------------------------------------
    def answers(self, df: pd.DataFrame) -> None:
        print("\n[Ex2] Answers")
        at_train = df.iloc[0]
        at_8x = df.iloc[-1]
        print(f"    Q1  At L=train_len:    learned_pe={at_train['learned_pe']:.3f}, "
              f"alibi={at_train['alibi']:.3f}")
        print(f"    Q2  At L=8·train_len:  learned_pe={at_8x['learned_pe']:.3f} "
              f"(≈ 1/L = {1.0/at_8x['test_len']:.3f}, i.e. essentially uniform), "
              f"alibi={at_8x['alibi']:.3f} (retained)")
        print("    Q3  Learned PE collapses *because* positions beyond the "
              "trained range all clamp to the same embedding — queries can "
              "no longer distinguish each other, so attention becomes "
              "uniform. ALiBi has no such state: the bias depends only on "
              "relative distance, so nearby-attention is preserved. This "
              "is the paper's 'train short, test long' result, reproduced "
              "at the level of a single attention layer.")


# ==========================================================================
# Exercise 3 — Local vs Global Attention Tradeoff
# ==========================================================================
@dataclass
class Exercise3Solution(_ALiBiExperiment):
    """
    Exercise 3 — visualise ALiBi's two documented downsides:

    (a) Recency bias — attention mass to the LAST K tokens.
        ALiBi keeps a fixed fraction focused on recent context;
        uniform (no-bias) attention dilutes as L grows.

    (b) Long-range suppression — attention to a "needle" placed at
        distance D from the query. ALiBi boosts the needle for small D
        but suppresses it for large D. Uniform attention assigns the
        same weight regardless of D. The crossover point is where ALiBi
        changes from helpful to harmful on a "fetch-the-needle" task.

    Both are computed on synthetic Q/K/V under a fixed seed, so the
    effect is attributable purely to the ALiBi bias, not to input noise.
    """

    seq_len: int = 64
    last_k: int = 4
    distances: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    # ------------------------------------------------------------------
    def _attention_to_last_k(self) -> Tuple[float, float]:
        """Return (uniform_baseline, alibi) attention mass on last K tokens
        for a query at position L−1, averaged over heads."""
        rng = self._rng(offset=3_000)
        x = rng.normal(0.0, 1.0, (self.seq_len, self.d_model))
        w_none = self._attention(x, use_alibi=False).mean(axis=0)  # (L, L)
        w_alb = self._attention(x, use_alibi=True).mean(axis=0)
        lo = self.seq_len - self.last_k
        baseline = float(w_none[-1, lo:].sum())
        alibi = float(w_alb[-1, lo:].sum())
        return baseline, alibi

    # ------------------------------------------------------------------
    def _attention_to_needle(self, D: int) -> Tuple[float, float]:
        """
        Place a synthetic query at position L−1. Measure attention to
        the single key at position L−1−D under (i) no bias, (ii) ALiBi.
        Returns (uniform, alibi) mean over heads.
        """
        rng = self._rng(offset=4_000 + D)
        x = rng.normal(0.0, 1.0, (self.seq_len, self.d_model))
        w_none = self._attention(x, use_alibi=False).mean(axis=0)
        w_alb = self._attention(x, use_alibi=True).mean(axis=0)
        key_pos = self.seq_len - 1 - D
        return float(w_none[-1, key_pos]), float(w_alb[-1, key_pos])

    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        rows: List[dict] = []
        b, a = self._attention_to_last_k()
        rows.append(dict(analysis="last-K recency",
                         x=self.last_k, uniform=b, alibi=a))
        for D in self.distances:
            n, aD = self._attention_to_needle(D)
            rows.append(dict(analysis="needle @ distance D",
                             x=D, uniform=n, alibi=aD))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def plot(self, df: pd.DataFrame, save_to: str = "fig_alibi_ex3.png") -> None:
        needle = df[df.analysis == "needle @ distance D"]
        lastk = df[df.analysis == "last-K recency"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

        # (a) Recency mass
        labels = ["uniform (no bias)", "ALiBi"]
        vals = [lastk.uniform.iloc[0], lastk.alibi.iloc[0]]
        bars = ax1.bar(labels, vals, color=["steelblue", "seagreen"])
        for b in bars:
            ax1.text(b.get_x() + b.get_width() / 2,
                     b.get_height() + 0.01,
                     f"{b.get_height():.3f}", ha="center", fontsize=10)
        ax1.set_ylabel(f"attention mass on last {self.last_k} tokens")
        ax1.set_title("(a) Recency — ALiBi keeps recent context sharp")
        ax1.set_ylim(0, max(vals) * 1.25)
        ax1.grid(True, axis="y", alpha=0.3)

        # (b) Needle retrieval vs distance
        ax2.plot(needle.x, needle.uniform, marker="o", linewidth=2,
                 label="uniform (no bias)")
        ax2.plot(needle.x, needle.alibi, marker="s", linewidth=2,
                 label="ALiBi")
        # Mark the crossover point where alibi drops below uniform
        below = needle.alibi < needle.uniform
        if below.any():
            first = needle.loc[below.idxmax(), "x"]
            ax2.axvline(first, color="firebrick", linestyle=":",
                        linewidth=1.2, label=f"crossover D≈{first}")
        ax2.set_xscale("log", base=2)
        ax2.set_xlabel("needle distance D (log₂)")
        ax2.set_ylabel("attention weight on needle")
        ax2.set_title("(b) Long-range — ALiBi suppresses far needles")
        ax2.legend(); ax2.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[Ex3] saved: {Path(save_to).resolve()}")

    # ------------------------------------------------------------------
    def answers(self, df: pd.DataFrame) -> None:
        lastk = df[df.analysis == "last-K recency"].iloc[0]
        needle = df[df.analysis == "needle @ distance D"]
        below = needle.alibi < needle.uniform
        crossover = int(needle.loc[below.idxmax(), "x"]) if below.any() else None

        print("\n[Ex3] Answers")
        print(f"    Q1  Recency: uniform={lastk.uniform:.3f}, "
              f"ALiBi={lastk.alibi:.3f} — ALiBi keeps a "
              f"{lastk.alibi / max(lastk.uniform, 1e-9):.1f}× larger "
              f"share on recent tokens. Good for short-context tasks.")
        print(f"    Q2  Needle retrieval: for close needles (D ≤ "
              f"{crossover or '∞'}) ALiBi BEATS uniform; for far needles "
              f"it LOSES. This is the tradeoff the technical card's "
              f"'cons' describes.")
        print("    Q3  Practical reading: use ALiBi when recent context "
              "dominates (chat, streaming, next-token prediction, code "
              "completion). Avoid it — or pair it with RoPE — for "
              "tasks that must fetch tokens from the far end of a long "
              "context (document QA, first-token retrieval).")


# ==========================================================================
# Expected-outcomes block (learner verification)
# ==========================================================================
def print_expected_outcomes() -> None:
    print("=" * 74)
    print("EXPECTED OUTCOMES — use to verify your run")
    print("=" * 74)
    print("""
    Ex1  slopes(H=4) = [0.2500, 0.0625, 0.0156, 0.0039]           (exact)
         log2-spacing within any H = −8/H                          (exact)
         m_last == 2^-8 for every H                                 (exact)

    Ex2  L = 32  (train):  learned_pe ≈ alibi ≈ 0.9–1.0            (±0.05)
         L = 256 (8×):     learned_pe ≈ 1/L ≈ 0.004                (±0.002)
                            alibi       ≥ 0.5 (retained)
         ⇒ learned-PE extrapolation collapses, ALiBi does not.

    Ex3  last-K recency:  alibi > uniform for every reasonable K.   (qual.)
         needle retrieval: alibi > uniform for D ≲ 8–16,
                            alibi < uniform for D ≳ 32.
         Crossover is reported on the plot as a vertical line.

    Tolerances
    ----------
    Determinism: every metric is computed from a fixed seed, so EXACT
    equality across runs (to floating-point round-off ≈1e-9) is
    expected. The tolerances above are for pedagogical clarity only.
    """)


# ==========================================================================
# Driver
# ==========================================================================
def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print_expected_outcomes()

    # ---------- Exercise 1 -------------------------------------------
    print("\n" + "=" * 74)
    print("EXERCISE 1 — Slope Spectrum Geometry")
    print("=" * 74)
    ex1 = Exercise1Solution()
    ex1.theory_checks()
    df1 = ex1.run()
    print(df1.to_string(index=False))
    ex1.plot(df1, save_to="fig_alibi_ex1.png")
    ex1.answers(df1)

    # ---------- Exercise 2 -------------------------------------------
    print("\n" + "=" * 74)
    print("EXERCISE 2 — Length Extrapolation (train short, test long)")
    print("=" * 74)
    ex2 = Exercise2Solution()
    df2 = ex2.run()
    print(df2.to_string(index=False))
    ex2.plot(df2, save_to="fig_alibi_ex2.png")
    ex2.answers(df2)

    # ---------- Exercise 3 -------------------------------------------
    print("\n" + "=" * 74)
    print("EXERCISE 3 — Local vs Global Attention Tradeoff")
    print("=" * 74)
    ex3 = Exercise3Solution()
    df3 = ex3.run()
    print(df3.to_string(index=False))
    ex3.plot(df3, save_to="fig_alibi_ex3.png")
    ex3.answers(df3)

    print("\nDONE. Figures saved to fig_alibi_ex1.png, "
          "fig_alibi_ex2.png, fig_alibi_ex3.png")


if __name__ == "__main__":
    main()



# python3 -m venv .venv
# source .venv/bin/activate
# pip install numpy matplotlib pandas
# python3 alibi_exercises.py

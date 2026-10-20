"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
# Exerexercise 1 — Tune filtering strictness
# Goal: Understand the quality vs. quantity tradeoff.
#     Run the pipeline with filtering_strictness = 0.5, 0.7, 0.9.
#     Plot pass rate and downstream delta vs strictness.
#     Hint: at 0.5 the downstream delta may go up because more data is kept — even noisy data helps if the baseline was tiny. At 0.9, the delta may drop because too few examples remain. The sweet spot is where downstream delta is maximised, not where pass rate is highest.
# Exercise 2 — Measure the effect of dedup aggressiveness
# Goal: See how MinHash threshold affects duplicate detection and diversity.
#     Run with dedup_aggressiveness = 0.6, 0.8, 0.95.
#     Report pass rate, n-gram entropy, and the number of duplicates removed at each level.
#     Hint: high threshold (0.95) is lax — it only removes near-identical copies; low threshold (0.6) aggressively removes near-duplicates. The card's third con (mode collapse) is visible if entropy stays low even after dedup — it means the teacher, not the filter, is the problem.
# Exercise 3 — Add a real chain-of-thought judge
# Goal: Extend the pipeline with a reasoning-quality check.
#     Add a ReasoningQualityJudge class that scores reasoning traces on a 0–1 scale based on: number of distinct steps, presence of "Step 1/Step 2", numerical consistency with the final answer.
#     Plug it into SyntheticDataPipeline._filter as an additional stage.
#     Hint: the MockJudge already gives a proxy. The point of this exercise is to show how the DRY interface (score(ex) -> float) makes it trivial to swap in a stronger judge. For a real implementation, use the technical card's claude-3-5-sonnet pattern — see RealTeacherLLM for the exact API call.


synthetic_data_generation_exercises.py
======================================

Complete solutions to Exercises 1–3 from the Synthetic Data Generation
learning module.

Design principles
-----------------
- DRY: solutions reuse `SyntheticDataPipeline`, `MockTeacherLLM`,
  `DownstreamProxy`, and `ngram_entropy` from the main module.
- Dataclasses: every experiment is a `@dataclass` that encapsulates its
  config and results.
- Full documentation: module, class, and method docstrings; comments
  explain 'why', not just 'what'.
- Reproducibility: fixed seeds; results are identical on the same
  Python / library versions.

Usage
-----
    python3 synthetic_data_generation_exercises.py

Saves figures `fig_ex1_sdg.png`, `fig_ex2_sdg.png`, `fig_ex3_sdg.png`
in the current working directory.

Dependencies
------------
Same as `synthetic_data_generation.py`:
    pip3 install numpy pandas matplotlib scikit-learn
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import everything we need from the main module.
from synthetic_data_generation import (
    GenerationConfig,
    HeuristicFilter,
    MockJudge,
    MockTeacherLLM,
    MinHashDeduplicator,
    PipelineResult,
    Seed,
    SyntheticDataPipeline,
    SyntheticExample,
    DownstreamProxy,
    ngram_entropy,
)


# ======================================================================
# EXERCISE 1 — Tuning filtering strictness
# ======================================================================

@dataclass
class Exercise1Solution:
    """
    Sweep `filtering_strictness` ∈ {0.5, 0.7, 0.9} and record:
      - pass rate
      - downstream delta (proxy)
      - 3-gram entropy

    The learner should observe that the *highest* pass rate does NOT
    correspond to the *best* downstream delta — because a lax filter keeps
    more data, but a strict filter keeps better data. The exercise teaches
    the quality/quantity tradeoff.
    """
    strictness_values: Tuple[float, ...] = (0.50, 0.70, 0.90)
    base_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 42

    def run(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for strict in self.strictness_values:
            # Clone the base config, overriding only the field under test.
            cfg = GenerationConfig(
                teacher_temperature=self.base_config.teacher_temperature,
                seed_prompt_count=self.base_config.seed_prompt_count,
                filtering_strictness=strict,
                dedup_aggressiveness=self.base_config.dedup_aggressiveness,
                max_examples_per_seed=self.base_config.max_examples_per_seed,
                random_seed=self.seed,
            )
            pipe = SyntheticDataPipeline(
                teacher=MockTeacherLLM(seed=self.seed), config=cfg,
            )
            result = pipe.run()
            delta = DownstreamProxy().evaluate(result.kept)
            rows.append(dict(
                filtering_strictness=strict,
                n_kept=len(result.kept),
                pass_rate=result.pass_rate,
                diversity=ngram_entropy([e.as_text() for e in result.examples], n=3),
                baseline_acc=delta["baseline_acc"],
                synthetic_acc=delta["synthetic_acc"],
                downstream_delta=delta["delta"],
            ))
        return pd.DataFrame(rows)

    def plot(self, df: pd.DataFrame, save_to: Path = Path("fig_ex1_sdg.png")) -> None:
        """Plot pass rate and downstream delta vs strictness."""
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        x = df["filtering_strictness"]
        ax1.plot(x, df["pass_rate"], marker="o", color="steelblue",
                 label="Pass rate")
        ax1.set_xlabel("Filtering strictness")
        ax1.set_ylabel("Pass rate", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax2 = ax1.twinx()
        ax2.plot(x, df["downstream_delta"], marker="s", color="darkorange",
                 label="Downstream delta (proxy)")
        ax2.set_ylabel("Downstream delta (proxy)", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")
        ax1.set_title("Exercise 1 — Quality vs Quantity vs Downstream Delta")
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex1] saved {save_to.resolve()}")


# ======================================================================
# EXERCISE 2 — Effect of dedup aggressiveness
# ======================================================================

@dataclass
class Exercise2Solution:
    """
    Sweep `dedup_aggressiveness` ∈ {0.6, 0.8, 0.95} and record:
      - pass rate (dedup is the last stage, so pass rate is dominated by
        heuristics + judge; the interesting number is *duplicates removed*)
      - duplicates removed (count)
      - 3-gram entropy after dedup

    Higher threshold = laxer dedup (only near-identical copies removed).
    Lower threshold = stricter dedup (near-duplicates removed too).
    """
    thresholds: Tuple[float, ...] = (0.60, 0.80, 0.95)
    seed: int = 42

    def run(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for thr in self.thresholds:
            cfg = GenerationConfig(
                teacher_temperature=0.9,
                seed_prompt_count=40,
                filtering_strictness=0.70,
                dedup_aggressiveness=thr,
                max_examples_per_seed=3,
                random_seed=self.seed,
            )
            pipe = SyntheticDataPipeline(
                teacher=MockTeacherLLM(seed=self.seed), config=cfg,
            )
            result = pipe.run()
            n_duplicates = sum(1 for r in result.filter_results if r.is_duplicate)
            rows.append(dict(
                dedup_threshold=thr,
                n_generated=len(result.examples),
                n_kept=len(result.kept),
                n_duplicates_removed=n_duplicates,
                pass_rate=result.pass_rate,
                diversity_after_dedup=ngram_entropy(
                    [e.as_text() for e in result.kept], n=3
                ),
            ))
        return pd.DataFrame(rows)

    def plot(self, df: pd.DataFrame, save_to: Path = Path("fig_ex2_sdg.png")) -> None:
        """Plot duplicates-removed and post-dedup entropy vs threshold."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        x = df["dedup_threshold"]
        ax1.bar(x.astype(str), df["n_duplicates_removed"], color="indianred")
        ax1.set_title("Exercise 2 — Duplicates removed")
        ax1.set_xlabel("Dedup threshold (higher = laxer)")
        ax1.set_ylabel("Duplicates removed")
        ax2.plot(x, df["diversity_after_dedup"], marker="o", color="seagreen")
        ax2.set_title("Exercise 2 — Diversity after dedup")
        ax2.set_xlabel("Dedup threshold")
        ax2.set_ylabel("3-gram entropy (bits)")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex2] saved {save_to.resolve()}")


# ======================================================================
# EXERCISE 3 — Reasoning-quality judge
# ======================================================================

@dataclass
class ReasoningQualityJudge:
    """
    Scores a reasoning trace on [0, 1] based on:
      - step count        (Step N / numbered lists)   → up to 0.4
      - distinct words    (avoids "Step 1: step 1")   → up to 0.3
      - answer agreement  (numeric answer appears in reasoning) → up to 0.3

    The point is the interface: `score(ex) -> float`, identical to
    `MockJudge`, so it can be plugged into the pipeline as a drop-in
    replacement (DRY).
    """
    def score(self, ex: SyntheticExample) -> float:
        if not ex.reasoning:
            return 0.0
        text = ex.reasoning
        # 1) step-count signal
        steps = len(re.findall(r"Step\s*\d+|^\s*\d+[.)]\s", text, re.MULTILINE))
        step_score = min(1.0, steps / 3.0) * 0.4
        # 2) vocabulary diversity
        words = re.findall(r"[A-Za-z]+", text.lower())
        vocab_ratio = len(set(words)) / max(1, len(words))
        vocab_score = min(1.0, vocab_ratio * 3.0) * 0.3
        # 3) answer agreement (only for numeric answers)
        answer = ex.answer.strip()
        answer_score = 0.0
        if answer and re.fullmatch(r"-?\d+(?:\.\d+)?", answer):
            answer_score = 0.3 if answer in text else 0.0
        return round(min(1.0, step_score + vocab_score + answer_score), 4)


@dataclass
class Exercise3Solution:
    """
    Re-run the pipeline with `ReasoningQualityJudge` added as an extra
    filter stage, and compare:
      - kept examples
      - pass rate
      - downstream delta
    against the baseline (MockJudge only).
    """
    seed: int = 42
    judge_threshold: float = 0.40

    def run(self) -> pd.DataFrame:
        cfg = GenerationConfig(
            teacher_temperature=0.9,
            seed_prompt_count=40,
            filtering_strictness=0.70,
            dedup_aggressiveness=0.80,
            max_examples_per_seed=3,
            random_seed=self.seed,
        )

        # --- Baseline: MockJudge only ---
        base = SyntheticDataPipeline(
            teacher=MockTeacherLLM(seed=self.seed), config=cfg,
        ).run()

        # --- Augmented: MockJudge + ReasoningQualityJudge ---
        pipe = SyntheticDataPipeline(
            teacher=MockTeacherLLM(seed=self.seed), config=cfg,
        )
        # Note: we re-run with a *single* pipeline and post-filter the kept
        # set, to isolate the reasoning-judge's effect from the base filter.
        aug = pipe.run()
        rj = ReasoningQualityJudge()
        aug_kept = [ex for ex in aug.kept if rj.score(ex) >= self.judge_threshold]

        rows = []
        for name, kept, all_examples in [
            ("baseline (MockJudge only)", base.kept, base.examples),
            ("augmented (+ReasoningJudge)", aug_kept, aug.examples),
        ]:
            delta = DownstreamProxy().evaluate(kept)
            rows.append(dict(
                pipeline=name,
                n_kept=len(kept),
                pass_rate=len(kept) / len(all_examples) if all_examples else 0.0,
                diversity=ngram_entropy([e.as_text() for e in kept], n=3),
                baseline_acc=delta["baseline_acc"],
                synthetic_acc=delta["synthetic_acc"],
                downstream_delta=delta["delta"],
            ))
        return pd.DataFrame(rows)

    def plot(self, df: pd.DataFrame, save_to: Path = Path("fig_ex3_sdg.png")) -> None:
        """Compare baseline vs augmented pipeline on three metrics."""
        metrics = ["pass_rate", "diversity", "downstream_delta"]
        x = np.arange(len(metrics))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(x - w / 2, df.iloc[0][metrics].astype(float), w,
               label=df.iloc[0]["pipeline"], color="steelblue")
        ax.bar(x + w / 2, df.iloc[1][metrics].astype(float), w,
               label=df.iloc[1]["pipeline"], color="seagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title("Exercise 3 — Baseline vs Augmented (Reasoning Judge)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex3] saved {save_to.resolve()}")


# ======================================================================
# DRIVER
# ======================================================================

def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)

    print("=" * 70)
    print(" EXERCISE 1 — Tuning filtering strictness")
    print("=" * 70)
    ex1 = Exercise1Solution()
    df1 = ex1.run()
    print(df1.to_string(index=False))
    ex1.plot(df1)

    print("\n" + "=" * 70)
    print(" EXERCISE 2 — Dedup aggressiveness")
    print("=" * 70)
    ex2 = Exercise2Solution()
    df2 = ex2.run()
    print(df2.to_string(index=False))
    ex2.plot(df2)

    print("\n" + "=" * 70)
    print(" EXERCISE 3 — Adding a reasoning-quality judge")
    print("=" * 70)
    ex3 = Exercise3Solution()
    df3 = ex3.run()
    print(df3.to_string(index=False))
    ex3.plot(df3)


if __name__ == "__main__":
    main()


# python3 -m venv .venv
# source .venv/bin/activate
# pip3 install numpy pandas matplotlib scikit-learn
# python3 syntheticdatagen_learning.py            # main module
# python3 syntheticdatagen_exercises.py  # exercise solutions

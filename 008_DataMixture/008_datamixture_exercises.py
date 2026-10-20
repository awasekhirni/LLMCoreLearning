"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#
 Complete solutions to Exercises 1–3 from the Data Mixture & Curriculum
 Learning module.

   Exercise 1 — Temperature sweep + capability Pareto frontier.
   Exercise 2 — Curriculum stabilization via stage-2 loss variance.
   Exercise 3 — Sampler validation via KL divergence + chi-square.

 Design principles
 -----------------

 - Dataclasses everywhere — each exercise is a `@dataclass` with a single
   `run()` and one `plot()` so it is trivially instantiable.
 - Reproducibility — every experiment threads `SEED` through
   `np.random.default_rng`.

 Dependencies
 ------------
 numpy, pandas, matplotlib (required)
 scipy (optional — only for p-value; a manual fallback is provided)

 Import contract
 ---------------
 This file does `from data_mixture_curriculum import ...`.  Place both
 files in the same directory, or add that directory to PYTHONPATH.

    datamixture_learning.py
    datamixture_exercise.py


Exercise 1 — Temperature sweep and the Pareto frontier

Question. Sweep temperature across T ∈ {0.1, 0.3, 0.5, 0.7, 1.0}, train the bigram proxy under each setting, and plot final_math_loss (y) vs final_web_loss (x) as a curve. Which T sits on the Pareto frontier? Which T is dominated? Write a report with a single figure and 3–4 sentences.

Hint. Reuse MixtureExperiment.run_flat(T) for each T. Plot one scatter per T. The Pareto frontier is the set of points with no other point strictly to the lower-left. You should find that T=0.3 dominates T=1.0 (better math for a tiny web cost), but T=0.1 may over-flatten and lose web performance entirely. This is the card's "capability Pareto frontier" benchmark metric.
Exercise 2 — Does curriculum stabilize training?

Question. Run the same experiment as Exercise 1 but add run_curriculum(). For each method, compute the variance of the loss curve in stage 2 only (steps ≥ 8). Which method has the lowest loss variance in the specialized stage, and does that match the card's claim that "curriculum stabilizes small-model training"?

Hint. history[history["step"] >= 8][domain].var(). You should find that curriculum has lower stage-2 variance than flat-T=0.3 — because stage 2 is reached gradually (via stage 1), so the model doesn't suffer the "cold-start on hard data" destabilization that flat-T=0.3 from step 0 would cause. This is the card's "curriculum stabilizes small-model training" pros-list item made measurable.
Exercise 3 — Verify the sampler mathematically

Question. For T=0.3, sample n ∈ {100, 1_000, 10_000, 100_000} draws and compute both (a) the empirical KL divergence to the theory and (b) the chi-square goodness-of-fit statistic against the expected distribution. Plot both on a log-x axis and confirm that both decrease like 1/√n (up to a constant). This proves the sampler is not a bug.

Hint. Use DomainMixtureSampler.empirical_distribution(n). For chi-square, use scipy.stats.chisquare or compute Σ (obs − exp)² / exp manually — the critical value for df=4 at α=0.01 is ≈ 13.28, and you should see the statistic drop below the critical value as n grows. If you do not have scipy, math.erfc-based p-values are fine; or simply plot the statistic and confirm it decreases monotonically. This is the "verify sampling matches p_i ∝ n_i^T" check that every practitioner does before trusting a mixture optimizer.
"""

# python3 -m venv .venv
# source .venv/bin/activate
# pip3 install numpy pandas matplotlib scipy   # scipy is optional but recommended

# # 1) Main module
# python3 data_mixture_curriculum.py
# # → prints corpus stats, mixture tables, final losses, ELI5 walkthrough
# # → saves fig_input.png, fig_processing.png, fig_outcome.png, fig_sampling.png

# # 2) Exercise solutions
# python3 exercise_solutions.py
# # → prints Pareto frontier + variance table + KL/chi-square table
# # → saves fig_ex1.png, fig_ex2.png, fig_ex3.png
# #

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_mixture_curriculum import (
    SEED,
    CharBigramModel,
    CurriculumScheduler,
    DomainMixtureSampler,
    MixtureExperiment,
    generate_corpus,
)

#: scipy is optional — only used for a nicer p-value in Exercise 3.
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:                                  # pragma: no cover
    SCIPY_AVAILABLE = False


def _section(title: str) -> None:
    """Print a visually distinct section header."""
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


# ======================================================================
# EXERCISE 1 — Temperature sweep & Pareto frontier
# ======================================================================

@dataclass
class Exercise1Solution:
    """
    Sweep temperature and tabulate per-domain final losses.

    The "Pareto frontier" is the set of (T, method) points such that no
    other point has both lower web loss AND lower math loss.  Points on
    the frontier represent genuine capability trade-offs; points *off*
    the frontier are dominated (strictly worse on both axes).

    Attributes
    ----------
    corpus : dict[str, str]
    temperatures : tuple[float, ...]
        Temperatures to sweep.  Default = (0.1, 0.3, 0.5, 0.7, 1.0).
    total_chars : int
        Passed to `MixtureExperiment`.
    n_snapshots : int
        Passed to `MixtureExperiment`.
    """
    corpus: Dict[str, str]
    temperatures: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0)
    total_chars: int = 100_000
    n_snapshots: int = 10

    def run(self) -> pd.DataFrame:
        """
        Return one row per temperature with ``final_web_loss`` and
        ``final_math_loss`` (the two axes of the Pareto plot).
        """
        sizes = {d: len(t) for d, t in self.corpus.items()}
        exp = MixtureExperiment(
            corpus=self.corpus, domain_sizes=sizes,
            total_chars=self.total_chars, n_snapshots=self.n_snapshots,
        )
        rows: List[dict] = []
        for T in self.temperatures:
            res = exp.run_flat(T)
            rows.append({
                "temperature": T,
                "web_loss": res["final"]["web"],
                "math_loss": res["final"]["math"],
                "code_loss": res["final"]["code"],
            })
        return pd.DataFrame(rows)

    @staticmethod
    def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to the Pareto frontier: rows with no other row strictly
        better on *both* web_loss and math_loss.
        """
        keep: List[int] = []
        for i, row in df.iterrows():
            dominated = False
            for j, other in df.iterrows():
                if i == j:
                    continue
                if (other["web_loss"] <= row["web_loss"]
                        and other["math_loss"] <= row["math_loss"]
                        and (other["web_loss"] < row["web_loss"]
                             or other["math_loss"] < row["math_loss"])):
                    dominated = True
                    break
            if not dominated:
                keep.append(i)
        return df.loc[keep].sort_values("web_loss").reset_index(drop=True)

    def plot(self, df: pd.DataFrame, save_to: str = "fig_ex1.png") -> None:
        """Pareto scatter + annotated temperature labels."""
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(df["web_loss"], df["math_loss"], s=140,
                   c=df["temperature"], cmap="viridis", edgecolor="black")
        for _, r in df.iterrows():
            ax.annotate(f"T={r['temperature']}", (r["web_loss"], r["math_loss"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=10)
        frontier = self.pareto_frontier(df)
        ax.plot(frontier["web_loss"], frontier["math_loss"],
                color="red", linestyle="--", alpha=0.6,
                label="Pareto frontier")
        ax.set_xlabel("final web loss (lower = better)")
        ax.set_ylabel("final math loss (lower = better)")
        ax.set_title("Exercise 1 — Capability Pareto across temperatures")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex1] figure saved → {Path(save_to).resolve()}")


# ======================================================================
# EXERCISE 2 — Curriculum stabilizes stage-2 training
# ======================================================================

@dataclass
class Exercise2Solution:
    """
    Compare the variance of stage-2 loss across three methods.

    Theory (from the card): a two-stage curriculum "stabilizes
    small-model training".  Operationally we test that with the
    *variance of the loss curve in the specialized stage*: a method
    that reaches stage 2 gradually should have lower variance than
    one that starts at T=0.3 from step 0.

    Attributes
    ----------
    corpus, total_chars, n_snapshots : same as Ex1.
    stage2_start : int
        First step that belongs to stage 2.  Default = 8 (of 10),
        matching the 80/20 curriculum.
    """
    corpus: Dict[str, str]
    total_chars: int = 100_000
    n_snapshots: int = 10
    stage2_start: int = 8

    def run(self) -> pd.DataFrame:
        """
        Return one row per method with stage-2 variance for each domain.
        """
        sizes = {d: len(t) for d, t in self.corpus.items()}
        exp = MixtureExperiment(
            corpus=self.corpus, domain_sizes=sizes,
            total_chars=self.total_chars, n_snapshots=self.n_snapshots,
        )
        runs = {
            "uniform (T=1.0)": exp.run_uniform(),
            "flat T=0.3":       exp.run_flat(0.3),
            "curriculum":       exp.run_curriculum(),
        }
        rows: List[dict] = []
        for name, res in runs.items():
            hist = res["history"]
            stage2 = hist[hist["step"] >= self.stage2_start]
            row = {"method": name}
            for d in self.corpus:
                row[f"stage2_var_{d}"] = float(stage2[d].var())
            row["stage2_var_mean"] = float(np.mean(
                [row[f"stage2_var_{d}"] for d in self.corpus]
            ))
            rows.append(row)
        return pd.DataFrame(rows)

    def plot(self, df: pd.DataFrame, save_to: str = "fig_ex2.png") -> None:
        """Grouped bar of stage-2 variance per method."""
        fig, ax = plt.subplots(figsize=(10, 5))
        cols = [c for c in df.columns if c.startswith("stage2_var_") and c != "stage2_var_mean"]
        width = 0.8 / len(df)
        x = np.arange(len(cols))
        for i, (_, row) in enumerate(df.iterrows()):
            ax.bar(x + i * width, [row[c] for c in cols], width,
                   label=row["method"], edgecolor="black")
        ax.set_xticks(x + width * (len(df) - 1) / 2)
        ax.set_xticklabels([c.replace("stage2_var_", "") for c in cols], rotation=15)
        ax.set_ylabel("variance of stage-2 loss")
        ax.set_title("Exercise 2 — Curriculum stabilizes stage 2")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex2] figure saved → {Path(save_to).resolve()}")


# ======================================================================
# EXERCISE 3 — Validate the sampler via KL and chi-square
# ======================================================================

@dataclass
class Exercise3Solution:
    """
    Statistically validate that ``DomainMixtureSampler`` matches
    ``p_i ∝ n_i^T``.

    Metrics
    -------
    * ``KL(empirical || expected)`` — should decay like 1/√n.
    * ``chi-square = Σ (obs − exp)² / exp`` — should also decay; the
      critical value for df = (#domains − 1) at α=0.01 is ≈ 13.28 for 5
      domains.  We report a p-value via scipy if available, otherwise
      report only the statistic (still valid for the trend plot).
    """
    domain_sizes: Dict[str, int]
    temperature: float = 0.3
    n_values: Tuple[int, ...] = (100, 1_000, 10_000, 100_000)

    def run(self) -> pd.DataFrame:
        """Return one row per n with KL, chi-square, and p-value."""
        sampler = DomainMixtureSampler(self.domain_sizes, self.temperature)
        rows: List[dict] = []
        for n in self.n_values:
            rng = np.random.default_rng(SEED)
            df = sampler.empirical_distribution(n=n, rng=rng)
            obs_counts = df["count"].to_numpy(dtype=float)
            exp_counts = (df["expected"] * n).to_numpy(dtype=float)
            chi2 = float(np.sum((obs_counts - exp_counts) ** 2 / exp_counts))
            if SCIPY_AVAILABLE:
                p = float(scipy_stats.chi2.sf(chi2, df=len(df) - 1))
            else:
                p = float("nan")
            kl = float(np.sum(df["expected"] * np.log(
                df["expected"] / np.clip(df["observed"], 1e-12, None)
            )))
            rows.append({"n": n, "kl": kl, "chi2": chi2, "p_value": p})
        return pd.DataFrame(rows)

    def plot(self, df: pd.DataFrame, save_to: str = "fig_ex3.png") -> None:
        """Log-log plot: both metrics should decay like 1/√n."""
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.loglog(df["n"], df["kl"], marker="o",
                  label="KL(empirical || expected)")
        ax.loglog(df["n"], df["chi2"], marker="s",
                  label="chi-square statistic")
        # Reference slope: 1/sqrt(n) (law of large numbers).
        ref = df["chi2"].iloc[0] * (df["n"].iloc[0] / df["n"].to_numpy()) ** 0.5
        ax.loglog(df["n"], ref, "--", alpha=0.6, label="∝ 1/√n reference")
        ax.set_xlabel("number of samples (n)")
        ax.set_ylabel("statistic (lower = more accurate)")
        ax.set_title("Exercise 3 — Sampler validation (KL & chi-square)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex3] figure saved → {Path(save_to).resolve()}")


# ======================================================================
# DRIVER
# ======================================================================

def main() -> None:
    """Run all three exercise solutions end-to-end."""
    pd.set_option("display.width", 170)
    pd.set_option("display.precision", 4)
    corpus = generate_corpus()

    # ---------------------------------------------------------------- 1
    _section("EXERCISE 1 — Temperature sweep & Pareto frontier")
    ex1 = Exercise1Solution(corpus=corpus)
    df1 = ex1.run()
    print(df1.to_string(index=False))
    print("\nPareto frontier (no other T strictly beats these on both axes):")
    print(Exercise1Solution.pareto_frontier(df1).to_string(index=False))
    ex1.plot(df1, "fig_ex1.png")

    # ---------------------------------------------------------------- 2
    _section("EXERCISE 2 — Curriculum stabilizes stage-2 loss")
    ex2 = Exercise2Solution(corpus=corpus)
    df2 = ex2.run()
    print(df2.to_string(index=False))
    ex2.plot(df2, "fig_ex2.png")

    # ---------------------------------------------------------------- 3
    _section("EXERCISE 3 — Validate sampler (KL + chi-square)")
    sizes = {d: len(t) for d, t in corpus.items()}
    ex3 = Exercise3Solution(domain_sizes=sizes, temperature=0.3)
    df3 = ex3.run()
    print(df3.to_string(index=False))
    if not SCIPY_AVAILABLE:
        print("\n[note] scipy unavailable — p-value column is NaN. "
              "chi2 < 13.28 (df=4, α=0.01) still means 'not rejected'.")
    ex3.plot(df3, "fig_ex3.png")

    _section("DONE")
    print("Figures: fig_ex1.png, fig_ex2.png, fig_ex3.png")


if __name__ == "__main__":
    main()

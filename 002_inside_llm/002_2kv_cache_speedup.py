"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

ELIM5
Imagine you're reading a long book out loud, one word at a time. Every time you read a new word, you don't re-read the whole book from page 1 — you just read the new word, because you remember everything before. That's what the KV cache does for the LLM: it remembers the Key and Value for every word already seen, so the LLM only has to compute new ones. Without the cache, the LLM would have to re-process the entire book every single time it added a word — that's very slow! This exercise proves it: we time both approaches and watch the cache version stay fast while the no-cache version gets slower and slower

Packages to install
pip3 install numpy>=1.24, pandas>=2.0, matplotlib>=3.7, scikit-learn>=1.3


Input
A prompt of prompt_length tokenIDs (random,seed=42) and a
list of n_new values to sweep

Expected outcome
-  Cached timing grows linearly in n_new: time_cached ≈ a * n_new + b.
No-cache timing grows quadratically: time_no_cache ≈ c * n_new^2.
Speedup ratio (no_cache / cache) increases with n_new.
All runs are deterministic (same seed → identical token sequences; only timing differs).
A PNG fig_kv_cache_speedup.png is saved showing both curves + speedup ratio.

------
Benchmark the KV cache vs. naive re-computation in MiniLLM generation.

Educational objectives:
    - Empirically demonstrate that the KV cache turns O(L^2 * d) per-step
      generation cost into O(L * d).
    - Practice careful benchmarking (warm-up, multiple repeats, medians).
    - Plot timing curves and a speedup ratio on a single figure.

Design principles:
    - Open/Closed: we extend MiniLLM via subclassing rather than modifying it.
    - Single Responsibility: CacheAwareMiniLLM does generation;
      KVCacheBenchmark does timing; BenchmarkPlotter does plotting.
    - DRY: timing helper reused for both modes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Callable
import time
import numpy as np
import matplotlib.pyplot as plt

from 002_inside_llm import MiniLLM, TransformerConfig, Sampler


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class BenchmarkConfig:
    """All knobs for the benchmark in one place.

    Attributes:
        prompt_length:  Number of tokens in the seed prompt.
        n_new_values:  Sweep of generation lengths to benchmark.
        n_repeats:      Times to repeat each measurement (median is reported).
        seed:          RNG seed for prompt + model init.
        warmup_runs:    Discarded runs before measurement (CPU warm-up, JIT).
    """
    prompt_length: int = 8
    n_new_values: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    n_repeats: int = 3
    seed: int = 42
    warmup_runs: int = 1


# ============================================================================
# 2. CACHE-AWARE MINI-LLM (extends the existing MiniLLM)
# ============================================================================

class CacheAwareMiniLLM(MiniLLM):
    """MiniLLM with an additional generate path that DISABLES the cache.

    Why subclass instead of modifying MiniLLM?
        - Open/Closed Principle: we add behavior without touching the
          original class, so existing tests keep working.
    """

    def generate_no_cache(self, prompt: np.ndarray, n_new: int,
                          strategy: str = "greedy",
                          temperature: float = 1.0) -> np.ndarray:
        """Generate n_new tokens WITHOUT using the KV cache.

        Algorithm:
            At every step, run a full forward pass over ALL tokens
            accumulated so far. This is the O(L^2) naive baseline.

        Args:
            prompt:     1-D int array of seed token IDs.
            n_new:      Number of new tokens to generate.
            strategy:   "greedy" or "temperature".
            temperature: Sampling temperature (only used if strategy="temperature").

        Returns:
            Full token sequence (prompt + generated) as a 1-D np.ndarray.
        """
        tokens = prompt.copy()
        # Reset any residual cache state.
        for blk in self.blocks:
            blk.attn.reset_cache()
        for _ in range(n_new):
            # Full forward pass over entire sequence so far.
            logits = self.forward(tokens, use_cache=False)
            last_logits = logits[-1]
            if strategy == "greedy":
                next_id = Sampler.greedy(last_logits)
            elif strategy == "temperature":
                next_id = Sampler.temperature(last_logits, temperature, self.rng)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            tokens = np.append(tokens, next_id)
        return tokens


# ============================================================================
# 3. BENCHMARK
# ============================================================================

class KVCacheBenchmark:
    """Times cached vs. no-cache generation across a sweep of n_new values."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.cfg_model = TransformerConfig()
        # Build a single model instance and reuse it for both modes
        # (fair comparison: same weights).
        self.model = CacheAwareMiniLLM(self.cfg_model)
        # Build a deterministic prompt.
        rng = np.random.default_rng(config.seed)
        self.prompt = rng.integers(0, self.cfg_model.vocab_size,
                                    size=config.prompt_length,
                                    dtype=np.int64)

    def _time_callable(self, callable_: Callable[[], np.ndarray]) -> float:
        """Run callable n_repeats times; return median wall-clock seconds."""
        # Warm-up runs (discarded) — stabilizes CPU caches & freq scaling.
        for _ in range(self.config.warmup_runs):
            callable_()
        times = []
        for _ in range(self.config.n_repeats):
            t0 = time.perf_counter()
            callable_()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return float(np.median(times))

    def run(self) -> dict:
        """Sweep n_new values; time both modes; return results dict."""
        results = {
            "n_new": list(self.config.n_new_values),
            "time_cached": [],
            "time_no_cache": [],
            "speedup": [],
        }
        for n in self.config.n_new_values:
            # Cached generation: O(L) per step.
            t_cached = self._time_callable(
                lambda n=n: self.model.generate(
                    self.prompt, n_new=n, strategy="greedy", use_cache=True
                )
            )
            # No-cache generation: O(L^2) per step.
            t_no_cache = self._time_callable(
                lambda n=n: self.model.generate_no_cache(
                    self.prompt, n_new=n, strategy="greedy"
                )
            )
            speedup = t_no_cache / t_cached if t_cached > 0 else float("inf")
            results["time_cached"].append(t_cached)
            results["time_no_cache"].append(t_no_cache)
            results["speedup"].append(speedup)
            print(f"n_new={n:>4d}  cached={t_cached*1e3:>8.2f} ms  "
                  f"no_cache={t_no_cache*1e3:>8.2f} ms  "
                  f"speedup={speedup:>5.2f}x")
        return results


# ============================================================================
# 4. PLOTTER
# ============================================================================

class BenchmarkPlotter:
    """Single responsibility: render benchmark results to a figure + PNG."""

    @staticmethod
    def plot(results: dict, save_path: str = "fig_kv_cache_speedup.png") -> None:
        """Two-panel plot: (top) timing curves, (bottom) speedup ratio."""
        n_new = results["n_new"]
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        # Panel 1: raw timings.
        axes[0].plot(n_new, np.array(results["time_cached"]) * 1e3,
                     "o-", color="seagreen", label="With KV Cache (O(L))")
        axes[0].plot(n_new, np.array(results["time_no_cache"]) * 1e3,
                     "s-", color="indianred", label="No Cache (O(L²))")
        axes[0].set_ylabel("Generation time (ms)")
        axes[0].set_title("KV Cache vs. No-Cache Generation Time")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Panel 2: speedup ratio.
        axes[1].bar(n_new, results["speedup"], color="steelblue",
                    label="Speedup (no_cache / cached)")
        axes[1].set_xlabel("Number of new tokens generated (n_new)")
        axes[1].set_ylabel("Speedup ratio (×)")
        axes[1].set_title("KV Cache Speedup Grows with Sequence Length")
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"\nFigure saved to: {save_path}")
        plt.show()


# ============================================================================
# 5. ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 2 — KV Cache Speedup Benchmark")
    print("=" * 70)
    cfg = BenchmarkConfig()
    bench = KVCacheBenchmark(cfg)
    results = bench.run()
    BenchmarkPlotter.plot(results)

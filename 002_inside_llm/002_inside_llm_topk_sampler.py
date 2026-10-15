"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

# Imagine your smart friend is guessing the next word, but instead of considering every word they ever learned (which could be 50,000 words!), they only let themselves think about their top 5 favorite guesses. They throw all the other words in the trash. Then, among those 5, they pick one based on how confident they are — but they're not allowed to peek at the trash. This is top-k sampling: keep only the k best guesses, ignore the rest, and pick from those. Why? Because if your friend considers ridiculous words like "purple banana truck" alongside reasonable guesses like "mat" or "floor", they might occasionally pick something silly. Top-k keeps the friend focused

Packages to install
pip3 install numpy>=1.24, pandas>=2.0, matplotlib>=3.7, scikit-learn>=1.3

Input
A 1-Dimensional NumPy array of raw logits (shape [vocab_size])
an integer k>=1, a positive temperature, and a seeded np.random.Generator



exercise_1_topk_sampler.py
==========================
Top-k sampling strategy for an LLM, plus a verification test suite.

Educational objectives:
    - Understand top-k sampling by implementing it from scratch.
    - Verify the edge case: k=1 must equal greedy argmax.
    - Verify the edge case: k=vocab_size must match temperature sampling.
    - Demonstrate renormalization after filtering.

Design principles:
    - DRY: reuses Softmax.forward from chapter3_llm_internals.
    - Single Responsibility: TopKSampler only samples;
      TopKSamplerTestSuite only verifies.
    - Open/Closed: TopKSampler is open for extension (e.g., add min-k filter)
      without modifying existing methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

# Reuse the production Softmax from the main module (DRY).
from 002_inside_llm import Softmax, TransformerConfig, MiniLLM, Sampler


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class TopKConfig:
    """Single source of truth for TopKSampler hyper-parameters.

    Attributes:
        k:           Number of highest-probability tokens to keep. Must be >= 1.
        temperature: Sharpness of the distribution. >0. Lower = sharper.
                     temperature=1.0 leaves logits unchanged.
        seed:        RNG seed for reproducible sampling.
    """
    k: int = 5
    temperature: float = 1.0
    seed: int = 42


# ============================================================================
# 2. TOP-K SAMPLER
# ============================================================================

class TopKSampler:
    """Sampler that keeps only the top-k tokens by probability, then samples.

    Algorithm:
        1. Scale logits by temperature: x' = logits / T.
        2. Apply softmax to get probabilities.
        3. Sort token indices by descending probability.
        4. Keep only the first k indices; zero out the rest.
        5. Renormalize the kept probabilities to sum to 1.
        6. Sample one token index according to the renormalized distribution.

    Edge cases:
        - k <= 0                  -> ValueError
        - k > vocab_size           -> clamped to vocab_size (with a warning print)
        - temperature <= 0         -> ValueError (use greedy for T=0)
        - empty logits array       -> ValueError
    """

    def __init__(self, config: TopKConfig):
        self.config = config
        # Single RNG per sampler so behavior is reproducible & isolated.
        self.rng = np.random.default_rng(config.seed)

    def sample(self, logits: np.ndarray) -> int:
        """Sample one token ID from logits using top-k filtering.

        Args:
            logits: 1-D array of raw scores. Shape (vocab_size,).

        Returns:
            Integer token ID in [0, vocab_size).

        Raises:
            TypeError:  if logits is not an np.ndarray.
            ValueError: if logits is empty, k <= 0, or temperature <= 0.
        """
        # --- Input validation (fail loud, fail early) ---
        if not isinstance(logits, np.ndarray):
            raise TypeError(f"logits must be np.ndarray, got {type(logits)}")
        if logits.ndim != 1:
            raise ValueError(f"logits must be 1-D, got shape {logits.shape}")
        if logits.size == 0:
            raise ValueError("logits is empty")
        if self.config.k <= 0:
            raise ValueError(f"k must be >= 1, got {self.config.k}")
        if self.config.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0; use Sampler.greedy() for T=0"
            )

        vocab_size = logits.shape[0]
        k_eff = min(self.config.k, vocab_size)
        if k_eff < self.config.k:
            print(f"[TopK] Warning: k={self.config.k} > vocab_size={vocab_size}; "
                  f"clamping to {k_eff}.")

        # --- Step 1: temperature scaling ---
        scaled = logits.astype(np.float32) / self.config.temperature

        # --- Step 2: softmax -> probabilities ---
        probs = Softmax.forward(scaled)              # reuse existing class (DRY)

        # --- Step 3: sort descending by probability ---
        # argsort is ascending, so [::-1] reverses to descending.
        sorted_idx = np.argsort(probs)[::-1]         # (vocab_size,)

        # --- Step 4: keep top-k indices ---
        kept_idx = sorted_idx[:k_eff]                # (k_eff,)

        # --- Step 5: renormalize ---
        kept_probs = probs[kept_idx].astype(np.float64)  # promote precision
        kept_probs = kept_probs / kept_probs.sum()

        # Sanity: probabilities must now sum to 1.
        assert abs(kept_probs.sum() - 1.0) < 1e-6, \
            f"Renormalization failed: sum={kept_probs.sum()}"

        # --- Step 6: sample ---
        return int(self.rng.choice(kept_idx, p=kept_probs))


# ============================================================================
# 3. TEST SUITE
# ============================================================================

class TopKSamplerTestSuite:
    """Verifies TopKSampler against known edge cases and reference distributions."""

    @staticmethod
    def _build_logits(seed: int = 0, vocab_size: int = 20) -> np.ndarray:
        """Generate deterministic logits for tests."""
        rng = np.random.default_rng(seed)
        return rng.standard_normal(vocab_size).astype(np.float32) * 3

    @classmethod
    def test_k_equals_one_matches_greedy(cls) -> dict:
        """When k=1, top-k must always pick argmax (same as greedy)."""
        logits = cls._build_logits(seed=1)
        config = TopKConfig(k=1, temperature=1.0, seed=99)
        sampler = TopKSampler(config)
        # Sample many times; result should be identical every time.
        samples = [sampler.sample(logits) for _ in range(50)]
        greedy = int(np.argmax(logits))
        unique_samples = set(samples)
        return {
            "test": "k=1 matches greedy",
            "greedy_token": greedy,
            "unique_topk_samples": list(unique_samples),
            "passed": unique_samples == {greedy},
        }

    @classmethod
    def test_k_equals_vocab_matches_temperature_distribution(cls) -> dict:
        """When k=vocab_size, top-k distribution should match temperature sampling."""
        logits = cls._build_logits(seed=2)
        k = logits.shape[0]
        # Two samplers with the same seed sequence for fair comparison.
        topk = TopKSampler(TopKConfig(k=k, temperature=1.0, seed=7))
        temp = Sampler()  # static methods; need an RNG
        rng_temp = np.random.default_rng(7)
        n_trials = 3000
        topk_counts = np.zeros(k, dtype=np.int64)
        temp_counts = np.zeros(k, dtype=np.int64)
        for _ in range(n_trials):
            topk_counts[topk.sample(logits)] += 1
            temp_counts[
                temp.temperature(logits, temperature=1.0, rng=rng_temp)
            ] += 1
        # Compare empirical distributions.
        p_topk = topk_counts / n_trials
        p_temp = temp_counts / n_trials
        max_diff = float(np.max(np.abs(p_topk - p_temp)))
        return {
            "test": "k=vocab_size matches temperature sampling",
            "max_abs_distribution_diff": max_diff,
            "tolerance": 0.05,
            "passed": max_diff < 0.05,
        }

    @classmethod
    def test_renormalization_sums_to_one(cls) -> dict:
        """After filtering, kept probabilities must sum to exactly 1.0."""
        logits = cls._build_logits(seed=3)
        config = TopKConfig(k=5, temperature=1.0, seed=11)
        sampler = TopKSampler(config)
        # Sample to ensure renormalization path runs without error.
        _ = sampler.sample(logits)
        # Manually reconstruct kept probs to inspect.
        probs = Softmax.forward(logits)
        kept_idx = np.argsort(probs)[::-1][:config.k]
        kept_probs = probs[kept_idx] / probs[kept_idx].sum()
        return {
            "test": "renormalization sum",
            "kept_probs_sum": float(kept_probs.sum()),
            "passed": abs(kept_probs.sum() - 1.0) < 1e-6,
        }

    @classmethod
    def test_invalid_inputs_raise(cls) -> dict:
        """Empty logits, k<=0, and temperature<=0 must raise ValueError."""
        results = {}
        # Empty logits
        try:
            TopKSampler(TopKConfig(k=3)).sample(np.array([], dtype=np.float32))
            results["empty_logits"] = "FAILED: did not raise"
        except ValueError:
            results["empty_logits"] = "PASSED"
        # k <= 0
        try:
            TopKSampler(TopKConfig(k=0)).sample(cls._build_logits())
            results["k_zero"] = "FAILED: did not raise"
        except ValueError:
            results["k_zero"] = "PASSED"
        # temperature <= 0
        try:
            TopKSampler(TopKConfig(k=3, temperature=0.0)).sample(cls._build_logits())
            results["temp_zero"] = "FAILED: did not raise"
        except ValueError:
            results["temp_zero"] = "PASSED"
        return {"test": "invalid inputs raise", "results": results,
                "passed": all(v == "PASSED" for v in results.values())}

    @classmethod
    def run_all(cls) -> None:
        """Run every test and pretty-print results."""
        print("\n" + "=" * 70)
        print("TopKSampler Test Suite")
        print("=" * 70)
        for method in [
            cls.test_k_equals_one_matches_greedy,
            cls.test_k_equals_vocab_matches_temperature_distribution,
            cls.test_renormalization_sums_to_one,
            cls.test_invalid_inputs_raise,
        ]:
            result = method()
            status = "✓ PASS" if result.get("passed") else "✗ FAIL"
            print(f"\n[{status}] {result.pop('test')}")
            for k, v in result.items():
                print(f"    {k}: {v}")


# ============================================================================
# 4. DEMO: TOP-K INSIDE A REAL GENERATION
# ============================================================================

def demo_topk_in_generation() -> None:
    """Show top-k generation in a MiniLLM and compare to greedy."""
    print("\n" + "=" * 70)
    print("Demo: Top-k vs Greedy Generation")
    print("=" * 70)
    cfg = TransformerConfig()
    model = MiniLLM(cfg)
    prompt = np.array([2, 7, 1, 4], dtype=np.int64)
    print(f"Prompt: {prompt.tolist()}")

    # Greedy generation
    out_greedy = model.generate(prompt, n_new=8, strategy="greedy", use_cache=True)
    print(f"Greedy output:   {out_greedy.tolist()}")

    # Top-k generation: we sample by reusing the LM's logits at each step.
    # We monkey-patch the strategy here for demonstration purposes.
    # In production you would add "top_k" as a strategy in MiniLLM.generate.
    print("\nTop-k (k=3) sampled outputs (3 trials):")
    for trial in range(3):
        # Generate using top-k manually.
        tokens = prompt.copy()
        for blk in model.blocks:
            blk.attn.reset_cache()
        rng = np.random.default_rng(100 + trial)
        sampler = TopKSampler(TopKConfig(k=3, temperature=1.0, seed=100 + trial))
        # Prime cache with prompt.
        logits = model.forward(tokens, use_cache=True)
        last = logits[-1]
        for _ in range(8):
            tok = sampler.sample(last)
            tokens = np.append(tokens, tok)
            new_logits = model.forward(np.array([tok], dtype=np.int64),
                                         use_cache=True)
            last = new_logits[-1]
        print(f"  Trial {trial + 1}: {tokens.tolist()}")


# ============================================================================
# 5. ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    TopKSamplerTestSuite.run_all()
    demo_topk_in_generation()

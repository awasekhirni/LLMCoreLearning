"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
synthetic_data_generation.py
=============================

Learning module for "Synthetic Data Generation (Textbook-Quality & Self-Instruct)"
from the AI curriculum technical card.

Concepts covered
----------------
- Seed generation (topic × difficulty × format)
- Teacher LLM interface (abstract) + MockTeacherLLM (deterministic, no API needed)
  + RealTeacherLLM (Anthropic scaffold, uses the technical card's snippet pattern)
- Filtering pipeline: heuristic filter + mock LLM-as-judge + MinHash dedup
- Benchmark metrics: quality pass rate, downstream delta proxy, n-gram entropy
- Visualisation: seeds, funnel, pass rate, diversity

Design principles
-----------------
- a single `SyntheticDataPipeline.run()` orchestrates seed → generate → filter → measure.
- Dataclasses: every stateful concept (`GenerationConfig`, `Seed`, `SyntheticExample`,
  `FilterResult`, `PipelineResult`) is a `@dataclass`.
- Full documentation: module, class, and method docstrings; comments explain *why*.

Usage
-----
    python3 synthetic_data_generation.py

Saves figures `fig_sdg_1.png`, `fig_sdg_2.png`, `fig_sdg_3.png`, `fig_sdg_4.png`
in the current working directory.

Dependencies
------------
    pip3 install numpy pandas matplotlib scikit-learn
Optional (only for RealTeacherLLM):
    pip3 install anthropic
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import hashlib
import math
import random
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------
# Third-party
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Fixed seeds for full reproducibility
# ----------------------------------------------------------------------
random.seed(42)
np.random.seed(42)


# ======================================================================
# CONSTANTS — seed catalog (source of truth for generation)
# ======================================================================

# Each seed is a (topic, difficulty) tuple. 40 seeds is enough to exercise
# the pipeline without being unwieldy. Format is chosen by the generator.
SEED_CATALOG: List[Tuple[str, str]] = [
    ("algebra",        "easy"),   ("algebra",        "medium"),
    ("algebra",        "hard"),   ("geometry",       "easy"),
    ("geometry",       "medium"), ("geometry",       "hard"),
    ("probability",    "easy"),   ("probability",    "medium"),
    ("probability",    "hard"),   ("calculus",       "easy"),
    ("calculus",       "medium"), ("calculus",       "hard"),
    ("arrays",         "easy"),   ("arrays",         "medium"),
    ("arrays",         "hard"),   ("sorting",        "easy"),
    ("sorting",        "medium"), ("sorting",        "hard"),
    ("strings",        "easy"),   ("strings",        "medium"),
    ("graphs",         "medium"), ("graphs",         "hard"),
    ("dynamic_prog",   "medium"), ("dynamic_prog",   "hard"),
    ("recursion",      "easy"),   ("recursion",      "medium"),
    ("recursion",      "hard"),   ("fractions",      "easy"),
    ("fractions",      "medium"), ("decimals",       "easy"),
    ("decimals",       "medium"), ("word_problems",  "easy"),
    ("word_problems",  "medium"), ("word_problems",  "hard"),
    ("number_theory",  "medium"), ("number_theory",  "hard"),
    ("statistics",     "easy"),   ("statistics",     "medium"),
    ("logic",          "easy"),   ("logic",          "hard"),
]


# ======================================================================
# DATACLASSES — all stateful concepts
# ======================================================================

@dataclass
class GenerationConfig:
    """
    All hyperparameters of the pipeline, matching the technical card's
    `key_hyperparameters`:

    - teacher_temperature   → diversity of teacher output
    - seed_prompt_count     → how many seeds drive generation
    - filtering_strictness  → [0,1] threshold for the heuristic + judge
    - dedup_aggressiveness  → [0,1] Jaccard similarity threshold above which
                              two examples are considered duplicates
    """
    teacher_temperature: float = 0.9
    seed_prompt_count: int = 40
    filtering_strictness: float = 0.70
    dedup_aggressiveness: float = 0.80
    max_examples_per_seed: int = 3
    random_seed: int = 42

    def __post_init__(self) -> None:
        # Fail fast on obviously invalid config — this is an input-validation step.
        if not 0.0 <= self.teacher_temperature <= 2.0:
            raise ValueError("teacher_temperature must be in [0, 2]")
        if not 0.0 <= self.filtering_strictness <= 1.0:
            raise ValueError("filtering_strictness must be in [0, 1]")
        if not 0.0 <= self.dedup_aggressiveness <= 1.0:
            raise ValueError("dedup_aggressiveness must be in [0, 1]")
        if self.seed_prompt_count <= 0:
            raise ValueError("seed_prompt_count must be > 0")


@dataclass
class Seed:
    """A seed is a (topic, difficulty) pair — the generator's launchpad."""
    seed_id: int
    topic: str
    difficulty: str


@dataclass
class SyntheticExample:
    """
    One generated example.

    Fields mirror the technical card's `input_dataset` / `outcome`:
    - problem   : the instruction / question shown to the student
    - reasoning : a chain-of-thought trace (for reasoning-trace mode)
    - answer    : the final answer (numeric for math, code for code)
    - is_code   : True if the answer is code (needs unit-test verification)
    """
    seed: Seed
    problem: str
    reasoning: str
    answer: str
    is_code: bool = False

    def as_text(self) -> str:
        """String form used for dedup and diversity metrics."""
        return f"{self.problem}\n{self.reasoning}\n{self.answer}"


@dataclass
class FilterResult:
    """Outcome of filtering a single example. `passed` is the final decision."""
    passed: bool
    reasons: List[str] = field(default_factory=list)
    quality_score: float = 0.0       # composite score, higher is better
    is_duplicate: bool = False


@dataclass
class PipelineResult:
    """Everything the pipeline produced, ready for metrics + plots."""
    seeds: List[Seed]
    examples: List[SyntheticExample]
    filter_results: List[FilterResult]
    kept: List[SyntheticExample]
    rejected: List[SyntheticExample]
    elapsed_seconds: float

    @property
    def pass_rate(self) -> float:
        """Fraction of generated examples that survived filtering."""
        return len(self.kept) / len(self.examples) if self.examples else 0.0


# ======================================================================
# TEACHER LLM — abstract interface + mock + real scaffold
# ======================================================================

class TeacherLLM(ABC):
    """
    Abstract teacher: any object that can generate a synthetic example
    given a seed. Kept narrow so real implementations (Anthropic, OpenAI,
    local vLLM) all fit — swap one in without touching the pipeline.
    """
    @abstractmethod
    def generate(self, seed: Seed, temperature: float) -> Optional[SyntheticExample]:
        """Return a SyntheticExample, or None if generation failed."""
        raise NotImplementedError


class MockTeacherLLM(TeacherLLM):
    """
    Deterministic, template-based teacher. Exercises the *same* pipeline as
    a real teacher (parsing, formatting, reasoning) but with zero API cost.

    Why a mock? Real teachers are non-deterministic and cost money. This
    module must be runnable by a learner with no key — and results must be
    reproducible. The mock teacher intentionally simulates *imperfections*:
    ~10% of outputs omit "Answer:", ~5% have a wrong-type answer, ~5% are
    near-duplicates of an earlier output. This makes the filtering pipeline
    meaningful — a pipeline that only works on perfect outputs is useless.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        # counters for tracking simulated imperfections
        self._n_generated = 0
        self._n_dupe_hint = 0

    def generate(self, seed: Seed, temperature: float) -> Optional[SyntheticExample]:
        self._n_generated += 1

        # ~10%: teacher emits a free-text blob (no structured Answer) — bad example
        if self.rng.random() < 0.10:
            return SyntheticExample(
                seed=seed,
                problem=f"Discuss {seed.topic} at {seed.difficulty} difficulty.",
                reasoning="(no reasoning trace)",
                answer="",                     # empty — filter will reject
                is_code=False,
            )

        # ~5%: wrong-type answer (word instead of number for math)
        wrong_type = self.rng.random() < 0.05

        # Topic-appropriate templates
        if seed.topic in {
            "algebra", "geometry", "probability", "calculus", "fractions",
            "decimals", "word_problems", "number_theory", "statistics", "logic",
        }:
            example = self._make_math(seed, wrong_type)
        else:
            example = self._make_code(seed)

        # ~5%: near-duplicate — simulate model collapse by mutating
        if self.rng.random() < 0.05 and example is not None:
            example = SyntheticExample(
                seed=seed,
                problem=example.problem + " ",   # single-char change
                reasoning=example.reasoning,
                answer=example.answer,
                is_code=example.is_code,
            )
            self._n_dupe_hint += 1

        return example

    # -------- private helpers: templated generation --------------------
    def _make_math(self, seed: Seed, wrong_type: bool) -> SyntheticExample:
        """Generate a templated math problem with chain-of-thought reasoning."""
        a = self.rng.randint(1, 20)
        b = self.rng.randint(1, 10)
        if seed.difficulty == "easy":
            problem = f"What is {a} + {b}?"
            answer = a + b
        elif seed.difficulty == "medium":
            problem = f"What is ({a} + {b}) * {a}?"
            answer = (a + b) * a
        else:
            problem = f"Solve {a}x + {b} = {a*a + b} for x."
            answer = a
        reasoning = (
            f"Step 1: identify the operation(s) in '{seed.topic}'.\n"
            f"Step 2: compute using {'addition' if seed.difficulty=='easy' else 'combined operations'}.\n"
            f"Step 3: verify the result is an integer."
        )
        ans_str = "eight" if wrong_type else str(answer)
        return SyntheticExample(
            seed=seed, problem=problem,
            reasoning=reasoning, answer=ans_str, is_code=False,
        )

    def _make_code(self, seed: Seed) -> SyntheticExample:
        """Generate a small, runnable code snippet as the 'answer'."""
        n = self.rng.randint(2, 8)
        if seed.topic == "arrays":
            problem = f"Write a function that returns a list of {n} zeros."
            answer = f"def make_zeros():\n    return [0] * {n}"
        elif seed.topic == "sorting":
            problem = f"Write a function that sorts a list of {n} integers ascending."
            answer = "def sort_list(xs):\n    return sorted(xs)"
        elif seed.topic == "strings":
            problem = "Write a function that reverses a string."
            answer = "def reverse(s):\n    return s[::-1]"
        elif seed.topic == "graphs":
            problem = "Write a function that returns the number of nodes in an adjacency list."
            answer = "def num_nodes(g):\n    return len(g)"
        elif seed.topic == "dynamic_prog":
            problem = f"Write a function that returns the {n}-th Fibonacci number."
            answer = (
                "def fib(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a"
            )
        elif seed.topic == "recursion":
            problem = "Write a recursive factorial function."
            answer = "def fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)"
        else:
            problem = f"Write a simple function about {seed.topic}."
            answer = "def noop():\n    return None"
        return SyntheticExample(
            seed=seed, problem=problem,
            reasoning="Consider the signature, then the simplest correct body.",
            answer=answer, is_code=True,
        )


class RealTeacherLLM(TeacherLLM):
    """
    Scaffold for a real teacher (Anthropic Claude), matching the technical
    card's `code_snippet`. Left un-executed by default so the module runs
    without an API key. To use, install `anthropic` and set ANTHROPIC_API_KEY.
    """

    def __init__(self, model: str = "claude-3-5-sonnet-20241022") -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:                    # pragma: no cover
            raise ImportError(
                "pip install anthropic  # to use RealTeacherLLM"
            ) from exc
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model

    def generate(self, seed: Seed, temperature: float) -> Optional[SyntheticExample]:
        prompt = (
            f"You are generating training data for a math/code tutor.\n"
            f"Topic: {seed.topic}. Difficulty: {seed.difficulty}.\n"
            f"Respond EXACTLY in the format:\n"
            f"Problem: <one-line problem>\n"
            f"Reasoning: <2-4 step chain of thought>\n"
            f"Answer: <final answer only>"
        )
        try:
            r = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = r.content[0].text
        except Exception:                             # noqa: BLE001
            return None

        parsed = parse_teacher_output(text)
        if parsed is None:
            return None
        problem, reasoning, answer = parsed
        is_code = seed.topic in {"arrays", "sorting", "strings", "graphs",
                                 "dynamic_prog", "recursion"}
        return SyntheticExample(seed, problem, reasoning, answer, is_code=is_code)


def parse_teacher_output(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse 'Problem: ... / Reasoning: ... / Answer: ...' from free text.
    Returns None if any section is missing (an essential filter step —
    malformed teacher output must be discarded, not passed downstream).
    """
    m = re.search(
        r"Problem:\s*(.+?)\s*Reasoning:\s*(.+?)\s*Answer:\s*(.+?)\s*$",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


# ======================================================================
# FILTERING — heuristic + LLM-as-judge + MinHash dedup
# ======================================================================

@dataclass
class HeuristicFilter:
    """
    Cheap, deterministic checks that catch the most common failure modes:

    - empty / too-short / too-long fields
    - missing Answer: section
    - math answers that are not numeric
    - code answers that do not `def` a function
    """
    min_problem_len: int = 10
    max_problem_len: int = 400
    min_reasoning_len: int = 10

    def check(self, ex: SyntheticExample) -> FilterResult:
        reasons: List[str] = []
        if len(ex.problem) < self.min_problem_len:
            reasons.append("problem_too_short")
        if len(ex.problem) > self.max_problem_len:
            reasons.append("problem_too_long")
        if len(ex.reasoning) < self.min_reasoning_len:
            reasons.append("reasoning_too_short")
        if not ex.answer.strip():
            reasons.append("empty_answer")
        if not ex.is_code and ex.answer:
            # numeric answer check for math (allows -3, 3.14, 1/2)
            if not re.fullmatch(r"-?\d+(?:\.\d+)?(?:/\d+)?", ex.answer.strip()):
                reasons.append("math_answer_not_numeric")
        if ex.is_code and "def " not in ex.answer:
            reasons.append("code_answer_missing_def")
        return FilterResult(passed=not reasons, reasons=reasons, quality_score=1.0 if not reasons else 0.0)


@dataclass
class MockJudge:
    """
    Mock LLM-as-judge. Assigns a [0,1] quality score based on problem
    length, reasoning richness, and answer format. Real judges would call
    a strong LLM; the pipeline's interface is identical.

    A score >= `config.filtering_strictness` is required to pass.
    """
    rng: random.Random = field(default_factory=lambda: random.Random(123))
    # base score is 0.6 plus up to 0.4 from measurable quality signals
    base: float = 0.60

    def score(self, ex: SyntheticExample) -> float:
        if not ex.problem or not ex.answer:
            return 0.0
        s = self.base
        # length: 20-200 chars → up to +0.15
        s += min(0.15, len(ex.problem) / 200.0 * 0.15)
        # reasoning: each distinct sentence adds 0.03, capped at +0.15
        sents = re.split(r"[.!?]\s+", ex.reasoning)
        s += min(0.15, len(sents) * 0.03)
        # code answers get a small bonus if they contain `return`
        if ex.is_code and "return" in ex.answer:
            s += 0.05
        # small deterministic jitter so scores are not perfectly discrete
        s += self.rng.uniform(0.0, 0.05)
        return round(min(1.0, s), 4)


def _shingles(text: str, n: int = 3) -> set:
    """Word-level n-gram shingles — input to the MinHash sketch."""
    words = text.lower().split()
    if len(words) < n:
        return set(words)
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _minhash_signature(shingles: set, num_perm: int = 64, seed: int = 0) -> List[int]:
    """
    Classic MinHash: for each of `num_perm` hash permutations, take the
    minimum hash of the shingle set. Jaccard similarity of two sets is
    estimated by the fraction of matching signature positions.
    """
    if not shingles:
        return [0] * num_perm
    sig: List[int] = []
    for i in range(num_perm):
        h = min(
            int(hashlib.md5(f"{seed}:{i}:{s!r}".encode()).hexdigest(), 16) % (2**32)
            for s in shingles
        )
        sig.append(h)
    return sig


def estimate_jaccard(a: List[int], b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


@dataclass
class MinHashDeduplicator:
    """
    Greedy dedup: iterate examples in order; drop an example if its
    estimated Jaccard similarity to any *previously kept* example exceeds
    `threshold`. O(N × kept) but adequate for educational scale.

    `threshold` is derived from `config.dedup_aggressiveness` (higher →
    more aggressive). Examples with similarity > threshold are duplicates.
    """
    threshold: float = 0.80
    num_perm: int = 64

    def dedup(self, examples: Sequence[SyntheticExample]) -> List[bool]:
        """Return a boolean mask: True = duplicate (should be dropped)."""
        signatures: List[List[int]] = []
        duplicate_mask: List[bool] = []
        for ex in examples:
            sig = _minhash_signature(_shingles(ex.as_text()), self.num_perm)
            is_dup = any(
                estimate_jaccard(sig, prev) > self.threshold
                for prev in signatures
            )
            duplicate_mask.append(is_dup)
            if not is_dup:
                signatures.append(sig)
        return duplicate_mask


# ======================================================================
# METRICS — quality pass rate, diversity (n-gram entropy), downstream proxy
# ======================================================================

def ngram_entropy(texts: Sequence[str], n: int = 3) -> float:
    """
    Shannon entropy of the word n-gram distribution.
    Higher = more diverse output (technical card's `benchmark_metrics`).

    A collapsed teacher produces a few repeated n-grams → low entropy.
    """
    counter: Counter = Counter()
    total = 0
    for t in texts:
        words = t.lower().split()
        for i in range(max(0, len(words) - n + 1)):
            counter[tuple(words[i:i + n])] += 1
            total += 1
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total) for c in counter.values() if c > 0
    )


@dataclass
class DownstreamProxy:
    """
    A *proxy* for the card's "downstream delta vs no-synthetic baseline".
    A real downstream eval requires training a model; here we compute a
    tiny supervised classifier accuracy using a hashing-vectorizer + logistic
    regression on problem→topic classification.

    Two training sets are compared:
      1. "no-synthetic": a small held-out chunk (held from generation) — a stand-in
         for scarce human data. We simulate this with a fixed small subset.
      2. "synthetic+": the filtered pipeline output.

    The delta is the accuracy improvement — positive means synthetic data
    helped the classifier learn topic structure. This is a *pedagogical proxy*,
    NOT a claim about LLM downstream performance.
    """
    test_frac: float = 0.25

    def evaluate(self, examples: Sequence[SyntheticExample]) -> Dict[str, float]:
        if len(examples) < 8:
            return {"baseline_acc": float("nan"),
                    "synthetic_acc": float("nan"),
                    "delta": float("nan")}
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import LogisticRegression

        rng = random.Random(42)
        texts = [ex.problem for ex in examples]
        labels = [ex.seed.topic for ex in examples]

        idx = list(range(len(texts)))
        rng.shuffle(idx)
        n_test = max(1, int(len(texts) * self.test_frac))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        # Baseline: use only the first ~half of the training examples (simulates scarcity)
        n_base = max(4, len(train_idx) // 2)
        base_train = train_idx[:n_base]

        vec = HashingVectorizer(n_features=2**12, alternate_sign=False)
        X = vec.fit_transform(texts)
        clf_base = LogisticRegression(max_iter=200).fit(X[base_train], [labels[i] for i in base_train])
        clf_syn  = LogisticRegression(max_iter=200).fit(X[train_idx],  [labels[i] for i in train_idx])
        y_test   = [labels[i] for i in test_idx]

        base_acc = (clf_base.predict(X[test_idx]) == np.array(y_test)).mean()
        syn_acc  = (clf_syn .predict(X[test_idx]) == np.array(y_test)).mean()
        return {
            "baseline_acc": float(base_acc),
            "synthetic_acc": float(syn_acc),
            "delta": float(syn_acc - base_acc),
        }


# ======================================================================
# PIPELINE — the DRY orchestrator
# ======================================================================

@dataclass
class SyntheticDataPipeline:
    """
    End-to-end pipeline: seed → generate → filter → measure.

    The three stages mirror the technical card's `implementation_guide`:
      1. Generate diverse seeds (high temperature).
      2. Generate actual data with structured prompts.
      3. Filter rigorously (heuristics + LLM-as-judge + dedup).

    Everything is stored in `PipelineResult` so the caller can inspect
    every intermediate artifact — no hidden state.
    """
    teacher: TeacherLLM
    config: GenerationConfig

    def _make_seeds(self) -> List[Seed]:
        """Sample `seed_prompt_count` distinct (topic, difficulty) seeds."""
        catalog = list(SEED_CATALOG)
        self.config_rng = random.Random(self.config.random_seed)
        self.config_rng.shuffle(catalog)
        n = min(self.config.seed_prompt_count, len(catalog))
        return [
            Seed(seed_id=i, topic=t, difficulty=d)
            for i, (t, d) in enumerate(catalog[:n])
        ]

    def _generate(self, seeds: List[Seed]) -> List[SyntheticExample]:
        """Generate `max_examples_per_seed` examples per seed; drop failures."""
        out: List[SyntheticExample] = []
        for seed in seeds:
            for _ in range(self.config.max_examples_per_seed):
                ex = self.teacher.generate(seed, self.config.teacher_temperature)
                if ex is not None:
                    out.append(ex)
        return out

    def _filter(
        self, examples: List[SyntheticExample]
    ) -> Tuple[List[SyntheticExample], List[SyntheticExample], List[FilterResult]]:
        """
        Combined filtering: heuristics → judge → dedup.

        Order matters: run cheap heuristics first (rejects obvious garbage),
        then judge (more expensive), then dedup (O(N × kept)).
        """
        heur = HeuristicFilter()
        judge = MockJudge()
        results: List[FilterResult] = []

        # Stage A — heuristics
        for ex in examples:
            results.append(heur.check(ex))

        # Stage B — judge (skip examples already rejected by heuristics)
        for i, ex in enumerate(examples):
            if not results[i].passed:
                continue
            q = judge.score(ex)
            results[i].quality_score = q
            if q < self.config.filtering_strictness:
                results[i].passed = False
                results[i].reasons.append(f"judge_score_{q:.2f}_below_threshold")

        # Stage C — dedup (only among examples still passing)
        candidates = [(i, ex) for i, (ex, r) in enumerate(zip(examples, results)) if r.passed]
        if candidates:
            deduper = MinHashDeduplicator(
                threshold=self.config.dedup_aggressiveness
            )
            dup_mask = deduper.dedup([ex for _, ex in candidates])
            for (i, _), is_dup in zip(candidates, dup_mask):
                if is_dup:
                    results[i].passed = False
                    results[i].is_duplicate = True
                    results[i].reasons.append("duplicate")

        kept = [ex for ex, r in zip(examples, results) if r.passed]
        rejected = [ex for ex, r in zip(examples, results) if not r.passed]
        return kept, rejected, results

    def run(self) -> PipelineResult:
        """Execute the full pipeline and return all artefacts."""
        t0 = time.perf_counter()
        seeds = self._make_seeds()
        examples = self._generate(seeds)
        kept, rejected, results = self._filter(examples)
        return PipelineResult(
            seeds=seeds, examples=examples, filter_results=results,
            kept=kept, rejected=rejected,
            elapsed_seconds=time.perf_counter() - t0,
        )


# ======================================================================
# VISUALISATION
# ======================================================================

@dataclass
class PipelineVisualizer:
    """Four-panel visualisation of the pipeline's behaviour."""
    result: PipelineResult

    def plot_all(self, out_dir: Path = Path(".")) -> None:
        """Produce four figures and display them."""
        self._plot_seeds(out_dir / "fig_sdg_1.png")
        self._plot_funnel(out_dir / "fig_sdg_2.png")
        self._plot_pass_rate_by_topic(out_dir / "fig_sdg_3.png")
        self._plot_diversity(out_dir / "fig_sdg_4.png")

    # ---- individual figures ------------------------------------------
    def _plot_seeds(self, path: Path) -> None:
        """(a) Distribution of generated seeds by topic — shows input diversity."""
        topics = [s.topic for s in self.result.seeds]
        counts = Counter(topics)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.bar(list(counts.keys()), list(counts.values()), color="steelblue")
        ax.set_title("(a) Seeds generated — input diversity")
        ax.set_xlabel("Topic")
        ax.set_ylabel("Number of seeds")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.show()
        print(f"[Viz] saved {path.resolve()}")

    def _plot_funnel(self, path: Path) -> None:
        """(b) Filtering funnel — shows how many examples survive each stage."""
        n_total = len(self.result.examples)
        n_after_heur = sum(1 for r in self.result.filter_results
                           if not any(x.startswith("judge_") or x == "duplicate"
                                      for x in r.reasons))
        n_after_judge = sum(1 for r in self.result.filter_results
                            if not r.is_duplicate and r.passed) + \
                        sum(1 for r in self.result.filter_results if r.is_duplicate)
        n_kept = len(self.result.kept)
        stages = ["Generated", "After heuristics", "After judge", "After dedup"]
        counts = [n_total, n_after_heur, n_after_judge, n_kept]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.barh(stages, counts, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"])
        for b, c in zip(bars, counts):
            ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                    f"{c}", va="center")
        ax.invert_yaxis()
        ax.set_title("(b) Filtering funnel — examples surviving each stage")
        ax.set_xlabel("Number of examples")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.show()
        print(f"[Viz] saved {path.resolve()}")

    def _plot_pass_rate_by_topic(self, path: Path) -> None:
        """(c) Pass rate per topic — reveals topic-level quality issues."""
        topics = sorted({ex.seed.topic for ex in self.result.examples})
        kept_by_topic = Counter(ex.seed.topic for ex in self.result.kept)
        total_by_topic = Counter(ex.seed.topic for ex in self.result.examples)
        rates = [kept_by_topic[t] / total_by_topic[t] for t in topics]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.bar(topics, rates, color="seagreen")
        ax.set_ylim(0, 1.05)
        ax.set_title("(c) Quality pass rate by topic")
        ax.set_xlabel("Topic")
        ax.set_ylabel("Pass rate")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.show()
        print(f"[Viz] saved {path.resolve()}")

    def _plot_diversity(self, path: Path) -> None:
        """
        (d) Diversity (n-gram entropy) over successive generation batches.
        Simulates the technical card's `diversity metrics (n-gram entropy)`.
        """
        texts = [ex.as_text() for ex in self.result.examples]
        # Entropy in cumulative batches of 20
        batch = 20
        xs, ys = [], []
        for i in range(batch, len(texts) + 1, batch):
            xs.append(i)
            ys.append(ngram_entropy(texts[:i], n=3))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(xs, ys, marker="o", color="darkorange")
        ax.set_title("(d) Diversity (3-gram entropy) over cumulative generation")
        ax.set_xlabel("Number of examples generated")
        ax.set_ylabel("3-gram Shannon entropy (bits)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.show()
        print(f"[Viz] saved {path.resolve()}")


# ======================================================================
# ELI5 WALKTHROUGH — simplified, self-contained block
# ======================================================================

def eli5_walkthrough() -> None:
    """
    A plain-language, toy walkthrough that demonstrates the pipeline.
    Prints every step in a way a five-year-old could follow.
    """
    print("\n" + "=" * 70)
    print(" ELI5 WALKTHROUGH — how synthetic data is made and cleaned")
    print("=" * 70)

    print("\n[Step 1] A teacher writes some homework problems...")
    tiny_seeds = [
        Seed(seed_id=0, topic="algebra", difficulty="easy"),
        Seed(seed_id=1, topic="arrays",  difficulty="easy"),
    ]
    teacher = MockTeacherLLM(seed=7)
    drafts: List[SyntheticExample] = []
    for s in tiny_seeds:
        for _ in range(2):
            ex = teacher.generate(s, temperature=0.9)
            if ex is not None:
                drafts.append(ex)
                print(f"   → wrote: {ex.problem!r}")

    print("\n[Step 2] We check each problem — is it complete and clear?")
    heur = HeuristicFilter()
    survivors: List[SyntheticExample] = []
    for ex in drafts:
        r = heur.check(ex)
        if r.passed:
            print(f"   ✓ kept:    {ex.problem!r}")
            survivors.append(ex)
        else:
            print(f"   ✗ dropped: {ex.problem!r}  (because {r.reasons})")

    print("\n[Step 3] We ask a judge — is each problem high enough quality?")
    judge = MockJudge()
    final: List[SyntheticExample] = []
    for ex in survivors:
        q = judge.score(ex)
        if q >= 0.70:
            print(f"   ✓ kept:    score={q:.2f}")
            final.append(ex)
        else:
            print(f"   ✗ dropped: score={q:.2f} below 0.70")

    print("\n[Step 4] We look for near-duplicate problems and drop them.")
    if final:
        deduper = MinHashDeduplicator(threshold=0.80)
        dup_mask = deduper.dedup(final)
        deduped = [ex for ex, is_dup in zip(final, dup_mask) if not is_dup]
        print(f"   kept {len(deduped)} of {len(final)} after dedup")

    print("\n[Step 5] Done! We have a clean, textbook-quality mini-dataset.")
    print(f"   final size = {len(final)} examples\n")


# ======================================================================
# DRIVER
# ======================================================================

def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    # ---------- ELI5 walkthrough -------------------------------------
    eli5_walkthrough()

    # ---------- Full pipeline ----------------------------------------
    print("=" * 70)
    print(" FULL PIPELINE — Seed → Generate → Filter → Measure")
    print("=" * 70)

    config = GenerationConfig(
        teacher_temperature=0.9,
        seed_prompt_count=40,
        filtering_strictness=0.70,
        dedup_aggressiveness=0.80,
        max_examples_per_seed=3,
        random_seed=42,
    )
    pipeline = SyntheticDataPipeline(teacher=MockTeacherLLM(seed=42), config=config)
    result = pipeline.run()

    print(f"\nGenerated: {len(result.examples)} examples in {result.elapsed_seconds:.2f}s")
    print(f"Kept:      {len(result.kept)}")
    print(f"Rejected:  {len(result.rejected)}")
    print(f"Pass rate: {result.pass_rate:.1%}")

    # ---------- Benchmark metrics -----------------------------------
    print("\n--- Benchmark metrics (technical card's benchmark_metrics) ---")
    div = ngram_entropy([ex.as_text() for ex in result.examples], n=3)
    print(f"Quality pass rate          : {result.pass_rate:.3f}")
    print(f"Diversity (3-gram entropy) : {div:.3f} bits")

    proxy = DownstreamProxy()
    delta = proxy.evaluate(result.kept)
    print(f"Downstream proxy baseline  : {delta['baseline_acc']:.3f}")
    print(f"Downstream proxy synthetic : {delta['synthetic_acc']:.3f}")
    print(f"Downstream delta           : {delta['delta']:+.3f}  "
          f"(positive = synthetic helped)")

    # ---------- Summary table ---------------------------------------
    reason_counts = Counter()
    for r in result.filter_results:
        for reason in r.reasons:
            # strip numeric suffixes for cleaner tallies
            key = reason.split("_below_threshold")[0]
            reason_counts[key] += 1
    df = pd.DataFrame(
        sorted(reason_counts.items(), key=lambda kv: -kv[1]),
        columns=["rejection reason", "count"],
    )
    print("\n--- Rejection reasons ---")
    print(df.to_string(index=False))

    # ---------- Visualisations --------------------------------------
    PipelineVisualizer(result).plot_all(Path("."))

    print("\nDone. Figures saved: fig_sdg_1.png .. fig_sdg_4.png")


if __name__ == "__main__":
    main()

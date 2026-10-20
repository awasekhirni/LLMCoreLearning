"""

# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
============================================================================
 data_mixture_curriculum.py
============================================================================
 Educational implementation of Data Mixture & Curriculum Learning for LLM
 pretraining.  Covers:
   - Temperature-controlled domain mixture: p_i ∝ n_i^T (T in 0.3-0.7 typical)
   - Empirical verification of the sampler via KL / chi-square
   - Two-stage curriculum scheduler (general -> specialized)
   - A tiny, deterministic char-bigram language model used as a *loss proxy*
     so mixture and curriculum effects are measurable on a laptop
   - End-to-end comparison and 4 publication-quality figures

 Purpose
 -------
 Teaching component for an AI curriculum.  Every class has a docstring and
 every non-obvious line is commented with "why".  No GPU, no network, no
 LLM training required — the point is to make the *statistics* of mixture
 sampling visible and testable.

 Design principles
 -----------------
 - DRY  — one `DomainMixtureSampler` powers flat, uniform, and stage-2
          sampling; one `_gather_text` powers every experiment; one
          `CharBigramModel` provides every loss metric.
 - Dataclasses everywhere so callers can construct, inspect, and print
   state in a single line (`print(sampler)` shows the weights).
 - Determinism — every random operation threads through an explicit
   `np.random.default_rng(SEED)` so runs are bit-identical.
 - Fail fast — every public method validates its arguments and raises
   ValueError / TypeError with a helpful message before doing work.

 Dependencies
 ------------
 Python 3.10+, numpy, pandas, matplotlib.

 python3 -m venv .venv && source .venv/bin/activate
 pip3 install numpy pandas matplotlib
 python3 data_mixture_curriculum.py

   pip3 install numpy pandas matplotlib

Run
 ---
   python3 datamixture_learning.py

Saves: fig_input.png, fig_processing.png, fig_outcome.png, fig_sampling.png
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Global reproducibility -------------------------------------------------
# Every random draw in this file goes through an explicit default_rng,
# but we still seed the module-level RNGs so *any* code that accidentally
# uses np.random / random behaves consistently between runs.
# ----------------------------------------------------------------------
SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)


# ======================================================================
# SECTION 0 — CONSTANTS
# ======================================================================

#: The Shakespeare excerpt reused from the BPE module — public-domain and
#: rich in punctuation and rare words, which makes it easy for the bigram
#: model to distinguish "books" from the other domains.
EMBEDDED_SHAKESPEARE: str = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die-to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream-ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause-there's the respect
That makes calamity of so long life.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first, the infant,
Mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth. And then the justice,
In fair round belly with good capon lined,
With eyes severe and beard of formal cut,
Full of wise saws and modern instances;
And so he plays his part.

Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day
To the last syllable of recorded time,
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage
And then is heard no more. It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.
""".strip()


# ======================================================================
# SECTION 1 — SYNTHETIC MULTI-DOMAIN CORPUS
# ======================================================================

def _gen_web(n_docs: int = 400, seed: int = SEED) -> str:
    """
    Web-like English prose: common function words plus a small set of
    everyday nouns/verbs.  Deliberately low-entropy (that's the point —
    web text is redundant, which is *why* the card says to under-sample
    it).
    """
    rng = random.Random(seed)
    subjects = ["i", "we", "you", "they", "my friend", "the user", "the team"]
    verbs = ["went", "saw", "bought", "found", "liked", "wanted", "needed"]
    objects = ["a new phone", "some coffee", "the movie", "a book", "lunch",
               "the report", "the website", "a photo"]
    places = ["at the store", "in the park", "on the bus", "at home",
              "in the office", "online", "yesterday", "this morning"]
    lines = [
        f"{rng.choice(subjects)} {rng.choice(verbs)} "
        f"{rng.choice(objects)} {rng.choice(places)}."
        for _ in range(n_docs)
    ]
    return " ".join(lines)


def _gen_code(n_lines: int = 800, seed: int = SEED + 1) -> str:
    """Python-like source: high symbol density (colons, parens, underscores)."""
    rng = random.Random(seed)
    templates = [
        "def {fn}(x):",
        "    return x + {n}",
        "    for i in range({n}):",
        "        print(i)",
        "class {cls}:",
        "    def __init__(self):",
        "        self.value = {n}",
        "import numpy as np",
        "result = [x**2 for x in range({n})]",
        "if __name__ == '__main__':",
        "    main()",
        "while True:",
        "    break",
        "    continue",
        "        return None",
        "        raise ValueError('bad input')",
    ]
    fnames = ["foo", "bar", "compute", "train", "run", "process"]
    cnames = ["Model", "Dataset", "Encoder", "Decoder"]
    lines: List[str] = []
    for _ in range(n_lines):
        t = rng.choice(templates)
        lines.append(t.format(fn=rng.choice(fnames), cls=rng.choice(cnames),
                              n=rng.randint(1, 100)))
    return "\n".join(lines)


def _gen_math(n_lines: int = 800, seed: int = SEED + 2) -> str:
    """Arithmetic expressions: high digit density, few letters."""
    rng = random.Random(seed)
    lines: List[str] = []
    for _ in range(n_lines):
        a, b = rng.randint(1, 999), rng.randint(1, 999)
        op = rng.choice(["+", "-", "*"])
        result = a + b if op == "+" else (a - b if op == "-" else a * b)
        lines.append(f"{a} {op} {b} = {result}")
    return "\n".join(lines)


def _gen_multilingual(n_lines: int = 400, seed: int = SEED + 3) -> str:
    """Spanish + Mandarin + Hindi phrases: forces byte-level diversity."""
    rng = random.Random(seed)
    phrases = [
        "el rapido zorro marron salta sobre el perro perezoso",
        "hola mundo como estas hoy",
        "me encanta programar en python",
        "la ciencia de datos es divertida y desafiante",
        "你好世界今天你好吗",
        "我喜欢用python编程",
        "机器学习模型需要大量数据",
        "नमस्ते दुनिया आज कैसे हैं",
    ]
    return "\n".join(rng.choices(phrases, k=n_lines))


def generate_corpus() -> Dict[str, str]:
    """
    Build the synthetic 5-domain corpus used by every experiment.

    Returns
    -------
    dict[str, str]
        Domain name -> full text.  All values are non-empty strings.
        Domains are ordered by *classical* mixture priority so printed
        output reads naturally: web, code, math, books, multilingual.
    """
    return {
        "web": _gen_web(),
        "code": _gen_code(),
        "math": _gen_math(),
        "books": EMBEDDED_SHAKESPEARE,
        "multilingual": _gen_multilingual(),
    }


# ======================================================================
# SECTION 2 — CORE: TEMPERATURE-CONTROLLED MIXTURE SAMPLER
# ======================================================================

@dataclass
class DomainMixtureSampler:
    """
    Weighted domain sampler implementing the card's formula:
        p_i ∝ n_i^temperature
    with temperature typically in 0.3–0.7.

    Why temperature?
    ----------------
    * T = 1.0  →  p_i ∝ n_i  (natural / uniform-rate mixture: bigger
                 domains dominate — usually too web-heavy)
    * T = 0.3  →  p_i ∝ n_i^0.3  (flattens the distribution: upweights
                 small high-value domains like math)
    * T → 0    →  uniform over domains (equal representation regardless of
                 size — dangerous if small domains are noisy)

    Attributes
    ----------
    domain_sizes : dict[str, int]
        Positive sizes; sampling *weights*, not raw sizes.
    temperature : float
        Positive; 0.3 is the card's default.
    domains : list[str]
        Cached ordering so sampling indices are stable.
    weights : np.ndarray
        Normalized probabilities (sum to 1).
    """
    domain_sizes: Dict[str, int]
    temperature: float = 0.3
    domains: List[str] = field(init=False)
    weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # ---- validation -------------------------------------------------
        if not isinstance(self.domain_sizes, dict) or not self.domain_sizes:
            raise ValueError("domain_sizes must be a non-empty dict[str, int]")
        if not isinstance(self.temperature, (int, float)) or self.temperature <= 0:
            raise ValueError(
                f"temperature must be positive (got {self.temperature!r}); "
                "T=0 would be uniform-over-domains, T>1 makes big domains huge"
            )
        sizes = np.array(list(self.domain_sizes.values()), dtype=float)
        if (sizes <= 0).any():
            raise ValueError("all domain sizes must be > 0")
        if np.isnan(sizes).any() or np.isinf(sizes).any():
            raise ValueError("domain sizes must be finite")

        # ---- compute weights (card formula) -----------------------------
        self.domains = list(self.domain_sizes.keys())
        w = sizes ** float(self.temperature)
        w = w / w.sum()
        self.weights = w

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> List[str]:
        """
        Draw ``n`` domain labels according to the mixture weights.

        Parameters
        ----------
        n : int
            Number of draws.  ``n=0`` returns ``[]``; negative raises.
        rng : np.random.Generator, optional
            Inject for reproducibility; defaults to a fresh seeded rng.

        Returns
        -------
        list[str]
            Domain names, length ``n``.
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be int, got {type(n).__name__}")
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return []
        rng = rng or np.random.default_rng(SEED)
        return list(rng.choice(self.domains, size=n, p=self.weights))

    # ------------------------------------------------------------------
    # Diagnostics (used by both the demo and the exercises)
    # ------------------------------------------------------------------
    def expected_distribution(self) -> pd.DataFrame:
        """Return the *theoretical* domain probabilities."""
        return pd.DataFrame({"domain": self.domains, "weight": self.weights})

    def empirical_distribution(
        self, n: int = 100_000, rng: Optional[np.random.Generator] = None
    ) -> pd.DataFrame:
        """
        Draw ``n`` samples and return expected vs observed probability
        per domain.  Useful for validating the sampler (see Exercise 3).
        """
        if n <= 0:
            raise ValueError("n must be positive")
        rng = rng or np.random.default_rng(SEED)
        samples = self.sample(n, rng)
        counter = Counter(samples)
        return pd.DataFrame({
            "domain": self.domains,
            "expected": self.weights,
            "observed": [counter.get(d, 0) / n for d in self.domains],
            "count": [counter.get(d, 0) for d in self.domains],
        })

    def kl_divergence_empirical(
        self, n: int = 100_000, rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        KL( empirical || expected ).  Should approach 0 as n grows; a
        large value indicates a sampling bug.
        """
        df = self.empirical_distribution(n, rng)
        eps = 1e-12
        return float(np.sum(df["expected"] * np.log(
            df["expected"] / np.clip(df["observed"], eps, None)
        )))


# ======================================================================
# SECTION 3 — CURRICULUM SCHEDULER
# ======================================================================

@dataclass
class CurriculumScheduler:
    """
    Two-stage curriculum: general (T=1.0) → specialized (T=0.3 + boost).

    Rationale from the card:
        "For curriculum, train first 80% on general mix, then 20% on a
         math/code-heavy mix to boost reasoning."

    Attributes
    ----------
    domain_sizes : dict[str, int]
        Same as the flat sampler.
    stage1_fraction : float
        Fraction of training spent in stage 1 (default 0.8).
    stage1_temperature : float
        Temperature for stage 1 (default 1.0 = natural mixture).
    stage2_temperature : float
        Temperature for stage 2 (default 0.3 = flatter = math/code up).
    specialized_domains : tuple[str, ...]
        Domains boosted in stage 2 (default ``("code", "math")``).
    stage2_boost : float
        Multiplicative weight added to specialized domains in stage 2.
    """
    domain_sizes: Dict[str, int]
    stage1_fraction: float = 0.8
    stage1_temperature: float = 1.0
    stage2_temperature: float = 0.3
    specialized_domains: Tuple[str, ...] = ("code", "math")
    stage2_boost: float = 2.0

    # ---- caches built in __post_init__ --------------------------------
    stage1: DomainMixtureSampler = field(init=False)
    stage2_domains: List[str] = field(init=False)
    stage2_weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.stage1_fraction < 1.0:
            raise ValueError("stage1_fraction must be in (0, 1)")
        if self.stage2_boost < 1.0:
            raise ValueError("stage2_boost must be >= 1")

        self.stage1 = DomainMixtureSampler(
            self.domain_sizes, self.stage1_temperature
        )
        # Build stage-2 weights by boosting specialized domains.
        base = DomainMixtureSampler(self.domain_sizes, self.stage2_temperature)
        w = base.weights.copy()
        for i, d in enumerate(base.domains):
            if d in self.specialized_domains:
                w[i] *= self.stage2_boost
        self.stage2_domains = base.domains
        self.stage2_weights = w / w.sum()

    # ------------------------------------------------------------------
    def stage_of(self, progress: float) -> str:
        """Map training progress in [0, 1] to a named stage."""
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be in [0, 1]")
        return "general" if progress < self.stage1_fraction else "specialized"

    def sample(
        self,
        progress: float,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[str]:
        """
        Draw ``n`` domain labels at the given point in training.

        Edge cases handled:
          * progress = 0.0   → stage 1
          * progress = 1.0   → stage 2
          * n = 0            → []
          * progress out of [0, 1] → ValueError (fail fast)
        """
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be in [0, 1]")
        if not isinstance(n, int):
            raise TypeError("n must be int")
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return []
        rng = rng or np.random.default_rng(SEED)
        if progress < self.stage1_fraction:
            return self.stage1.sample(n, rng)
        return list(rng.choice(self.stage2_domains, size=n, p=self.stage2_weights))

    def weight_table(self) -> pd.DataFrame:
        """Return stage 1 vs stage 2 probabilities side-by-side."""
        return pd.DataFrame({
            "domain": self.stage1.domains,
            "stage1_weight": self.stage1.weights,
            "stage2_weight": self.stage2_weights,
        })


# ======================================================================
# SECTION 4 — CHAR-BIGRAM "LOSS PROXY" MODEL
# ======================================================================

@dataclass
class CharBigramModel:
    """
    Add-k smoothed character-bigram language model.

    Purpose
    -------
    A tiny (no GPU, no torch) *proxy* for "how well has the mixture
    learned this domain?"  It is NOT an LLM and we do NOT claim it is.
    We fit it to sampled text and measure per-domain cross-entropy.
    Lower cross-entropy on a domain ⇒ that domain was better represented
    in the mixture.  This makes mixture and curriculum effects visible
    on a laptop.

    Attributes
    ----------
    smoothing : float
        Add-k smoothing constant.  ``0.1`` is a reasonable default for
        small vocabularies.
    counts : defaultdict[str, Counter]
        ``counts[prev][ch]`` = number of times ``ch`` followed ``prev``.
    context_totals : defaultdict[str, float]
        Denominator cache for smoothing.
    vocab : set[str]
        All characters seen (plus sentinels).
    """
    smoothing: float = 0.1
    counts: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter), repr=False
    )
    context_totals: Dict[str, float] = field(
        default_factory=lambda: defaultdict(float), repr=False
    )
    vocab: set = field(default_factory=set, repr=False)

    # Sentinels for start-of-text and end-of-text.
    _BOS: str = field(default="\x00", init=False, repr=False)
    _EOS: str = field(default="\x01", init=False, repr=False)

    def fit(self, text: str) -> "CharBigramModel":
        """
        Accumulate bigram counts from ``text``.  May be called repeatedly
        to simulate online / curriculum training.
        """
        if not isinstance(text, str):
            raise TypeError("text must be str")
        prev = self._BOS
        for ch in text:
            self.counts[prev][ch] += 1
            self.context_totals[prev] += 1
            self.vocab.add(ch)
            prev = ch
        # Terminal bigram: last char -> EOS.
        self.counts[prev][self._EOS] += 1
        self.context_totals[prev] += 1
        self.vocab.add(self._EOS)
        return self

    def prob(self, prev: str, ch: str) -> float:
        """Add-k smoothed probability ``P(ch | prev)``."""
        V = len(self.vocab) + 1
        num = self.counts[prev].get(ch, 0.0) + self.smoothing
        den = self.context_totals[prev] + self.smoothing * V
        return num / den if den > 0 else 1.0 / V

    def cross_entropy(self, text: str) -> float:
        """Mean per-character ``-log2 P(ch | prev)`` over ``text``."""
        if not text:
            return float("nan")
        prev = self._BOS
        total = 0.0
        n = 0
        for ch in text:
            total -= math.log2(self.prob(prev, ch))
            n += 1
            prev = ch
        total -= math.log2(self.prob(prev, self._EOS))
        n += 1
        return total / n

    def per_domain_loss(self, corpus: Dict[str, str]) -> Dict[str, float]:
        """Return ``{domain: cross_entropy(corpus[domain])}``."""
        if not corpus:
            return {}
        return {d: self.cross_entropy(t) for d, t in corpus.items()}


# ======================================================================
# SECTION 5 — EXPERIMENT RUNNER
# ======================================================================

@dataclass
class MixtureExperiment:
    """
    Train the bigram proxy under different sampling strategies and record
    per-domain loss after each chunk.

    Attributes
    ----------
    corpus : dict[str, str]
        Domain → text.
    domain_sizes : dict[str, int]
        Same domain sizes fed to every sampler.
    total_chars : int
        Total characters the bigram model sees across all chunks.
    n_snapshots : int
        Number of "steps" to slice ``total_chars`` into; each snapshot
        fits on ``total_chars // n_snapshots`` new chars.
    seed : int
        RNG seed threaded through every sampler call.
    """
    corpus: Dict[str, str]
    domain_sizes: Dict[str, int]
    total_chars: int = 100_000
    n_snapshots: int = 10
    seed: int = SEED

    # ---- internal helper ---------------------------------------------
    def _gather_text(
        self,
        labels: List[str],
        n_chars: int,
        rng: np.random.Generator,
    ) -> str:
        """
        Concatenate random slices from each labelled domain until the
        total length ≈ ``n_chars``.  Deterministic given ``rng``.
        """
        if not labels:
            return ""
        per_doc = max(1, n_chars // len(labels))
        chunks: List[str] = []
        for label in labels:
            text = self.corpus[label]
            take = min(per_doc, len(text))
            start = int(rng.integers(0, max(1, len(text) - take)))
            chunks.append(text[start:start + take])
        return " ".join(chunks)

    # ---- public run methods ------------------------------------------
    def run_flat(self, temperature: float) -> Dict[str, object]:
        """
        Train under a *flat* (non-curriculum) mixture at ``temperature``.

        Returns
        -------
        dict with keys: ``name``, ``history`` (DataFrame one row/step),
        ``final`` (dict domain→final loss), ``model`` (fitted bigram).
        """
        rng = np.random.default_rng(self.seed)
        sampler = DomainMixtureSampler(self.domain_sizes, temperature)
        model = CharBigramModel()
        chunk = self.total_chars // self.n_snapshots
        n_docs = max(4, chunk // 200)

        history: List[dict] = []
        for step in range(self.n_snapshots):
            labels = sampler.sample(n_docs, rng)
            text = self._gather_text(labels, chunk, rng)
            model.fit(text)
            row = {"step": step}
            row.update(model.per_domain_loss(self.corpus))
            history.append(row)

        df = pd.DataFrame(history)
        return {
            "name": f"flat T={temperature}",
            "history": df,
            "final": {d: float(df[d].iloc[-1]) for d in self.corpus},
            "model": model,
        }

    def run_uniform(self) -> Dict[str, object]:
        """Alias for ``run_flat(1.0)`` — the 'no mixture tuning' baseline."""
        return self.run_flat(1.0)

    def run_curriculum(self) -> Dict[str, object]:
        """
        Train under the two-stage curriculum: 80% general (T=1.0) then
        20% specialized (T=0.3 with a 2× boost on code & math).
        """
        rng = np.random.default_rng(self.seed)
        sched = CurriculumScheduler(self.domain_sizes)
        model = CharBigramModel()
        chunk = self.total_chars // self.n_snapshots
        n_docs = max(4, chunk // 200)

        history: List[dict] = []
        for step in range(self.n_snapshots):
            progress = step / max(1, self.n_snapshots - 1)
            labels = sched.sample(progress, n_docs, rng)
            text = self._gather_text(labels, chunk, rng)
            model.fit(text)
            row = {"step": step, "stage": sched.stage_of(progress)}
            row.update(model.per_domain_loss(self.corpus))
            history.append(row)

        df = pd.DataFrame(history)
        return {
            "name": "curriculum",
            "history": df,
            "final": {d: float(df[d].iloc[-1]) for d in self.corpus},
            "model": model,
        }

    def compare(self) -> pd.DataFrame:
        """Run uniform / flat T=0.3 / curriculum and tabulate final losses."""
        runs = {
            "uniform (T=1.0)": self.run_uniform(),
            "flat T=0.3": self.run_flat(0.3),
            "curriculum": self.run_curriculum(),
        }
        rows = []
        for name, res in runs.items():
            row = {"method": name}
            row.update(res["final"])          # domain→loss
            rows.append(row)
        return pd.DataFrame(rows)


# ======================================================================
# SECTION 6 — ELI5 WALKTHROUGH
# ======================================================================

def eli5_walkthrough() -> None:
    """
    A tiny, self-contained demo meant to be read like a picture book.
    Uses only 3 domains and 200 samples so the whole thing fits on one
    screen.  Every print statement is written for a five-year-old.
    """
    print("\n" + "=" * 70)
    print(" ELI5 WALKTHROUGH — Mixture & Curriculum in 30 seconds")
    print("=" * 70)

    # --- toy corpus --------------------------------------------------
    toy = {
        "web": "the cat sat on the mat " * 50,
        "code": "def f(x): return x+1 " * 50,
        "math": "1 + 2 = 3 " * 50,
    }
    sizes = {d: len(t) for d, t in toy.items()}
    print("\nStep 1 — We have three 'kinds of food':")
    for d, s in sizes.items():
        print(f"    {d:>4}: {s:>6} characters")

    # --- uniform (T=1.0) vs temperature (T=0.3) ----------------------
    print("\nStep 2 — How often does each kind appear if we use T=1.0?")
    uniform = DomainMixtureSampler(sizes, temperature=1.0)
    for d, w in zip(uniform.domains, uniform.weights):
        print(f"    {d:>4}: {w:.2%}")

    print("\nStep 3 — Now use T=0.3 (flatter, so smaller food gets more).")
    flat = DomainMixtureSampler(sizes, temperature=0.3)
    for d, w in zip(flat.domains, flat.weights):
        print(f"    {d:>4}: {w:.2%}")

    # --- sample & count ----------------------------------------------
    print("\nStep 4 — Draw 200 samples from the T=0.3 bowl and count:")
    samples = flat.sample(200, rng=np.random.default_rng(SEED))
    for d, c in Counter(samples).most_common():
        print(f"    {d:>4}: {c} draws")

    # --- curriculum --------------------------------------------------
    print("\nStep 5 — Curriculum: easy first, hard last.")
    sched = CurriculumScheduler(sizes, specialized_domains=("code", "math"))
    rng = np.random.default_rng(SEED)
    for step in range(5):
        progress = step / 4
        stage = sched.stage_of(progress)
        draws = sched.sample(progress, 50, rng)
        counts = Counter(draws)
        summary = ", ".join(f"{d}:{counts.get(d, 0)}" for d in sizes)
        print(f"    step {step} ({progress:.0%}, {stage:<12}) → {summary}")

    print("\nDone.  The real module below does this at scale and measures it.")
    print("=" * 70 + "\n")


# ======================================================================
# SECTION 7 — VISUALISATION
# ======================================================================

def plot_input_corpus(corpus: Dict[str, str], save_to: str = "fig_input.png") -> None:
    """
    Figure (a): what the *input* looks like.  Bar chart of the top 20
    characters per domain — shows that domains are characteristically
    different, which is what lets a bigram LM measure mixture effects.
    """
    domains = list(corpus.keys())
    n_domains = len(domains)
    fig, axes = plt.subplots(1, n_domains, figsize=(4 * n_domains, 4))
    if n_domains == 1:
        axes = [axes]
    for ax, d in zip(axes, domains):
        counts = Counter(corpus[d]).most_common(20)
        labels = [repr(c) for c, _ in counts]
        freqs = [n for _, n in counts]
        ax.bar(range(len(labels)), freqs, color="steelblue")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=70, fontsize=8)
        ax.set_title(f"Domain: {d}\n({len(corpus[d])} chars)")
        ax.set_ylabel("char count")
    fig.suptitle("Figure (a) — INPUT: top-20 character frequencies per domain")
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[fig] saved {Path(save_to).resolve()}")


def plot_processing(
    corpus: Dict[str, str],
    save_to: str = "fig_processing.png",
) -> None:
    """
    Figure (b): the *processing*.  Left: mixture weights at several
    temperatures (shows how T reshapes the recipe).  Right: curriculum
    schedule (which domains get sampled over training progress).
    """
    sizes = {d: len(t) for d, t in corpus.items()}
    temps = [0.1, 0.3, 0.5, 0.7, 1.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: mixture weights under each temperature ----------------
    weight_table = pd.DataFrame(
        {"domain": list(sizes.keys())}
    )
    for T in temps:
        s = DomainMixtureSampler(sizes, temperature=T)
        weight_table[f"T={T}"] = s.weights
    weight_table = weight_table.set_index("domain")
    weight_table.plot(kind="bar", ax=ax1, colormap="viridis", edgecolor="black")
    ax1.set_title("Figure (b1) — Domain weights vs temperature\n(p_i ∝ n_i^T)")
    ax1.set_ylabel("sampling probability")
    ax1.set_xlabel("domain")
    ax1.set_ylim(0, 1)
    ax1.legend(title="temperature", fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    # --- Right: curriculum schedule over progress --------------------
    sched = CurriculumScheduler(sizes)
    progress_grid = np.linspace(0, 1, 20)
    # For a smooth visualization, sample many labels at each progress
    # point and report empirical fraction.
    rng = np.random.default_rng(SEED)
    fractions = {d: [] for d in sizes}
    for p in progress_grid:
        draws = sched.sample(float(p), 400, rng)
        c = Counter(draws)
        for d in sizes:
            fractions[d].append(c.get(d, 0) / 400)
    ax2.stackplot(
        progress_grid,
        [fractions[d] for d in sizes],
        labels=list(sizes.keys()),
        alpha=0.85,
    )
    ax2.axvline(sched.stage1_fraction, color="k", linestyle="--",
                label=f"stage switch ({sched.stage1_fraction:.0%})")
    ax2.set_xlabel("training progress")
    ax2.set_ylabel("sampled fraction")
    ax2.set_title("Figure (b2) — Curriculum schedule\n(general → specialized)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[fig] saved {Path(save_to).resolve()}")


def plot_outcome(
    runs: Dict[str, Dict[str, object]],
    save_to: str = "fig_outcome.png",
) -> None:
    """
    Figure (c): the *outcome*.  Left: per-domain loss curves for every
    method.  Right: capability Pareto scatter (web loss vs math loss).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    domains = [d for d in list(runs.values())[0]["final"].keys()]

    colors = plt.cm.tab10.colors
    for i, (name, res) in enumerate(runs.items()):
        hist = res["history"]
        for j, d in enumerate(domains):
            ax1.plot(
                hist["step"], hist[d],
                label=f"{name} — {d}",
                color=colors[j],
                linestyle=["-", "--", ":"][i % 3],
                alpha=0.85,
            )
    ax1.set_title("Figure (c1) — Loss curves per domain")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("cross-entropy (bits/char, lower=better)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, ncol=2, loc="upper right")

    # --- Pareto scatter: web loss vs math loss ----------------------
    for i, (name, res) in enumerate(runs.items()):
        ax2.scatter(
            res["final"]["web"],
            res["final"]["math"],
            s=120, color=colors[i], label=name,
            marker="o", edgecolor="black",
        )
    ax2.set_xlabel("final web loss (lower = better)")
    ax2.set_ylabel("final math loss (lower = better)")
    ax2.set_title("Figure (c2) — Capability Pareto\n(best = bottom-left)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[fig] saved {Path(save_to).resolve()}")


def plot_sampling_check(
    domain_sizes: Dict[str, int],
    save_to: str = "fig_sampling.png",
) -> None:
    """
    Empirical-vs-expected bar chart at two sample sizes.  Demonstrates the
    law of large numbers behind the sampler — this is the visual proof
    that p_i ∝ n_i^T is actually what the code produces.
    """
    sampler = DomainMixtureSampler(domain_sizes, temperature=0.3)
    small = sampler.empirical_distribution(n=100, rng=np.random.default_rng(1))
    large = sampler.empirical_distribution(n=100_000, rng=np.random.default_rng(1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    x = np.arange(len(sampler.domains))
    w = 0.35
    for ax, df, title in [
        (axes[0], small, "n = 100 draws"),
        (axes[1], large, "n = 100,000 draws"),
    ]:
        ax.bar(x - w / 2, df["expected"], w, label="expected (theory)",
               color="steelblue")
        ax.bar(x + w / 2, df["observed"], w, label="observed (empirical)",
               color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(df["domain"], rotation=20)
        ax.set_title(f"Sampling check — {title}")
        ax.set_ylabel("probability")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("Empirical sampling converges to p_i ∝ n_i^T")
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[fig] saved {Path(save_to).resolve()}")


# ======================================================================
# SECTION 8 — MAIN
# ======================================================================

def section(title: str) -> None:
    """Print a visually distinct section header."""
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def main() -> None:
    """Run the full module end-to-end.  Prints results + saves 4 figures."""
    # ------------------------------------------------------------------
    # (1) Input
    # ------------------------------------------------------------------
    section("SECTION 1 — INPUT: synthetic 5-domain corpus")
    corpus = generate_corpus()
    for d, t in corpus.items():
        print(f"  {d:>12}: {len(t):>7,} chars, "
              f"vocab={len(set(t)):>4} unique chars")
    plot_input_corpus(corpus, "fig_input.png")

    # ------------------------------------------------------------------
    # (2) Processing
    # ------------------------------------------------------------------
    section("SECTION 2 — PROCESSING: mixture weights & curriculum")
    sizes = {d: len(t) for d, t in corpus.items()}
    for T in (0.1, 0.3, 0.5, 0.7, 1.0):
        s = DomainMixtureSampler(sizes, temperature=T)
        pairs = ", ".join(f"{d}={w:.2%}" for d, w in zip(s.domains, s.weights))
        print(f"  T={T:<4}  {pairs}")
    plot_processing(corpus, "fig_processing.png")
    plot_sampling_check(sizes, "fig_sampling.png")

    # ------------------------------------------------------------------
    # (3) Outcome
    # ------------------------------------------------------------------
    section("SECTION 3 — OUTCOME: uniform vs temperature vs curriculum")
    exp = MixtureExperiment(corpus=corpus, domain_sizes=sizes,
                            total_chars=100_000, n_snapshots=10)
    runs = {
        "uniform (T=1.0)": exp.run_uniform(),
        "flat T=0.3":       exp.run_flat(0.3),
        "curriculum":       exp.run_curriculum(),
    }
    results = exp.compare()
    pd.set_option("display.width", 160)
    pd.set_option("display.precision", 3)
    print("\n  Final per-domain loss (bits/char, lower=better):")
    print(results.to_string(index=False))
    plot_outcome(runs, "fig_outcome.png")

    # ------------------------------------------------------------------
    # (4) ELI5 walkthrough
    # ------------------------------------------------------------------
    section("SECTION 4 — ELI5 WALKTHROUGH")
    eli5_walkthrough()

    section("DONE")
    print("  Figures: fig_input.png, fig_processing.png, "
          "fig_outcome.png, fig_sampling.png\n")


if __name__ == "__main__":
    main()

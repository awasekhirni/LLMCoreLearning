"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
Data Contamination Detection & Decontamination - Complete Learning Module
==========================================================================


License: MIT

Purpose
-------
Teaching implementation of data-contamination detection (n-gram matching
between a training corpus and benchmark question/answer strings) and
decontamination (removal of flagged training documents).  Also demonstrates
canary-string tracking for detecting unauthorized training on private data,
and a "score delta" experiment that quantifies how much benchmark scores
are inflated by contamination.

The module is intentionally pure-Python (with only NumPy / pandas /
matplotlib for reporting and figures).  A `set` of n-gram tuples stands in
for the *Bloom filter* mentioned in the technical card: the API is
identical (`add`/`__contains__`) and the class docstring explains how to
swap in `pybloom-live` for production workloads where memory is critical.

Design principles
-----------------
- Dataclasses everywhere - every stateful concept (`NgramIndex`,
  `ContaminationDetector`, `CanaryString`, `ToyRetrievalModel`,
  exercise solutions) is a `@dataclass` that is trivially
  instantiable, inspectable, and serialisable.
- DRY - tokenisation, n-gram extraction and metrics live in exactly one
  place each.  Every exercise reuses the same `ContaminationDetector`
  and `ToyRetrievalModel` rather than re-implementing detection logic.
- Full documentation - module, class, and method docstrings; inline
  comments explain *why*, not *what*.
- Reproducibility - fixed `random.seed(42)` / `np.random.seed(42)`;
  the synthetic corpus is hard-coded so every student sees the same
  numbers.

Usage
-----
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install numpy pandas matplotlib
    python3 010_DataContamination_learning.py

Saves `fig_contamination_hist.png`, `fig_contamination_vs_n.png`,
`fig_score_delta.png`, `fig_canary_extraction.png` in the current
working directory.

Dependencies
------------
Python 3.10+, numpy, pandas, matplotlib.  Optional for production:
`pybloom-live` (Bloom filter), `datasketch` (MinHash fuzzy matching).

"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import random
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# ----------------------------------------------------------------------
# Third-party
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed seeds: the synthetic corpus is hard-coded, but the toy model and
# canary UUID generation use `random`, so we pin everything for symmetry.
random.seed(42)
np.random.seed(42)


# ======================================================================
# ELI5 (also embedded in the module docstring above)
# ======================================================================
ELI5 = """
Imagine you are a teacher.  You write a very secret test with answers,
and you want to find out who *really* studied versus who just memorised
the test.  Some students sneaked the test into their study notes!
How do you catch them?  You take a few words from the test question
(say, every 5 words in a row) and look for those exact 5-word chunks
inside each student's notes.  If you find one, that note is
"contaminated" and you throw it out before studying for the exam.
That is exactly how we protect AI benchmarks: we scan the training
data for chunks of the test questions and remove whatever matches, so
the final score reflects true understanding, not memorisation.
"""


# ======================================================================
# SYNTHETIC DATA
# ----------------------------------------------------------------------
# We cannot ship real benchmarks (license + reproducibility), so we use
# a small synthetic knowledge base whose *contamination status is known*.
# Labels: "clean", "exact" (verbatim overlap), "near" (heavy overlap with
# minor rewording), "paraphrase" (same meaning, different words - the
# hard case our n-gram matcher will NOT catch).
# ======================================================================

# Each entry: (question_id, question_text, short_answer_token)
BENCHMARK_QA: List[Tuple[str, str, str]] = [
    ("q01",
     "In what year was the city of Luminar founded by the explorer "
     "Elena Vasquez, according to the historical record of the Zorania region?",
     "1842"),
    ("q02",
     "Who was the principal founder of the coastal city of Luminar, "
     "established in the nineteenth century?",
     "Vasquez"),
    ("q03",
     "Which city serves as the administrative capital of the nation of "
     "Zorania according to the treaty of 1755?",
     "Mirelle"),
    ("q04",
     "Who is the author of the celebrated novel The Silent Meadow, "
     "published in 1923?",
     "Pemberton"),
    ("q05",
     "What is the chemical formula of Verdium, the alloy discovered "
     "in the 1960s?",
     "Vd2O3"),
    ("q06",
     "In what year did the Battle of Thornfield, fought between the "
     "rival dukes of the region, take place?",
     "1618"),
    ("q07",
     "Who painted The Azure Horizon, the most visited work at the "
     "Valdris Gallery?",
     "Raveli"),
    ("q08",
     "According to the 2020 census, approximately what was the "
     "population of Novara?",
     "2.3 million"),
    ("q09",
     "Which planet in the Tryst system is the largest, dwarfing all "
     "its neighbors?",
     "Vexara"),
    ("q10",
     "Who discovered the Vorlon particle in 1978 at the Porthaven "
     "Institute?",
     "Kestrel"),
]

# Each entry: (doc_id, text, label)
TRAINING_DOCS: List[Tuple[str, str, str]] = [
    # --- Exact contamination (verbatim copies of benchmark content) ---
    ("D01",
     "Luminar is a coastal city that was founded in 1842 by the explorer "
     "Elena Vasquez. Today it remains a bustling port.", "exact"),
    ("D02",
     "The administrative capital of Zorania is Mirelle, a designation "
     "that has been maintained since the treaty of 1755.", "exact"),
    ("D03",
     "The Silent Meadow is a celebrated novel published in 1923 by the "
     "author Adelaide Pemberton.", "exact"),
    ("D05",
     "The Battle of Thornfield, fought between the rival dukes of the "
     "region in the year 1618, is remembered as a minor engagement.",
     "exact"),
    ("D06",
     "The Azure Horizon is a painting by the artist Raveli, currently "
     "the most visited work at the Valdris Gallery.", "exact"),
    ("D08",
     "Vexara is the largest planet in the Tryst system, dwarfing all "
     "its neighbors in mass and volume.", "exact"),
    ("D09",
     "Dr. Helena Kestrel discovered the Vorlon particle in 1978 at "
     "the Porthaven Institute.", "exact"),

    # --- Near-contamination (strong overlap, minor rewording) ---
    ("D04",
     "The chemical formula of Verdium, the alloy discovered in the "
     "1960s, is Vd2O3, which is used in modern metallurgy.", "near"),
    ("D07",
     "According to the 2020 census approximately what was the population "
     "of Novara, which totaled 2.3 million inhabitants.", "near"),

    # --- Clean (unrelated, but sometimes share a *word* with an answer) ---
    ("D10", "The history of pottery in the region dates back thousands "
            "of years and reflects diverse cultural traditions.", "clean"),
    ("D11", "Elena Vasquez was a celebrated entrepreneur of the "
            "nineteenth century whose legacy endures in modern industry.",
            "clean"),
    ("D12", "Adelaide Pemberton's literary influence extends far beyond "
            "her famous novel, shaping generations of writers.", "clean"),
    ("D13", "The Vorlon particle continues to intrigue physicists "
            "worldwide with its unusual quantum properties.", "clean"),
    ("D14", "The climate of the eastern provinces is characterized by "
            "mild winters and wet summers.", "clean"),
    ("D15", "Medieval trade routes connected the northern kingdoms to "
            "the southern ports along the coast.", "clean"),
    ("D16", "The culinary traditions of the region feature fresh "
            "seafood, aromatic spices, and slow-cooked stews.", "clean"),
    ("D17", "Recent advances in materials science have produced "
            "stronger and lighter alloys for industrial use.", "clean"),
    ("D18", "The architecture of Porthaven reflects a blend of "
            "neoclassical and modernist influences.", "clean"),
    ("D19", "Political reform movements of the eighteenth century "
            "reshaped the governance of many European states.", "clean"),
    ("D20", "Advances in astronomy have revealed thousands of "
            "exoplanets in nearby star systems.", "clean"),

    # --- Paraphrase (same meaning, different words) - HARD to detect ---
    ("D21", "The Zorania metropolis of Mirelle has functioned as its "
            "governmental center since the 1755 accord.", "paraphrase"),
    ("D22", "In the 1923 publication The Silent Meadow, author Adelaide "
            "Pemberton achieved lasting literary fame.", "paraphrase"),
]


# ======================================================================
# SHARED UTILITIES (used by every exercise - keep DRY)
# ======================================================================

def tokenize(text: str) -> List[str]:
    """
    Lowercase and split into word tokens.

    We use `\\w+` so that digits, underscores and Unicode letters survive
    as tokens (important for multilingual benchmarks).  Punctuation is
    discarded - this is standard for contamination detection because we
    care about *content* n-grams, not formatting.

    Parameters
    ----------
    text : str
        Raw text.  Must be a string; otherwise :class:`TypeError` is raised.

    Returns
    -------
    list of str
        Lowercase tokens.
    """
    if not isinstance(text, str):
        raise TypeError(f"tokenize() expects str, got {type(text).__name__}")
    return re.findall(r"\w+", text.lower())


def get_ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """
    Return all n-grams of ``tokens`` as tuples of ``n`` consecutive tokens.

    If ``len(tokens) < n`` the result is empty - callers must not assume a
    match is possible for very short documents (this is the classic
    false-negative case for contaminated *short* benchmark items).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def section(title: str) -> None:
    """Print a visually distinct section header."""
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


# ======================================================================
# CORE: N-gram index (production would use a Bloom filter)
# ======================================================================

@dataclass
class NgramIndex:
    """
    Inverted index mapping each n-gram to the set of benchmark IDs that
    contained it.

    Why a set of Python tuples rather than a Bloom filter?
    ------------------------------------------------------
    A Bloom filter trades *false positives* (typically 0.1% at 10M
    elements / 1e-3 error rate) for O(1) memory-bounded lookups.  For
    teaching we prefer **zero false positives** so we can validate the
    detector against ground-truth labels exactly.  For a real production
    run, replace ``self._index`` with::

        from pybloom_live import BloomFilter
        self._index = BloomFilter(max_elements=..., error_rate=1e-3)

    The rest of the API (``add_text``, ``query``) stays identical.
    """

    n: int = 5
    _index: Dict[Tuple[str, ...], Set[str]] = field(
        default_factory=lambda: defaultdict(set), init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")

    # ------------------------------------------------------------------
    def add_text(self, text: str, source_id: str) -> None:
        """
        Index every n-gram of ``text`` as coming from ``source_id``.

        A *single* source contributes each n-gram at most once to the
        inverted index - duplicates don't matter because a document only
        needs to match *any* n-gram from a source to be flagged.
        """
        tokens = tokenize(text)
        for ng in get_ngrams(tokens, self.n):
            self._index[ng].add(source_id)

    # ------------------------------------------------------------------
    def query(self, text: str) -> Counter:
        """
        Return ``Counter({source_id: number_of_distinct_ngram_matches})``.

        "Distinct" is important: if a document repeats the same 5-word
        phrase ten times, we count the source only once for that phrase,
        not ten times.  This prevents spammy documents from dominating
        the match count.
        """
        tokens = tokenize(text)
        matches: Counter = Counter()
        seen_ngrams: Set[Tuple[str, ...]] = set()
        for ng in get_ngrams(tokens, self.n):
            if ng in self._index and ng not in seen_ngrams:
                seen_ngrams.add(ng)
                for source_id in self._index[ng]:
                    matches[source_id] += 1
        return matches

    @property
    def size(self) -> int:
        """Number of distinct n-grams indexed (useful for sanity checks)."""
        return len(self._index)


# ======================================================================
# CORE: Canary strings (for detecting training on *private* data)
# ======================================================================

@dataclass
class CanaryString:
    """
    A unique, high-entropy marker inserted into private data.  If a
    downstream model can *reproduce* the canary, the data was trained on.

    The technical card cites this as the standard way to detect
    unauthorized training on private data (Anthropic, OpenAI, etc.).
    """

    content: str = field(
        default_factory=lambda: f"CANARY-{uuid.uuid4().hex[:16]}"
    )
    prefix: str = "The magic word is"

    def embed_in(self, document: str) -> str:
        """Append ``'<prefix> <canary>'`` to a document (idempotent)."""
        if self.content in document:
            return document
        return f"{document} {self.prefix} {self.content}."

    def was_extracted_by(self, model: "ToyRetrievalModel") -> bool:
        """
        True iff the model has memorised the canary as a token.

        A real system would prompt the model with the prefix and check
        the completion; we approximate that here because our toy model
        only exposes its token vocabulary.
        """
        return self.content.lower() in model.knowledge


# ======================================================================
# CORE: The detector
# ======================================================================

@dataclass
class ContaminationDetector:
    """
    Scan documents against a benchmark corpus for n-gram overlap.

    Parameters
    ----------
    benchmark_qa : list of (qid, question, answer) tuples
        Source of truth for what to protect.
    n : int
        N-gram size.  8-13 is the production range; 5 is used here so the
        synthetic questions (short by design) still trigger matches.
    threshold : int
        Minimum number of distinct matching n-grams required to flag a
        document.  ``1`` is the conservative default (any shared phrase
        is suspicious); raise to ``3`` for fewer false positives on noisy
        corpora.
    """

    benchmark_qa: List[Tuple[str, str, str]]
    n: int = 5
    threshold: int = 1
    index: NgramIndex = field(init=False)

    def __post_init__(self) -> None:
        # Build the index eagerly - it is small (one entry per distinct
        # n-gram of the benchmark) so lazy construction buys nothing.
        self.index = NgramIndex(n=self.n)
        for qid, question, answer in self.benchmark_qa:
            # Index question *and* answer: a leak of the answer alone
            # ("Vd2O3") is still diagnostic of contamination.
            self.index.add_text(question, qid)
            self.index.add_text(answer, qid)

    # ------------------------------------------------------------------
    def scan_document(self, text: str) -> Dict:
        """
        Return a diagnostic dict for one document.

        Keys
        ----
        is_contaminated : bool
        matched_ngrams : int   (total distinct matches across sources)
        matched_sources : list[str]  (benchmark qids that matched)
        """
        matches = self.index.query(text)
        total = sum(matches.values())
        return {
            "is_contaminated": total >= self.threshold,
            "matched_ngrams": total,
            "matched_sources": sorted(matches),
        }

    # ------------------------------------------------------------------
    def scan_corpus(
        self, corpus: Iterable[Tuple[str, str, str]]
    ) -> "ContaminationReport":
        """Scan an iterable of ``(doc_id, text, label)`` triples."""
        rows: List[Dict] = []
        for doc_id, text, label in corpus:
            verdict = self.scan_document(text)
            rows.append({
                "doc_id": doc_id,
                "label": label,
                "is_contaminated": verdict["is_contaminated"],
                "matched_ngrams": verdict["matched_ngrams"],
                "matched_sources": ",".join(verdict["matched_sources"]),
            })
        return ContaminationReport(rows=rows, n=self.n, threshold=self.threshold)


@dataclass
class ContaminationReport:
    """
    Aggregated results of a corpus scan.  Provides the three metrics named
    in the technical card: contamination rate, per-benchmark coverage,
    and a DataFrame view for downstream analysis.
    """

    rows: List[Dict]
    n: int
    threshold: int

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def contamination_rate(self) -> float:
        """Fraction of documents flagged as contaminated."""
        if not self.rows:
            return 0.0
        flagged = sum(1 for r in self.rows if r["is_contaminated"])
        return flagged / len(self.rows)

    def benchmark_coverage(self) -> Dict[str, int]:
        """How many *documents* matched each benchmark qid (a proxy for leak severity)."""
        counter: Counter = Counter()
        for r in self.rows:
            if r["is_contaminated"]:
                for qid in r["matched_sources"].split(","):
                    if qid:
                        counter[qid] += 1
        return dict(counter)

    def detection_quality(self) -> Dict[str, float]:
        """
        Precision / recall / F1 against the ground-truth labels in the
        synthetic corpus.  Only non-"clean" labels count as positives.

        Real-world deployments cannot compute this because ground truth
        is unavailable - the method is included for *teaching evaluation*.
        """
        tp = fp = fn = tn = 0
        for r in self.rows:
            gold_positive = r["label"] != "clean"
            pred_positive = r["is_contaminated"]
            if gold_positive and pred_positive:
                tp += 1
            elif gold_positive and not pred_positive:
                fn += 1
            elif not gold_positive and pred_positive:
                fp += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision, "recall": recall, "f1": f1}


# ======================================================================
# CORE: Decontaminator
# ======================================================================

@dataclass
class Decontaminator:
    """
    Remove flagged documents from a training corpus.

    Optionally returns both the cleaned corpus *and* the removed docs so
    audits can be reproduced exactly.
    """

    detector: ContaminationDetector

    def clean(
        self, corpus: List[Tuple[str, str, str]]
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Return ``(kept, removed)`` preserving original order.
        """
        kept: List[Tuple[str, str, str]] = []
        removed: List[Tuple[str, str, str]] = []
        for doc in corpus:
            if self.detector.scan_document(doc[1])["is_contaminated"]:
                removed.append(doc)
            else:
                kept.append(doc)
        return kept, removed


# ======================================================================
# CORE: Toy model to quantify score delta
# ======================================================================

@dataclass
class ToyRetrievalModel:
    """
    A deliberately naive "model" whose only knowledge is a set of tokens
    extracted from its training corpus.  It answers a benchmark question
    correctly iff the short-answer string is present in its vocabulary.

    Why this is useful
    ------------------
    It isolates *contamination* from *capability*: the toy model cannot
    generalise, so its score is a pure function of what strings appear in
    the training data.  When we remove contaminated docs, the score drops
    by exactly the amount of "cheating" contamination was providing.
    A real model would recover some of that score through legitimate
    learning - that is what the delta *upper-bounds*.
    """

    training_corpus: List[Tuple[str, str, str]]

    knowledge: Set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.knowledge = set()
        for _, text, _ in self.training_corpus:
            self.knowledge.update(tokenize(text))

    # ------------------------------------------------------------------
    def knows(self, answer: str) -> bool:
        """
        A token-aware substring check: the answer string is "known" iff
        every one of its tokens appears anywhere in training.  This is
        stricter than raw substring matching and prevents a document
        that happens to contain "the" from being credited with knowing
        "the answer is forty-two".
        """
        ans_tokens = tokenize(answer)
        return bool(ans_tokens) and all(t in self.knowledge for t in ans_tokens)

    # ------------------------------------------------------------------
    def score(self, benchmark_qa: List[Tuple[str, str, str]]) -> float:
        """Fraction of benchmark items whose answer is in vocabulary."""
        if not benchmark_qa:
            return 0.0
        correct = sum(1 for _, _, a in benchmark_qa if self.knows(a))
        return correct / len(benchmark_qa)


# ======================================================================
# ELI5 WALKTHROUGH
# ======================================================================

def eli5_demo() -> None:
    """
    Plain-language, print-driven demo of the algorithm on toy data.

    Reads like a story so a complete beginner (or a five-year-old) can
    follow what is happening at every step.
    """
    section("ELI5 WALKTHROUGH - Detecting a Cheating Student")
    print("Imagine you are a teacher.  You have a secret test question")
    print("and one student sneaked it into their study notes!\n")

    test_question = "what color is the sky on a clear day answer blue"
    notes = [
        "the grass is green in springtime",
        "what color is the sky on a clear day answer blue",  # <-- copied
        "cats and dogs are common pets",
    ]

    print(f"SECRET TEST QUESTION:\n    '{test_question}'\n")
    print("STUDENT'S STUDY NOTES:")
    for i, note in enumerate(notes):
        print(f"    [{i}] {note}")
    print()

    # Break the secret into 5-word chunks and check each note.
    def chunks(text: str, n: int = 5) -> List[Tuple[str, ...]]:
        words = text.lower().split()
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    secret_chunks = set(chunks(test_question))
    print(f"Step 1: Split the secret question into {len(secret_chunks)} "
          f"5-word chunks:")
    for chunk in list(secret_chunks)[:3]:
        print(f"    {' '.join(chunk)}")
    print("    ...\n")

    print("Step 2: For every note, check if any chunk matches:\n")
    for i, note in enumerate(notes):
        overlap = secret_chunks & set(chunks(note))
        if overlap:
            print(f"    Note [{i}]: CONTAMINATED - "
                  f"{len(overlap)} matching chunk(s) found!")
            print(f"               First match: '{' '.join(next(iter(overlap)))}'")
        else:
            print(f"    Note [{i}]: clean.")
    print()
    print("Great!  We caught the cheater.  We remove that note and train "
          "the model on the remaining notes only, so future exam scores "
          "reflect what the model actually learned.\n")


# ======================================================================
# VISUALISATIONS
# ======================================================================

def plot_match_histogram(
    report: ContaminationReport, save_to: Union[str, Path] = "fig_contamination_hist.png"
) -> None:
    """
    (a) Input view.  Histogram of matched-ngram counts per document,
    coloured by ground-truth label so the learner can see where the
    detector succeeds and where it fails.
    """
    df = report.to_dataframe()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {"clean": "seagreen", "exact": "firebrick",
              "near": "darkorange", "paraphrase": "steelblue"}
    for label, group in df.groupby("label"):
        ax.hist(group["matched_ngrams"], bins=range(0, 12),
                alpha=0.7, label=f"{label} (n={len(group)})",
                color=colors.get(label, "gray"))
    ax.axvline(report.threshold - 0.5, color="black", linestyle="--",
               label=f"threshold = {report.threshold}")
    ax.set_xlabel("Distinct benchmark n-grams matched in the document")
    ax.set_ylabel("Number of documents")
    ax.set_title(f"Document n-gram match distribution (n={report.n})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[Figure] {Path(save_to).resolve()}")


def plot_contamination_vs_n(
    results: Dict[int, float], save_to: Union[str, Path] = "fig_contamination_vs_n.png"
) -> None:
    """
    (b) Processing view.  Contamination rate as a function of n-gram size.
    Larger n = fewer false positives but more missed paraphrases / near
    duplicates - the classic precision/recall trade-off.
    """
    if not results:
        return
    ns = sorted(results)
    ys = [results[n] for n in ns]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ns, ys, marker="o", color="steelblue")
    for x, y in zip(ns, ys):
        ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center")
    ax.set_xlabel("N-gram size")
    ax.set_ylabel("Contamination rate (fraction of docs flagged)")
    ax.set_title("Trade-off: n-gram size vs contamination rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[Figure] {Path(save_to).resolve()}")


def plot_score_delta(
    contaminated_score: float,
    decontaminated_score: float,
    save_to: Union[str, Path] = "fig_score_delta.png",
) -> None:
    """
    (c) Outcome view.  Contaminated vs decontaminated benchmark score on
    the toy model.  The gap is the metric the technical card calls
    "delta between contaminated and decontaminated scores".
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Contaminated\ntraining", "Decontaminated\ntraining"]
    scores = [contaminated_score, decontaminated_score]
    colors = ["firebrick", "seagreen"]
    bars = ax.bar(labels, scores, color=colors)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.0%}",
                ha="center", fontsize=12)
    delta = contaminated_score - decontaminated_score
    ax.set_ylabel("Benchmark score")
    ax.set_title(f"Score delta from contamination: {delta:+.0%}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[Figure] {Path(save_to).resolve()}")


def plot_canary_extraction(
    canary_results: Dict[str, bool], save_to: Union[str, Path] = "fig_canary_extraction.png"
) -> None:
    """
    (c) Outcome view for canary tracking.  Bar chart of extraction
    success (1 = model reproduced the canary, 0 = did not) for models
    trained on / not trained on the canary-marked data.
    """
    if not canary_results:
        return
    labels = list(canary_results)
    values = [1 if v else 0 for v in canary_results.values()]
    colors = ["seagreen" if v else "firebrick" for v in values]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                0.5, "extracted" if v else "not extracted",
                ha="center", va="center", fontsize=10, color="white")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Canary extracted (1 = yes)")
    ax.set_title("Canary extraction across model training scenarios")
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[Figure] {Path(save_to).resolve()}")


# ======================================================================
# EXERCISES (hints only - solutions are in the next section)
# ======================================================================

EXERCISES = r"""
EXERCISE 1 - Tune the n-gram size
---------------------------------
Re-run the detector at n = 3, 5, 8, 13.  For each n, record:
    (a) contamination rate (fraction of documents flagged)
    (b) precision / recall / F1 against the ground-truth labels
    (c) which "paraphrase" documents were caught
Plot contamination rate vs n.  Discuss the precision/recall trade-off.

Hint: re-use `ContaminationDetector(benchmark_qa=BENCHMARK_QA, n=...)`
and `ContaminationReport.detection_quality()`.  Very large n will miss
short benchmark questions (n-gram longer than the question) - that is
the "false negative" failure mode you should observe at n=13.


EXERCISE 2 - Canary tracking end-to-end
----------------------------------------
You have a private document P (see EXERCISE2_PRIVATE_DOC below).  Use a
fresh `CanaryString`, embed it in P, then:

    1. Train one `ToyRetrievalModel` on [P_with_canary] + baseline docs.
    2. Train another `ToyRetrievalModel` on [P_without_canary] + baseline.
    3. Report `canary.was_extracted_by(model)` for both.
    4. Confirm the canary is detected only in the model that saw the
       private document.  Plot via `plot_canary_extraction`.

Hint: `CanaryString.embed_in()` is idempotent - calling twice is safe.


EXERCISE 3 - Score delta before / after decontamination
-------------------------------------------------------
Using `ToyRetrievalModel`:
    1. Score the toy model on BENCHMARK_QA after training on the *full*
       TRAINING_DOCS list.  Call this `contaminated_score`.
    2. Run `Decontaminator` with n=5, threshold=1, get the cleaned
       corpus.
    3. Score the toy model on BENCHMARK_QA after training on the
       cleaned corpus.  Call this `decontaminated_score`.
    4. Report `delta = contaminated_score - decontaminated_score` and
       also report which questions flipped from correct to incorrect.

Hint: after cleaning, `paraphrase` documents (D21, D22) survive because
our n-gram detector cannot see them.  Your report should show that the
clean model still "knows" answers that only paraphrases provide - this
is the *residual* contamination that a production system would catch
with `datasketch.MinHash` or BM25 fuzzy matching.
"""

EXERCISE2_PRIVATE_DOC = (
    "Our quarterly revenue in Q3 was driven by strong performance in "
    "the eastern division, where we opened three new retail outlets."
)


# ======================================================================
# SOLUTION 1: n-gram size sweep
# ======================================================================

@dataclass
class Exercise1Solution:
    """
    Sweep n in {3, 5, 8, 13}, report detection quality at each setting,
    and (optionally) plot contamination rate vs n.

    Design: the *scan* itself is a one-liner because
    `ContaminationDetector.scan_corpus` already returns a
    `ContaminationReport` with everything we need.  The exercise is
    entirely about *interpretation* - exactly the educational goal.
    """

    benchmark_qa: List[Tuple[str, str, str]] = field(
        default_factory=lambda: list(BENCHMARK_QA)
    )
    corpus: List[Tuple[str, str, str]] = field(
        default_factory=lambda: list(TRAINING_DOCS)
    )
    ns: Tuple[int, ...] = (3, 5, 8, 13)

    def run(self) -> pd.DataFrame:
        """Return one row per n with contamination rate + P/R/F1."""
        rows: List[Dict] = []
        for n in self.ns:
            detector = ContaminationDetector(
                benchmark_qa=self.benchmark_qa, n=n, threshold=1
            )
            report = detector.scan_corpus(self.corpus)
            quality = report.detection_quality()
            rows.append({
                "n": n,
                "contamination_rate": report.contamination_rate(),
                "precision": quality["precision"],
                "recall": quality["recall"],
                "f1": quality["f1"],
                "tp": quality["tp"], "fp": quality["fp"],
                "fn": quality["fn"], "tn": quality["tn"],
            })
        return pd.DataFrame(rows)

    def plot(self, df: pd.DataFrame, save_to: Union[str, Path] = "fig_contamination_vs_n.png") -> None:
        plot_contamination_vs_n(
            dict(zip(df["n"], df["contamination_rate"])), save_to=save_to
        )


# ======================================================================
# SOLUTION 2: Canary tracking
# ======================================================================

@dataclass
class Exercise2Solution:
    """
    Embed a canary in a private document, train two toy models
    (one on the marked doc, one on an unmarked baseline), and report
    whether each model "extracted" the canary.
    """

    private_doc: str = EXERCISE2_PRIVATE_DOC
    baseline_docs: List[Tuple[str, str, str]] = field(
        default_factory=lambda: [d for d in TRAINING_DOCS if d[2] == "clean"]
    )

    def run(self) -> Dict[str, bool]:
        canary = CanaryString()
        marked = canary.embed_in(self.private_doc)

        # Model A: did *not* see the private document.
        baseline_corpus = self.baseline_docs
        model_a = ToyRetrievalModel(baseline_corpus)

        # Model B: did see the canary-marked private document.
        contaminated_corpus = baseline_corpus + [
            ("PRIVATE", marked, "clean")
        ]
        model_b = ToyRetrievalModel(contaminated_corpus)

        return {
            "trained WITHOUT private doc": canary.was_extracted_by(model_a),
            "trained WITH private doc": canary.was_extracted_by(model_b),
        }

    def plot(self, results: Dict[str, bool], save_to: Union[str, Path] = "fig_canary_extraction.png") -> None:
        plot_canary_extraction(results, save_to=save_to)


# ======================================================================
# SOLUTION 3: Score delta
# ======================================================================

@dataclass
class Exercise3Solution:
    """
    Compare toy-model benchmark scores before and after decontamination.

    Why toy-model? Because contamination's harm is only visible in the
    gap between "score from memorisation" and "score from real learning".
    The toy model has zero learning capacity, so its contaminated score
    is *pure* memorisation; the delta is therefore the leakage exactly.
    """

    benchmark_qa: List[Tuple[str, str, str]] = field(
        default_factory=lambda: list(BENCHMARK_QA)
    )
    full_corpus: List[Tuple[str, str, str]] = field(
        default_factory=lambda: list(TRAINING_DOCS)
    )
    n: int = 5
    threshold: int = 1

    def run(self) -> Dict:
        # 1. Baseline: contaminated training data.
        contaminated_model = ToyRetrievalModel(self.full_corpus)
        contaminated_score = contaminated_model.score(self.benchmark_qa)

        # 2. Decontaminate.
        detector = ContaminationDetector(
            benchmark_qa=self.benchmark_qa, n=self.n, threshold=self.threshold
        )
        kept, removed = Decontaminator(detector).clean(self.full_corpus)

        # 3. Re-score after cleanup.
        clean_model = ToyRetrievalModel(kept)
        decontaminated_score = clean_model.score(self.benchmark_qa)

        # 4. Diagnose: which questions flipped correct -> incorrect.
        flipped: List[Tuple[str, str]] = []
        for qid, question, answer in self.benchmark_qa:
            was_correct = contaminated_model.knows(answer)
            now_correct = clean_model.knows(answer)
            if was_correct and not now_correct:
                flipped.append((qid, answer))

        return {
            "contaminated_score": contaminated_score,
            "decontaminated_score": decontaminated_score,
            "delta": contaminated_score - decontaminated_score,
            "docs_kept": len(kept),
            "docs_removed": len(removed),
            "removed_ids": [d[0] for d in removed],
            "flipped_questions": flipped,
        }

    def plot(self, result: Dict, save_to: Union[str, Path] = "fig_score_delta.png") -> None:
        plot_score_delta(
            result["contaminated_score"],
            result["decontaminated_score"],
            save_to=save_to,
        )


# ======================================================================
# DRIVER
# ======================================================================

def main() -> None:
    """
    Run the full learning module end-to-end:

        0. ELI5 walkthrough.
        1. Scan the synthetic corpus, print metrics + save histogram.
        2. Run all three exercises, print tables, save figures.
        3. Summary of takeaways.
    """
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    # ----- 0. ELI5 ------------------------------------------------------
    eli5_demo()

    # ----- 1. Baseline scan --------------------------------------------
    section("BASELINE SCAN (n=5)")
    detector = ContaminationDetector(benchmark_qa=BENCHMARK_QA, n=5, threshold=1)
    report = detector.scan_corpus(TRAINING_DOCS)
    df = report.to_dataframe()
    print(df.to_string(index=False))
    print()
    print(f"Contamination rate  : {report.contamination_rate():.0%}")
    print(f"Benchmark coverage  : {report.benchmark_coverage()}")
    quality = report.detection_quality()
    print(f"Detection precision : {quality['precision']:.2f}")
    print(f"Detection recall    : {quality['recall']:.2f}")
    print(f"Detection F1        : {quality['f1']:.2f}")
    plot_match_histogram(report)

    # ----- 2a. Exercise 1 ----------------------------------------------
    section("EXERCISE 1 - n-gram size sweep")
    ex1 = Exercise1Solution()
    df1 = ex1.run()
    print(df1.to_string(index=False))
    ex1.plot(df1)

    # ----- 2b. Exercise 2 ----------------------------------------------
    section("EXERCISE 2 - Canary extraction")
    ex2 = Exercise2Solution()
    result2 = ex2.run()
    for label, extracted in result2.items():
        print(f"  {label:35s} -> canary extracted: {extracted}")
    ex2.plot(result2)

    # ----- 2c. Exercise 3 ----------------------------------------------
    section("EXERCISE 3 - Score delta before / after decontamination")
    ex3 = Exercise3Solution()
    result3 = ex3.run()
    print(f"  Contaminated score   : {result3['contaminated_score']:.0%}")
    print(f"  Decontaminated score : {result3['decontaminated_score']:.0%}")
    print(f"  Delta (leakage)      : {result3['delta']:+.0%}")
    print(f"  Docs kept / removed  : {result3['docs_kept']} / {result3['docs_removed']}")
    print(f"  Removed IDs          : {result3['removed_ids']}")
    print(f"  Questions flipped    : {result3['flipped_questions']}")
    ex3.plot(result3)

    # ----- 3. Summary ---------------------------------------------------
    section("SUMMARY")
    print("• 13-gram matching at n=5 catches exact and near duplication")
    print("  but misses paraphrases - the classic production limitation.")
    print("• Increasing n reduces false positives but increases false")
    print("  negatives on short benchmark items (see Exercise 1).")
    print("• Canary strings are a *zero-false-positive* detector for")
    print("  unauthorised training on private data (see Exercise 2).")
    print("• The contaminated-vs-decontaminated delta quantifies exactly")
    print("  how much benchmark score was purchased by leakage (Ex. 3).")
    print("• Production systems replace the set-based index with a Bloom")
    print("  filter (`pybloom-live`) and add fuzzy matching")
    print("  (`datasketch.MinHash` or BM25) to catch paraphrases.")


if __name__ == "__main__":
    main()


# python3 -m venv .venv
# source .venv/bin/activate
# pip3 install numpy pandas matplotlib
# python3 data_contamination.py

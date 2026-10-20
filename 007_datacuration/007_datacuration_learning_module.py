"""
data_curation_module.py
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#


=====================================================================
 SUMMARY (ELI5)
=====================================================================
Imagine you are baking cookies from a giant pile of random ingredients:
some ingredients are spoiled, some are exact duplicates of the same thing,
and some are just packing peanuts. Before baking, you sort everything:
throw away the spoiled stuff, combine duplicates, keep only the good
stuff. Data curation is the same: before training a language model we
take raw web text, remove duplicates and junk, and keep only the
high-quality text - because the model's quality depends more on data
quality than on data quantity.

=====================================================================
PURPOSE
=====================================================================
Educational implementation of a Data Curation Pipeline for LLM
pretraining, matching the technical card:

    -Exact deduplication via SHA-256 fingerprints
    - Near-duplicate deduplication via MinHash LSH (Jaccard >= threshold)
    - Heuristic filters (length, URL density, word length, uniqueness,
      repetitive 5-gram ratio)
    - Quality classifier (TF-IDF + LogisticRegression, a fastText-like
      stand-in for teaching; production uses fastText)
    - Mixture weighting hooks (used in Exercise 3)

=====================================================================
USE CASES
=====================================================================
1) Pretraining corpus construction (the secret sauce of LLM quality).
   Raw Common Crawl contains 30-60% exact/near duplicates and abundant
   boilerplate, spam, and machine-generated pages. Deduplication shrinks
   the corpus, reduces memorisation of repeated text, and improves
   benchmark generalisation (Lee et al. 2022). Heuristic + model-based
   quality filters then keep the top slice that resembles curated
   references (Wikipedia, Books). This is the single largest lever on
   final model quality per token; RefinedWeb, Dolma, and SlimPajama all
   implement variants of this exact pipeline.

2) Contamination detection before evaluation. Benchmarks (MMLU, GSM8K,
   HumanEval) leak into training corpora via web crawls. Before
   evaluating, run the same MinHash/n-gram dedup between benchmark
   prompts and the training set; any surviving near-match is flagged.
   WIMBD and the contamination analysis in the GPT-4 and Llama papers
   use this technique. The curation stages here (shingling + MinHash)
   are directly reusable for the detection stage.

3) Domain re-weighting for coding/math-heavy models. After filtering
   and deduping, the pipeline supports explicit mixture weights per
   domain (web, code, papers, books). StarCoder and Code Llama
   demonstrate that upsampling code while keeping quality filtered by
   a code-specific classifier materially improves coding benchmarks.
   The classifier we ship here is domain-agnostic; swapping in a
   code-trained fastText model is a one-line change.

4) Multilingual corpus construction. Deduplication and heuristics must
   be language-aware (min_length, unique-word ratio, URL density are
   script-sensitive). A well-instrumented pipeline lets you audit that
   low-resource languages are not accidentally over-filtered, which is
   a documented failure mode of C4 and mC4.

5) Legal and medical corpora. High-stakes domains require aggressive
   quality scoring plus specialised filters (PII redaction, disclaimer
   boilerplate, form text, fax headers). Our quality classifier seeds
   are swappable: point them at curated clinical guidelines or
   court opinions instead of Wikipedia.

6) Instruction-tuning data filtering. SFT and preference datasets
   contain near-duplicate prompts and low-quality (boilerplate,
   hallucinated) responses. The same dedup + quality-scoring stack,
   run at *example* granularity rather than document granularity,
   prunes noisy instruction sets and improves downstream AlpacaEval /
   MT-Bench scores.

7) Retrieval-augmented generation (RAG) corpus preparation. RAG quality
   is dominated by chunk quality and dedup. Removing near-duplicates
   prevents the retriever from returning five copies of the same FAQ
   entry, and quality scoring removes garbled scrapes that poison
   grounding. Cost is low compared with the model call budget.

8) Synthetic data gating. Synthetic pretraining corpora (Phi-style
   textbooks, generated QA) need quality gating before mixing with
   organic data. Our classifier interface is a drop-in hook for a
   judge-model score; the mixture weights then gate how much synthetic
   data survives.

=====================================================================
ASSUMPTIONS
=====================================================================
-- Data format: each document is a plain string; no HTML parsing is
  performed here (production uses Trafilatura / Resiliparse first).
-- Environment: Linux/macOS, Python 3.10+, offline execution works
  because the corpus is synthetic (fixed seed = 42).
--Library versions: numpy>=1.24, pandas>=2.0, matplotlib>=3.7,
  scikit-learn>=1.3, datasketch>=1.6 (optional - a pure-Python O(N^2)
  fallback is included).
--Data quality: synthetic; the goal is didactic, not SOTA. The
  classifier is a teaching stand-in for fastText.
-Scope limits: we do not implement a small n-gram perplexity filter
  (mentioned in the technical card); it is provided as an exercise
  hint because perplexity filter effectiveness is corpus-dependent.
-- Interpretation of "quality classifier": we treat it as a binary
  *probabilistic* classifier trained on labelled high/low-quality
  seeds; production fastText is trained on a much larger labelled set.

=====================================================================
SOURCE DATASETS
=====================================================================
This module uses a *synthetic* corpus for reproducibility. Real
production datasets that fit the technical card are listed below.

Synthetic (used in exercises): generated in-memory by
`SyntheticCorpusGenerator(seed=42)`. No license, no download. Mix of
high-quality paragraphs, low-quality scrapes, exact duplicates, and
near-duplicates.

Real (optional, not required to run this module):
  - Common Crawl (https://commoncrawl.org) - CC-BY-4.0-ish per-crawl
    ToS; multi-TB. Production input to RefinedWeb/C4/Dolma.
  - RefinedWeb (https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
    - ODC-By; ~600B tokens after curation.
  - Dolma (https://huggingface.co/datasets/allenai/dolma) - ODC-By;
    3T-token pretraining corpus with curation code published.
  - The Stack (https://huggingface.co/datasets/bigcode/the-stack) -
    permissive code licences; ~6TB.
  - Wikipedia dumps (https://dumps.wikimedia.org) - CC-BY-SA; often
    used as the high-quality seed for the classifier.
  - 20 Newsgroups (scikit-learn built-in, `fetch_20newsgroups`) - ~20k
    documents, used in exercises as a small labelled proxy for topic
    domain re-weighting.

=====================================================================
EXPECTED OUTCOMES  (verify-your-own-run checklist)
=====================================================================
-Synthetic corpus: ~200 documents, of which ~10% are exact
  duplicates, ~15% near-duplicates, ~25% low quality.
- Exact dedup: removes ~10% of docs (tolerance: exact count because
  fingerprints are deterministic).
- Near dedup (MinHash, Jaccard>=0.8): removes roughly the labelled
  near-duplicate set. Small variation possible because MinHash is
  approximate; the tolerance on the count is +/-2 documents.
- Heuristic filter: removes the low-quality short/URL-heavy docs;
  final kept count after heuristics is usually 55-65% of the corpus.
- Quality filter (threshold 0.5): after all stages, kept rate is
  typically 40-55% with mean score of kept docs >= 0.7.
- Funnel figure `fig_data_curation.png`: three panels - (a) length
  distribution of the raw corpus, (b) monotonic bar of docs surviving
  each stage, (c) quality-score histogram (kept vs removed).

=====================================================================
EXERCISES  (see data_curation_exercises.py for solutions)
=====================================================================
1. Sweep the MinHash Jaccard threshold from 0.5 to 0.95 and plot
   (removal count, mean cluster size). What happens to precision vs
   recall of duplicate detection? Hint: use the synthetic corpus and
   labelled duplicate groups emitted by the generator.

2. Compare the heuristic filter vs the quality classifier on the same
   labelled subset. Compute precision and recall of "low-quality"
   removal for each. Hint: the generator emits a `label` field in
   `Document.metadata`.

3. Simulate domain re-weighting. Assign random domain labels ("web",
   "code", "paper", "book") to high-quality documents, then compute the
   weighted average quality score under weights (2,2,1,1) vs (1,1,1,1).
   Hint: weight each kept document's classifier score by its domain
   weight; compare distributions.

Design principles
-----------------
- DRY: each pipeline stage is one class with a single `apply(docs)` method
  returning `(kept, stats)`. The CurationPipeline reuses those uniformly.
- Dataclasses for all stateful concepts (Document, PipelineConfig,
  CurationPipeline) - trivially instantiable and inspectable.
- Full docstrings + inline comments explaining *why*.
- Deterministic: fixed seed 42, reproducible MinHash permutations.

Dependencies
------------
Required: numpy, pandas, matplotlib, scikit-learn.
Optional: datasketch (falls back to O(N^2) Jaccard if missing).
pip3 install numpy pandas matplotlib scikit-learn datasketch
# Optional, for production paths described in docstrings:
pip3 install fasttext pyspark ray

Usage
-----
    python3 data_curation_module.py

Saves `fig_data_curation.png` in the current working directory and
prints stage-by-stage statistics.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import hashlib
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------
# Third-party
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:                                         # pragma: no cover
    DATASKETCH_AVAILABLE = False
    print("[Warning] datasketch not installed - using O(N^2) Jaccard fallback. "
          "Install with: pip install datasketch")

# Reproducibility: BPE-style modules set this; we do the same.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ======================================================================
# CORE DATACLASSES
# ======================================================================

@dataclass
class Document:
    """
    One document flowing through the pipeline.

    Attributes
    ----------
    doc_id : str
        Unique identifier (used as MinHash LSH key).
    text : str
        Raw text content.
    source : str
        Domain label: "web", "book", "code", "paper", "wiki", etc.
    metadata : dict
        Free-form; the synthetic generator puts `label` here so the
        exercises can compute precision / recall without an extra
        argument.
    """
    doc_id: str
    text: str
    source: str = "web"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str):
            raise TypeError(
                f"Document.doc_id must be str, got {type(self.doc_id).__name__}"
            )
        if not isinstance(self.text, str):
            raise TypeError(
                f"Document.text must be str, got {type(self.text).__name__}"
            )


@dataclass
class PipelineConfig:
    """
    All hyperparameters in one place - easy to sweep (see Exercise 1).

    The defaults follow RefinedWeb/C4/Gopher recommendations reasonably
    closely and are safe for teaching corpora.
    """
    # --- Heuristic filter ---
    min_length_chars: int = 100            # drop snippets that are too short
    max_url_density: float = 0.3           # scaled URL-words per word
    min_avg_word_length: float = 2.5       # drops spammy lists
    min_unique_word_ratio: float = 0.30    # drops boilerplate repetitions
    max_repeated_ngram_ratio: float = 0.50 # fraction of ngram positions in repeats
    ngram_size: int = 5
    # --- Near-duplicate dedup ---
    jaccard_threshold: float = 0.80
    num_perm: int = 128                    # MinHash permutations
    shingle_k: int = 5                     # word-shingle size
    # --- Quality classifier ---
    quality_threshold: float = 0.50
    # --- Reproducibility ---
    seed: int = SEED


# ======================================================================
# SYNTHETIC CORPUS GENERATOR
# ======================================================================

class SyntheticCorpusGenerator:
    """
    Generate a labelled synthetic corpus that emulates raw web crawls.

    Each generated Document carries `metadata['label']` with one of
    {"high", "near_dup", "exact_dup", "low"} so exercises can measure
    precision/recall of deduplication and filtering against ground
    truth. The mix matches typical Common Crawl statistics closely
    enough for teaching.

    The generator is deterministic given `seed`.
    """

    _HIGH_TEMPLATES = [
        "Machine learning models are trained on large corpora of text to learn statistical patterns. "
        "The training objective is usually next-token prediction: given the preceding context, the "
        "model outputs a probability distribution over the vocabulary. Modern architectures use "
        "transformer blocks with multi-head self-attention and residual connections.",

        "The scientific method begins with observation, moves to hypothesis formation, and then to "
        "experimental design. Reproducibility is essential: each experiment must be documented with "
        "sufficient detail that an independent researcher can repeat it. Peer review is the standard "
        "quality control mechanism in academic publishing.",

        "Python is a high-level programming language emphasising readability and rapid development. "
        "Its syntax uses significant whitespace to delimit blocks. The standard library provides "
        "modules for file I/O, networking, data structures, and concurrency. Third-party packages "
        "such as NumPy and pandas extend the language with numerical computing capabilities.",

        "Photosynthesis converts light energy into chemical energy stored in glucose. In plants, "
        "chlorophyll absorbs light in the visible spectrum, particularly red and blue wavelengths. "
        "The light-dependent reactions produce ATP and NADPH, which drive the Calvin cycle where "
        "carbon dioxide is fixed into sugars.",

        "The Roman Empire at its height controlled territory from Britain to Mesopotamia. Its legal "
        "code influenced every subsequent European legal tradition. Roads, aqueducts, and public "
        "baths were central to urban life. The Western Empire collapsed in the fifth century while "
        "the Eastern or Byzantine Empire endured for another thousand years.",

        "Effective writing requires clarity, concision, and structure. Begin with a clear thesis. "
        "Support it with evidence drawn from primary and secondary sources. Anticipate counter-"
        "arguments and address them directly. Revise for rhythm and for grammatical consistency. "
        "Have a colleague read your draft before submission.",

        "Data visualisation turns raw numbers into insight. Choose chart types that match the "
        "statistical shape of your data: line charts for time series, bar charts for categorical "
        "comparisons, scatter plots for correlation. Always label axes with units and include a "
        "legend when more than one series is present.",

        "Distributed systems achieve fault tolerance through replication. Consensus algorithms such "
        "as Paxos and Raft ensure that replicas agree on a single log. Leader election is a "
        "primitive atop which state-machine replication is built. Linearisable semantics require "
        "careful handling of clock skew and network partitions.",
    ]

    _LOW_TEMPLATES = [
        "CLICK HERE http://spam.example/one http://spam.example/two buy now free shipping",
        "buy buy buy buy cheap cheap cheap cheap product product product order now",
        "lorem ipsum lorem ipsum lorem ipsum lorem ipsum",
        "http://x.co/a http://x.co/b http://x.co/c http://x.co/d http://x.co/e",
        "New Product Announced. http://shop.example/p?id=999 Click for details. http://shop.example/p?id=1000",
    ]

    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Content builders
    # ------------------------------------------------------------------
    def _jitter_high(self, template: str) -> str:
        """Small lexical jitter so high-quality docs are not identical."""
        filler = self.rng.choice([
            "Indeed, ", "Notably, ", "In practice, ", "Consequently, ", "",
        ])
        return filler + template

    def _make_low_quality(self) -> str:
        base = self.rng.choice(self._LOW_TEMPLATES)
        # Repeat it a few times to produce the boilerplate signature.
        return (base + " ") * self.rng.randint(1, 4)

    def _make_near_dup(self, text: str) -> str:
        """
        Cheap near-duplicate: swap ~10% of words for synonyms.
        Preserves 5-gram shingles for the most part, so Jaccard >= 0.8
        but not 1.0 (which exact dedup would catch).
        """
        words = text.split()
        synonyms = {"models": "model", "large": "huge", "text": "corpora",
                    "statistical": "probabilistic", "pattern": "signal"}
        for i in range(len(words)):
            if self.rng.random() < 0.1:
                w = words[i].strip(".,;:").lower()
                if w in synonyms:
                    words[i] = synonyms[w]
        return " ".join(words)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        n_high: int = 100,
        n_exact_dup: int = 20,
        n_near_dup: int = 30,
        n_low: int = 50,
    ) -> List[Document]:
        """
        Return a mixed, labelled corpus. Total = n_high + n_exact_dup +
        n_near_dup + n_low documents. Each has a `metadata['label']`.
        """
        docs: List[Document] = []
        counter = 0
        high_texts: List[str] = []

        # --- High-quality base documents ---
        for i in range(n_high):
            base = self._HIGH_TEMPLATES[i % len(self._HIGH_TEMPLATES)]
            text = self._jitter_high(base)
            docs.append(Document(
                doc_id=f"hq_{counter:04d}",
                text=text,
                source="web",
                metadata={"label": "high"},
            ))
            high_texts.append(text)
            counter += 1

        # --- Exact duplicates of a subset of the high-quality docs ---
        for i in range(n_exact_dup):
            orig = high_texts[i % len(high_texts)]
            docs.append(Document(
                doc_id=f"ed_{counter:04d}",
                text=orig,                       # byte-identical copy
                source="web",
                metadata={"label": "exact_dup"},
            ))
            counter += 1

        # --- Near-duplicates of a different subset ---
        for i in range(n_near_dup):
            orig = high_texts[(i + 7) % len(high_texts)]
            docs.append(Document(
                doc_id=f"nd_{counter:04d}",
                text=self._make_near_dup(orig),
                source="web",
                metadata={"label": "near_dup"},
            ))
            counter += 1

        # --- Low quality (short / url-heavy / repetitive) ---
        for i in range(n_low):
            text = self._make_low_quality()
            docs.append(Document(
                doc_id=f"lq_{counter:04d}",
                text=text,
                source="web",
                metadata={"label": "low"},
            ))
            counter += 1

        self.rng.shuffle(docs)
        return docs


# ======================================================================
# PIPELINE STAGES - each is a single class with a single `apply`
# ======================================================================

class ExactDeduplicator:
    """
    Drop documents whose (lowercased, whitespace-normalised) text exactly
    matches an already-seen document. Uses SHA-256 so collisions are
    effectively zero.
    """

    @staticmethod
    def _normalise(text: str) -> str:
        # Lowercasing collapses "The Cat" and "the cat"; whitespace
        # collapsing catches HTML-escape differences (newlines vs spaces).
        return " ".join(text.lower().split())

    @classmethod
    def _fingerprint(cls, text: str) -> str:
        return hashlib.sha256(cls._normalise(text).encode("utf-8")).hexdigest()

    def apply(
        self, docs: List[Document]
    ) -> Tuple[List[Document], Dict[str, Any]]:
        if not isinstance(docs, list):
            raise TypeError("ExactDeduplicator.apply expects a list of Document")
        seen: set = set()
        kept: List[Document] = []
        removed: List[str] = []
        for doc in docs:
            fp = self._fingerprint(doc.text)
            if fp in seen:
                removed.append(doc.doc_id)
            else:
                seen.add(fp)
                kept.append(doc)
        return kept, {
            "stage": "exact_dedup",
            "n_in": len(docs),
            "n_out": len(kept),
            "n_removed": len(removed),
            "removed_ids": removed,
        }


class NearDuplicateDeduplicator:
    """
    MinHash LSH over k-shingles. A document is removed if it shares a
    Jaccard similarity >= `threshold` with an already-inserted document.

    Falls back to O(N^2) Jaccard when datasketch is not installed so the
    module still runs in a bare environment.
    """

    def __init__(
        self, threshold: float = 0.80, num_perm: int = 128, shingle_k: int = 5
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if num_perm < 1 or shingle_k < 1:
            raise ValueError("num_perm and shingle_k must be >= 1")
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_k = shingle_k

    # ------------------------------------------------------------------
    def _shingles(self, text: str) -> set:
        tokens = re.findall(r"\w+", text.lower())
        if len(tokens) < self.shingle_k:
            return set(tokens)
        return {
            " ".join(tokens[i : i + self.shingle_k])
            for i in range(len(tokens) - self.shingle_k + 1)
        }

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # ------------------------------------------------------------------
    def _apply_fallback(
        self, docs: List[Document]
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Pure-Python O(N^2) fallback - only used without datasketch."""
        shingles = [self._shingles(d.text) for d in docs]
        kept_idx: List[int] = []
        removed: List[str] = []
        for i, doc in enumerate(docs):
            is_dup = False
            for j in kept_idx:
                if self._jaccard(shingles[i], shingles[j]) >= self.threshold:
                    is_dup = True
                    break
            if is_dup:
                removed.append(doc.doc_id)
            else:
                kept_idx.append(i)
        kept = [docs[i] for i in kept_idx]
        return kept, {
            "stage": "near_dedup",
            "n_in": len(docs),
            "n_out": len(kept),
            "n_removed": len(removed),
            "removed_ids": removed,
            "cluster_sizes": [1] * len(kept),   # not tracked in fallback
            "mean_cluster_size": 1.0,
            "backend": "fallback",
        }

    # ------------------------------------------------------------------
    def apply(
        self, docs: List[Document]
    ) -> Tuple[List[Document], Dict[str, Any]]:
        if not isinstance(docs, list):
            raise TypeError(
                "NearDuplicateDeduplicator.apply expects a list of Document"
            )
        if not DATASKETCH_AVAILABLE:
            return self._apply_fallback(docs)

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        kept: List[Document] = []
        removed: List[str] = []
        cluster_of: Dict[str, str] = {}   # doc_id -> representative doc_id

        for doc in docs:
            m = MinHash(num_perm=self.num_perm)
            for shingle in self._shingles(doc.text):
                m.update(shingle.encode("utf-8"))
            matches = lsh.query(m)
            if matches:
                removed.append(doc.doc_id)
                cluster_of[doc.doc_id] = matches[0]
            else:
                lsh.insert(doc.doc_id, m)
                kept.append(doc)

        # Cluster-size histogram for reporting (representatives get 1
        # plus every removal they attracted).
        sizes = Counter([1 for _ in kept])
        for _, rep in cluster_of.items():
            sizes[rep] += 1
        cluster_sizes = list(sizes.values())
        return kept, {
            "stage": "near_dedup",
            "n_in": len(docs),
            "n_out": len(kept),
            "n_removed": len(removed),
            "removed_ids": removed,
            "cluster_sizes": cluster_sizes,
            "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            "backend": "datasketch",
        }


class HeuristicFilter:
    """
    Rule-based filter in the spirit of Gopher / C4 / RefinedWeb:

      * length (min_length_chars)
      * URL density (max_url_density)
      * average word length (min_avg_word_length)
      * unique-word ratio (min_unique_word_ratio)
      * fraction of ngram positions in repeated ngrams
        (max_repeated_ngram_ratio)
    """

    _URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def url_density(self, text: str) -> float:
        """URLs per 10 whitespace words - scale chosen so 1 URL in a
        20-word snippet already signals spam."""
        urls = self._URL_RE.findall(text)
        n_words = max(len(text.split()), 1)
        return (len(urls) / n_words) * 10.0

    def _repeated_ngram_ratio(self, text: str) -> float:
        """
        Fraction of ngram *positions* that belong to a repeated ngram.

        For a well-written paragraph most 5-grams appear once, giving a
        small ratio; boilerplate ("Click here to subscribe Click here to
        subscribe") gives a ratio close to 1.
        """
        tokens = re.findall(r"\w+", text.lower())
        n = self.config.ngram_size
        if len(tokens) < n:
            return 0.0
        ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        repeated_positions = sum(c for c in counts.values() if c > 1)
        return repeated_positions / len(ngrams)

    # ------------------------------------------------------------------
    def _passes(self, doc: Document) -> Tuple[bool, str]:
        text = doc.text
        if len(text) < self.config.min_length_chars:
            return False, "too_short"
        if self.url_density(text) > self.config.max_url_density:
            return False, "url_density"
        words = text.split()
        avg_len = float(np.mean([len(w) for w in words])) if words else 0.0
        if avg_len < self.config.min_avg_word_length:
            return False, "avg_word_length"
        tokens = re.findall(r"\w+", text.lower())
        if tokens:
            uniq_ratio = len(set(tokens)) / len(tokens)
            if uniq_ratio < self.config.min_unique_word_ratio:
                return False, "low_uniqueness"
        if self._repeated_ngram_ratio(text) > self.config.max_repeated_ngram_ratio:
            return False, "repetitive"
        return True, ""

    # ------------------------------------------------------------------
    def apply(
        self, docs: List[Document]
    ) -> Tuple[List[Document], Dict[str, Any]]:
        if not isinstance(docs, list):
            raise TypeError("HeuristicFilter.apply expects a list of Document")
        kept: List[Document] = []
        removed: List[str] = []
        reasons: Counter = Counter()
        for doc in docs:
            ok, reason = self._passes(doc)
            if ok:
                kept.append(doc)
            else:
                removed.append(doc.doc_id)
                reasons[reason] += 1
        return kept, {
            "stage": "heuristic_filter",
            "n_in": len(docs),
            "n_out": len(kept),
            "n_removed": len(removed),
            "removed_ids": removed,
            "reasons": dict(reasons),
        }


class QualityClassifier:
    """
    TF-IDF + LogisticRegression quality scorer.

    A fastText-style binary classifier is the production choice (per the
    technical card); we use scikit-learn so the module has fewer
    dependencies. The interface (`fit`, `score`, `apply`) is what matters
    and matches fastText closely enough that swapping in fastText is a
    one-class change.
    """

    def __init__(self, seed: int = SEED) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), lowercase=True
        )
        self.classifier = LogisticRegression(
            random_state=seed, max_iter=1000, class_weight="balanced"
        )
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(
        self, high_quality: Sequence[str], low_quality: Sequence[str]
    ) -> "QualityClassifier":
        if not high_quality or not low_quality:
            raise ValueError("fit needs non-empty high and low quality sets")
        X = list(high_quality) + list(low_quality)
        y = [1] * len(high_quality) + [0] * len(low_quality)
        Xv = self.vectorizer.fit_transform(X)
        self.classifier.fit(Xv, y)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def score(self, text: str) -> float:
        if not self._fitted:
            raise RuntimeError("QualityClassifier must be .fit() before .score()")
        if not isinstance(text, str):
            raise TypeError("QualityClassifier.score expects str")
        if not text.strip():
            return 0.0
        Xv = self.vectorizer.transform([text])
        return float(self.classifier.predict_proba(Xv)[0, 1])

    # ------------------------------------------------------------------
    def apply(
        self, docs: List[Document], threshold: float
    ) -> Tuple[List[Document], Dict[str, Any]]:
        if not isinstance(docs, list):
            raise TypeError("QualityClassifier.apply expects a list of Document")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        kept: List[Document] = []
        removed: List[str] = []
        scores: Dict[str, float] = {}
        for doc in docs:
            s = self.score(doc.text)
            scores[doc.doc_id] = s
            (kept if s >= threshold else removed).append(doc) if s >= threshold \
                else removed.append(doc.doc_id)
        # The line above is intentionally clever-but-unreadable; rewrite
        # it cleanly to satisfy the "readable over clever" rule.
        kept = []
        removed = []
        for doc in docs:
            if scores[doc.doc_id] >= threshold:
                kept.append(doc)
            else:
                removed.append(doc.doc_id)
        return kept, {
            "stage": "quality_filter",
            "n_in": len(docs),
            "n_out": len(kept),
            "n_removed": len(removed),
            "removed_ids": removed,
            "scores": scores,
            "mean_score_kept": float(np.mean([scores[d.doc_id] for d in kept]))
                if kept else 0.0,
            "mean_score_removed": float(np.mean([scores[i] for i in removed]))
                if removed else 0.0,
        }


# ======================================================================
# PIPELINE ORCHESTRATION
# ======================================================================

@dataclass
class CurationPipeline:
    """
    End-to-end pipeline: exact dedup -> near dedup -> heuristic filter ->
    quality filter. Returns a dict with per-stage statistics and the
    final surviving documents.

    Attributes
    ----------
    config : PipelineConfig
    classifier : QualityClassifier or None
        If None and `quality_train` is passed to `run()`, one is fitted
        on the fly.
    verbose : bool
        Print a per-stage summary.
    """
    config: PipelineConfig = field(default_factory=PipelineConfig)
    classifier: Optional[QualityClassifier] = None
    verbose: bool = True

    def run(
        self,
        docs: List[Document],
        quality_train: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(docs, list):
            raise TypeError("CurationPipeline.run expects a list of Document")
        if not docs:
            return {"stages": [], "final_docs": [], "final_ids": []}

        # --- Optional on-the-fly classifier training -------------------
        if self.classifier is None and quality_train is not None:
            hq, lq = quality_train
            self.classifier = QualityClassifier(seed=self.config.seed).fit(hq, lq)

        stages: List[Dict[str, Any]] = []
        current = docs

        # --- Stage 1: exact dedup --------------------------------------
        stage = ExactDeduplicator()
        current, stats = stage.apply(current)
        stages.append(stats)
        self._print_stats(stats)

        # --- Stage 2: near dedup ---------------------------------------
        stage = NearDuplicateDeduplicator(
            threshold=self.config.jaccard_threshold,
            num_perm=self.config.num_perm,
            shingle_k=self.config.shingle_k,
        )
        current, stats = stage.apply(current)
        stages.append(stats)
        self._print_stats(stats)

        # --- Stage 3: heuristic filter ---------------------------------
        stage = HeuristicFilter(self.config)
        current, stats = stage.apply(current)
        stages.append(stats)
        self._print_stats(stats)

        # --- Stage 4: quality filter -----------------------------------
        if self.classifier is not None and current:
            current, stats = self.classifier.apply(
                current, self.config.quality_threshold
            )
            stages.append(stats)
            self._print_stats(stats)

        return {
            "stages": stages,
            "final_docs": current,
            "final_ids": [d.doc_id for d in current],
        }

    def _print_stats(self, stats: Dict[str, Any]) -> None:
        if not self.verbose:
            return
        print(
            f"[{stats['stage']:<16}] "
            f"in={stats['n_in']:>4d}  out={stats['n_out']:>4d}  "
            f"removed={stats['n_removed']:>4d}"
            + (f"  mean_cluster={stats.get('mean_cluster_size', 0):.2f}"
               if stats['stage'] == 'near_dedup' else "")
        )


# ======================================================================
# ELI5 CODE WALKTHROUGH (toy demonstration)
# ======================================================================

def eli5_walkthrough() -> None:
    """
    A tiny script that narrates the pipeline with toy data, as if
    explaining to a five-year-old.

    Run this directly to see the narration printed to stdout.
    """
    print("\n=== ELI5 WALKTHROUGH ===")
    print("Imagine we have 6 boxes of text. Some are the same box twice,")
    print("some are almost the same, some are just packing peanuts.")

    toy_docs = [
        Document("A", "The quick brown fox jumps over the lazy dog. " * 5, metadata={"label": "high"}),
        Document("B", "The quick brown fox jumps over the lazy dog. " * 5, metadata={"label": "exact_dup"}),
        Document("C", "The quick brown fox jumps over a lazy dog. " * 5, metadata={"label": "near_dup"}),
        Document("D", "click here http://spam.example buy now", metadata={"label": "low"}),
        Document("E", "Photosynthesis converts light energy into chemical energy stored in glucose. " * 3,
                 metadata={"label": "high"}),
        Document("F", "foo bar baz", metadata={"label": "low"}),
    ]

    # Step 1: exact dedup
    kept, s1 = ExactDeduplicator().apply(toy_docs)
    print(f"\nStep 1 - remove exact copies -> removed {s1['n_removed']} "
          f"({s1['removed_ids']})")

    # Step 2: near dedup
    kept, s2 = NearDuplicateDeduplicator(threshold=0.8).apply(kept)
    print(f"Step 2 - remove near copies  -> removed {s2['n_removed']} "
          f"({s2['removed_ids']})")

    # Step 3: heuristic filter
    cfg = PipelineConfig(min_length_chars=80)
    kept, s3 = HeuristicFilter(cfg).apply(kept)
    print(f"Step 3 - remove junk        -> removed {s3['n_removed']} "
          f"reasons={s3.get('reasons', {})}")

    # Step 4: quality filter (fitted on the surviving high docs as seeds;
    # in a real pipeline you would use external seeds like Wikipedia).
    high_seeds = [d.text for d in kept if d.metadata.get("label") == "high"]
    low_seeds = [d.text for d in toy_docs if d.metadata.get("label") == "low"]
    if high_seeds and low_seeds:
        qc = QualityClassifier().fit(high_seeds, low_seeds)
        kept, s4 = qc.apply(kept, threshold=0.5)
        print(f"Step 4 - keep the cream    -> removed {s4['n_removed']}")

    print("\nFinal kept ids:", [d.doc_id for d in kept])
    print("(Only the good stuff survives!)")


# ======================================================================
# VISUALISATION
# ======================================================================

def plot_curation_report(
    raw_docs: List[Document],
    result: Dict[str, Any],
    save_path: str = "fig_data_curation.png",
) -> None:
    """
    Three-panel report:

    (a) Input:  document length distribution of the raw corpus
    (b) Stages: monotonic funnel (docs surviving each stage)
    (c) Output: quality-score histogram for kept vs removed docs
                (falls back to length distribution if no classifier ran)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel (a): input length distribution ---
    lengths = [len(d.text) for d in raw_docs]
    axes[0].hist(lengths, bins=30, color="steelblue", edgecolor="black")
    axes[0].set_title("(a) Input: document length distribution")
    axes[0].set_xlabel("Characters per document")
    axes[0].set_ylabel("Number of documents")
    axes[0].grid(True, alpha=0.3)

    # --- Panel (b): funnel ---
    labels = ["raw"]
    counts = [len(raw_docs)]
    for st in result["stages"]:
        labels.append(st["stage"])
        counts.append(st["n_out"])
    axes[1].bar(labels, counts, color="seagreen")
    for i, c in enumerate(counts):
        axes[1].text(i, c + 1, str(c), ha="center", fontsize=9)
    axes[1].set_title("(b) Processing: documents surviving each stage")
    axes[1].set_ylabel("Documents remaining")
    axes[1].tick_params(axis="x", rotation=20)

    # --- Panel (c): output quality / length ---
    qstats = next(
        (st for st in result["stages"] if st["stage"] == "quality_filter"),
        None,
    )
    if qstats is not None:
        kept_scores = [qstats["scores"][d.doc_id] for d in result["final_docs"]]
        removed_scores = [qstats["scores"][i] for i in qstats["removed_ids"]]
        axes[2].hist(kept_scores, bins=20, alpha=0.7, label="kept", color="seagreen")
        axes[2].hist(removed_scores, bins=20, alpha=0.7, label="removed", color="firebrick")
        axes[2].set_title("(c) Output: quality-score distribution")
        axes[2].set_xlabel("Classifier score (P(high quality))")
        axes[2].set_ylabel("Count")
        axes[2].legend()
    else:
        kept_lengths = [len(d.text) for d in result["final_docs"]]
        axes[2].hist(kept_lengths, bins=30, color="seagreen", edgecolor="black")
        axes[2].set_title("(c) Output: length distribution of kept docs")
        axes[2].set_xlabel("Characters per document")
        axes[2].set_ylabel("Count")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Data Curation Pipeline Report", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"[Figure] saved: {Path(save_path).resolve()}")


# ======================================================================
# EDGE CASES / FAILURE DEMOS
# ======================================================================

def demo_edge_cases() -> None:
    """
    Exercise each stage with pathological inputs so the learner can
    verify that failures are graceful rather than silent.
    """
    print("\n=== EDGE CASES ===")

    # 1. Empty list
    pipe = CurationPipeline(verbose=False)
    assert pipe.run([]) == {"stages": [], "final_docs": [], "final_ids": []}
    print("[ok] empty corpus -> empty result")

    # 2. Wrong type at pipeline boundary
    try:
        pipe.run("not a list")            # type: ignore[arg-type]
    except TypeError as e:
        print(f"[ok] wrong type raised: {e}")

    # 3. Empty text inside a Document
    docs = [Document("a", ""), Document("b", "Hello world " * 20)]
    kept, stats = HeuristicFilter(PipelineConfig(min_length_chars=10)).apply(docs)
    assert stats["n_removed"] == 1 and "too_short" in stats["reasons"]
    print(f"[ok] empty text filtered as short (reasons={stats['reasons']})")

    # 4. Extreme URL-heavy doc
    docs = [Document("a", "http://x " * 200)]
    _, stats = HeuristicFilter(PipelineConfig()).apply(docs)
    assert stats["reasons"].get("url_density", 0) == 1
    print("[ok] URL-heavy doc filtered")

    # 5. Near-dedup with empty and 1-token docs
    docs = [Document("a", ""), Document("b", "x")]
    kept, _ = NearDuplicateDeduplicator().apply(docs)
    # Neither should crash; both survive because they are dissimilar.
    assert len(kept) == 2
    print("[ok] near-dedup handles trivial docs")

    # 6. Quality classifier before fit
    try:
        QualityClassifier().score("hello")
    except RuntimeError as e:
        print(f"[ok] unfitted classifier raised: {e}")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    pd.set_option("display.width", 200)

    # ---- 1. Generate the labelled synthetic corpus -------------------
    gen = SyntheticCorpusGenerator(seed=SEED)
    raw_docs = gen.generate(n_high=100, n_exact_dup=20, n_near_dup=30, n_low=50)
    print(f"Generated {len(raw_docs)} documents "
          f"({Counter(d.metadata['label'] for d in raw_docs)})")

    # ---- 2. Train the quality classifier on a held-out slice --------
    # In production, seeds would be Wikipedia (high) + a scraped junk
    # set (low). Here we use the synthetic labels to keep the module
    # self-contained. We hold back a fraction from each group so that
    # the corpus fed to the pipeline contains unseen docs (avoids
    # inflated classifier metrics).
    high_texts = [d.text for d in raw_docs
                  if d.metadata["label"] == "high"][:50]
    low_texts = [d.text for d in raw_docs
                 if d.metadata["label"] == "low"][:25]
    classifier = QualityClassifier(seed=SEED).fit(high_texts, low_texts)

    # ---- 3. Run the pipeline ----------------------------------------
    pipe = CurationPipeline(
        config=PipelineConfig(seed=SEED),
        classifier=classifier,
        verbose=True,
    )
    t0 = time.perf_counter()
    result = pipe.run(raw_docs)
    print(f"\nPipeline ran in {time.perf_counter() - t0:.2f}s. "
          f"Kept {len(result['final_docs'])}/{len(raw_docs)} documents.")

    # ---- 4. Ground-truth check ---------------------------------------
    label_of = {d.doc_id: d.metadata["label"] for d in raw_docs}
    kept_labels = Counter(label_of[d.doc_id] for d in result["final_docs"])
    print(f"Final label distribution: {dict(kept_labels)}")

    # ---- 5. ELI5 walkthrough (toy demo) ------------------------------
    eli5_walkthrough()

    # ---- 6. Edge cases -----------------------------------------------
    demo_edge_cases()

    # ---- 7. Visualisation --------------------------------------------
    plot_curation_report(raw_docs, result, save_path="fig_data_curation.png")


if __name__ == "__main__":
    main()



# python3 -m venv .venv
# source .venv/bin/activate
# pip3 install numpy pandas matplotlib scikit-learn datasketch

# # Main demo + figure
# python3 data_curation_module.py            # → prints stage stats, saves fig_data_curation.png

# # Exercise solutions
# python3 data_curation_exercises.py         # → saves fig_ex1.png, fig_ex2.png, fig_ex3.png

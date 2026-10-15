"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#
tfidf_module.py
===============

Purpose
-------
A from-scratch, pedagogically clear implementation of TF-IDF (Term
Frequency - Inverse Document Frequency) with input validation,
edge-case handling, and reproducible behavior. Also wraps the production
scikit-learn implementation so learners can compare.

Design principles
-----------------
- DRY: preprocessing pipeline lives in one place (Tokenizer).
- Single Responsibility: Tokenizer tokenizes; TfidfVectorizer fits/transforms.
- Fail loud: invalid input raises with a helpful message; never silently coerces.
- Reproducibility: deterministic given the same input (no global RNG in fit).

pip install numpy scipy pandas scikit-learn matplotlib
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# Configuration dataclass: one place for all hyperparameters (DRY)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TfidfConfig:
    """Hyperparameters for the TF-IDF vectorizer.

    Attributes
    ----------
    lowercase : bool
        Lowercase all text before tokenizing.
    token_pattern : str
        Regex used to find tokens. Default matches words of length >= 2.
    stop_words : Optional[frozenset[str]]
        Tokens to drop. None keeps everything.
    smooth_idf : bool
        If True, add 1 to numerator and denominator of IDF (prevents div-by-zero
        and zero weights for corpus-wide terms).
    sublinear_tf : bool
        If True, replace tf with 1 + log(tf) to dampen very frequent terms.
    norm : Optional[str]
        'l2' (default), 'l1', or None. Applied to each row of the output matrix.
    min_df : int
        Minimum number of documents a term must appear in to be kept.
    max_df : float
        Maximum fraction of documents a term may appear in; above this it's dropped
        (catches corpus-wide stop words automatically).
    """

    lowercase: bool = True
    token_pattern: str = r"(?u)\b\w\w+\b"
    stop_words: Optional[frozenset[str]] = None
    smooth_idf: bool = True
    sublinear_tf: bool = False
    norm: Optional[str] = "l2"
    min_df: int = 1
    max_df: float = 1.0


# ---------------------------------------------------------------------------
# Document container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Document:
    """A single document: an identifier plus raw text."""
    doc_id: str
    text: str


# ---------------------------------------------------------------------------
# Tokenizer: owns ALL text-preprocessing logic (DRY)
# ---------------------------------------------------------------------------
class Tokenizer:
    """Deterministic, configurable text tokenizer.

    Responsibilities:
        - Lowercasing (optional)
        - Regex tokenization
        - Stop-word removal

    This class owns *all* preprocessing so that the vectorizer never
    re-implements cleaning logic (Single Responsibility + DRY).
    """

    def __init__(self, config: TfidfConfig) -> None:
        self._config = config
        # Compile the regex once; reusing it is faster than recompiling per call.
        self._token_re = re.compile(config.token_pattern)

    def tokenize(self, text: str) -> list[str]:
        """Return a list of tokens for a single piece of text.

        Parameters
        ----------
        text : str
            Raw input text. Must be a non-empty string.

        Returns
        -------
        list[str]
            Tokens in the order they appear, after lowercasing and
            stop-word removal.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"Tokenizer.tokenize expected str, got {type(text).__name__}"
            )
        if self._config.lowercase:
            text = text.lower()
        tokens = self._token_re.findall(text)
        if self._config.stop_words:
            stop = self._config.stop_words
            tokens = [t for t in tokens if t not in stop]
        return tokens


# ---------------------------------------------------------------------------
# Input validation utilities (DRY: used by every public vectorizer method)
# ---------------------------------------------------------------------------
def _validate_documents(documents: Sequence[Document]) -> None:
    """Validate the input collection before fitting or transforming.

    Raises
    ------
    TypeError
        If `documents` is not a sequence of `Document` instances.
    ValueError
        If the sequence is empty.
    """
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise TypeError("`documents` must be a sequence (list/tuple) of Document.")
    if len(documents) == 0:
        raise ValueError("`documents` must contain at least one Document.")
    for i, d in enumerate(documents):
        if not isinstance(d, Document):
            raise TypeError(
                f"Element at index {i} is {type(d).__name__}, expected Document."
            )
        if not isinstance(d.text, str):
            raise TypeError(
                f"Document '{d.doc_id}' has non-str text: {type(d.text).__name__}"
            )


# ---------------------------------------------------------------------------
# Main vectorizer
# ---------------------------------------------------------------------------
class TfidfVectorizer:
    """A from-scratch TF-IDF vectorizer.

    The implementation mirrors the math used by scikit-learn so learners can
    compare outputs directly:

        tf(t, d)    = count(t in d)            [or 1 + log(count) if sublinear]
        idf(t)      = ln((1 + N) / (1 + df(t))) + 1   [smooth_idf=True]
        tfidf(t, d) = tf(t, d) * idf(t)
        row(d)      = tfidf[:, d] / ||tfidf[:, d]||_2  [if norm='l2']

    Attributes (after fit)
    ----------
    vocabulary_ : dict[str, int]
        Token -> column index.
    idf_ : np.ndarray[float]
        Inverse document frequency per token, shape (n_features,).
    """

    def __init__(self, config: Optional[TfidfConfig] = None) -> None:
        self._config: TfidfConfig = config or TfidfConfig()
        self._tokenizer: Tokenizer = Tokenizer(self._config)
        # Learned state
        self.vocabulary_: dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self._is_fit: bool = False

    # -- public API ---------------------------------------------------------
    def fit(self, documents: Sequence[Document]) -> "TfidfVectorizer":
        """Learn vocabulary and IDF weights from `documents`."""
        _validate_documents(documents)
        n_docs = len(documents)

        # 1) Tokenize every document ONCE (DRY: tokenize lives in Tokenizer)
        tokenized = [self._tokenizer.tokenize(d.text) for d in documents]

        # 2) Build document-frequency counts per token.
        df_counts: dict[str, int] = {}
        for tokens in tokenized:
            # Use a set so each token is counted at most once per document.
            for t in set(tokens):
                df_counts[t] = df_counts.get(t, 0) + 1

        # 3) Apply min_df / max_df filtering.
        max_df_abs = self._config.max_df * n_docs
        kept = [
            t for t, df in df_counts.items()
            if df >= self._config.min_df and df <= max_df_abs
        ]
        kept.sort()  # deterministic column ordering
        self.vocabulary_ = {t: i for i, t in enumerate(kept)}

        # 4) Compute IDF (smoothed by default to avoid div-by-zero).
        idf = np.empty(len(kept), dtype=np.float64)
        for t, i in self.vocabulary_.items():
            df = df_counts[t]
            if self._config.smooth_idf:
                idf[i] = math.log((1 + n_docs) / (1 + df)) + 1.0
            else:
                idf[i] = math.log(n_docs / df) + 1.0
        self.idf_ = idf
        self._is_fit = True
        return self

    def transform(self, documents: Sequence[Document]) -> sparse.csr_matrix:
        """Transform documents into a sparse TF-IDF matrix (n_docs, n_features)."""
        if not self._is_fit:
            raise RuntimeError("TfidfVectorizer must be fit() before transform().")
        _validate_documents(documents)

        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        rows, cols, data = [], [], []

        for r, d in enumerate(documents):
            tokens = self._tokenizer.tokenize(d.text)
            if not tokens:
                # Empty document -> row of zeros (still valid output).
                continue
            counts: dict[int, int] = {}
            total = len(tokens)
            for t in tokens:
                col = self.vocabulary_.get(t)
                if col is None:
                    continue  # Out-of-vocabulary token; silently skipped.
                counts[col] = counts.get(col, 0) + 1

            for col, cnt in counts.items():
                tf = cnt
                if self._config.sublinear_tf and tf > 0:
                    tf = 1.0 + math.log(tf)
                else:
                    tf = float(tf) / total  # sklearn-style normalization
                rows.append(r)
                cols.append(col)
                data.append(tf * self.idf_[col])

        mat = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_docs, n_features), dtype=np.float64
        )
        if self._config.norm:
            mat = _normalize(mat, norm=self._config.norm)
        return mat

    def fit_transform(self, documents: Sequence[Document]) -> sparse.csr_matrix:
        """Convenience: fit then transform on the same data."""
        return self.fit(documents).transform(documents)

    def get_feature_names(self) -> list[str]:
        """Return vocabulary as an ordered list of tokens."""
        if not self._is_fit:
            raise RuntimeError("Call fit() first.")
        # Invert vocabulary_ dict, sorted by index.
        return [
            t for t, _ in sorted(self.vocabulary_.items(), key=lambda kv: kv[1])
        ]


# ---------------------------------------------------------------------------
# Row-normalization helper (DRY: shared by L1/L2 choices)
# ---------------------------------------------------------------------------
def _normalize(mat: sparse.csr_matrix, norm: str) -> sparse.csr_matrix:
    """L1 or L2 normalize each row of a sparse matrix, in place-safe."""
    if norm not in {"l1", "l2"}:
        raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")
    if norm == "l2":
        norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    else:
        norms = np.asarray(mat.sum(axis=1)).ravel()
    norms[norms == 0] = 1.0  # avoid div-by-zero for all-zero rows
    # Divide each row by its norm using broadcasting on the sparse matrix.
    inv = 1.0 / norms
    return sparse.diags(inv) @ mat


# ===========================================================================
# DEMO / SCENARIO RUNNER
# ===========================================================================
def scenario_happy_path() -> None:
    """Happy-path: build a small corpus, fit, transform, pretty-print."""
    print("\n" + "=" * 70)
    print("SCENARIO 1: Happy path (small toy corpus)")
    print("=" * 70)
    docs = [
        Document("d1", "the cat sat on the mat"),
        Document("d2", "the dog sat on the log"),
        Document("d3", "cats and dogs are pets"),
    ]
    vec = TfidfVectorizer(TfidfConfig(min_df=1))
    mat = vec.fit_transform(docs)
    feats = vec.get_feature_names()
    df = pd.DataFrame(mat.toarray(), columns=feats, index=[d.doc_id for d in docs])
    print("Vocabulary:", feats)
    print("IDF weights:", dict(zip(feats, vec.idf_.round(3))))
    print("\nTF-IDF matrix:\n", df.round(3))


def scenario_empty_text() -> None:
    """Edge case: a document with empty text -> zero row, no crash."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Empty text document (must not crash)")
    print("=" * 70)
    docs = [
        Document("d1", "alpha beta gamma"),
        Document("d2", ""),  # empty string
        Document("d3", "beta gamma delta"),
    ]
    vec = TfidfVectorizer()
    mat = vec.fit_transform(docs)
    print("Row for empty document d2 (should be all zeros):")
    print(mat.toarray()[1])


def scenario_wrong_type() -> None:
    """Edge case: passing a non-Document object must raise TypeError."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Wrong input type (must raise TypeError)")
    print("=" * 70)
    try:
        TfidfVectorizer().fit([("not", "a", "Document")])
    except TypeError as e:
        print("Correctly raised TypeError:", e)


def scenario_empty_collection() -> None:
    """Edge case: empty corpus must raise ValueError."""
    print("\n" + "=" * 70)
    print("SCENARIO 4: Empty corpus (must raise ValueError)")
    print("=" * 70)
    try:
        TfidfVectorizer().fit([])
    except ValueError as e:
        print("Correctly raised ValueError:", e)


def scenario_nan_text() -> None:
    """Edge case: NaN in text must raise TypeError (no silent coercion)."""
    print("\n" + "=" * 70)
    print("SCENARIO 5: NaN text (must raise TypeError)")
    print("=" * 70)
    try:
        TfidfVectorizer().fit([Document("d1", float("nan"))])  # type: ignore[arg-type]
    except TypeError as e:
        print("Correctly raised TypeError:", e)


def scenario_extreme_values() -> None:
    """Edge case: extremely long document with one dominant token."""
    print("\n" + "=" * 70)
    print("SCENARIO 6: Extreme value (one token repeated 10k times)")
    print("=" * 70)
    long_text = "spam " * 10_000 + "rareword"
    docs = [
        Document("d1", "alpha beta"),
        Document("d2", "beta gamma"),
        Document("d3", long_text),
    ]
    vec = TfidfVectorizer(TfidfConfig(min_df=1, max_df=1.0))
    mat = vec.fit_transform(docs)
    feats = vec.get_feature_names()
    df = pd.DataFrame(mat.toarray(), columns=feats, index=[d.doc_id for d in docs])
    print(df.round(3))
    print("Note: 'spam' has high TF in d3 but low IDF (appears in 1/3 docs).")


def scenario_compare_sklearn() -> None:
    """Sanity check: our output matches scikit-learn's TfidfVectorizer."""
    print("\n" + "=" * 70)
    print("SCENARIO 7: Cross-check against sklearn (should match to 1e-6)")
    print("=" * 70)
    from sklearn.feature_extraction.text import TfidfVectorizer as SkV

    docs = [
        Document("d1", "the cat sat on the mat"),
        Document("d2", "the dog sat on the log"),
        Document("d3", "cats and dogs are pets"),
    ]
    ours = TfidfVectorizer(TfidfConfig()).fit_transform(docs).toarray()
    theirs = SkV().fit_transform([d.text for d in docs]).toarray()
    diff = np.abs(ours - theirs).max()
    print(f"Max absolute difference vs sklearn: {diff:.2e}")
    assert diff < 1e-6, "Outputs diverge!"
    print("Match within tolerance.")


if __name__ == "__main__":
    scenario_happy_path()
    scenario_empty_text()
    scenario_wrong_type()
    scenario_empty_collection()
    scenario_nan_text()
    scenario_extreme_values()
    scenario_compare_sklearn()

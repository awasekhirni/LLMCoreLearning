"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.

tfidf_exercises.py
==================
 TF-IDF learning module exercises.

This script demonstrates:
1. Stop-word removal and its effect on vocabulary and L2 normalization.
2. Sublinear Term Frequency to dampen highly repetitive terms.
3. Cosine similarity retrieval using TF-IDF vectors.

Prerequisites:
- `tfidf_module.py` must be in the same directory.
- Python 3.10+, numpy, pandas, scikit-learn, scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer as SkV

# Import our custom module components
from tfidf_module import Document, TfidfVectorizer, TfidfConfig


# ===========================================================================
# EXERCISE 1: Add Stop-Word Removal
# ===========================================================================
def exercise_1_stop_words() -> None:
    """Demonstrates the impact of removing stop words.

    By filtering out common English connector words ('the', 'on', 'and', 'are'),
    we reduce the vocabulary size. Because TF-IDF rows are L2-normalized at the
    end, removing columns redistributes the mathematical weight to the remaining
    meaningful tokens (like 'cat', 'dog', 'sat').
    """
    print("\n" + "=" * 70)
    print("EXERCISE 1: Stop-Word Removal")
    print("=" * 70)

    docs = [
        Document("d1", "the cat sat on the mat"),
        Document("d2", "the dog sat on the log"),
        Document("d3", "cats and dogs are pets"),
    ]

    # 1a. Baseline (no stop words removed)
    vec_default = TfidfVectorizer(TfidfConfig(min_df=1))
    mat_default = vec_default.fit_transform(docs)
    feats_default = vec_default.get_feature_names()

    # 1b. With Stop Words
    # frozenset is used because TfidfConfig is a frozen dataclass and requires
    # hashable/immutable types for its fields.
    stop_config = TfidfConfig(
        min_df=1,
        stop_words=frozenset({"the", "on", "and", "are"})
    )
    vec_stop = TfidfVectorizer(stop_config)
    mat_stop = vec_stop.fit_transform(docs)
    feats_stop = vec_stop.get_feature_names()

    print(f"Vocabulary size BEFORE stop-word removal: {len(feats_default)}")
    print(f"Vocabulary size AFTER stop-word removal:  {len(feats_stop)}")
    print(f"Removed terms: {set(feats_default) - set(feats_stop)}")

    df_default = pd.DataFrame(
        mat_default.toarray(), columns=feats_default, index=[d.doc_id for d in docs]
    )
    df_stop = pd.DataFrame(
        mat_stop.toarray(), columns=feats_stop, index=[d.doc_id for d in docs]
    )

    print("\nTF-IDF Matrix BEFORE stop-word removal (rounded to 3 decimals):")
    print(df_default.round(3))
    print("\nTF-IDF Matrix AFTER stop-word removal (rounded to 3 decimals):")
    print(df_stop.round(3))

    # Verify L2 norm remains 1.0
    l2_norms = np.sqrt(np.sum(mat_stop.toarray()**2, axis=1))
    print(f"\nL2 norms of rows after stop-word removal: {l2_norms}")
    print("Observation: The rows remain unit vectors (L2 norm = 1.0). The weight")
    print("from removed stop words is mathematically redistributed to content words.")


# ===========================================================================
# EXERCISE 2: Implement Sublinear TF
# ===========================================================================
def exercise_2_sublinear_tf() -> None:
    """Demonstrates Sublinear TF for handling extreme token frequencies.

    Sublinear TF replaces the raw count `c` with `1 + log(c)`. This dampens
    the effect of a token repeated thousands of times, preventing it from
    completely overshadowing other meaningful tokens in the same document.
    """
    print("\n" + "=" * 70)
    print("EXERCISE 2: Sublinear Term Frequency")
    print("=" * 70)

    long_text = "spam " * 10_000 + "rareword"
    docs = [
        Document("d1", "alpha beta"),
        Document("d2", "beta gamma"),
        Document("d3", long_text),
    ]

    # 2a. Default (Raw TF)
    vec_default = TfidfVectorizer(TfidfConfig(min_df=1, max_df=1.0))
    mat_default = vec_default.fit_transform(docs)
    feats = vec_default.get_feature_names()

    # 2b. Sublinear TF
    vec_sub = TfidfVectorizer(TfidfConfig(min_df=1, max_df=1.0, sublinear_tf=True))
    mat_sub = vec_sub.fit_transform(docs)

    # Extract the specific weights for 'spam' in d3
    spam_idx = feats.index("spam")
    rare_idx = feats.index("rareword")

    weight_default_spam = mat_default[2, spam_idx]
    weight_sub_spam = mat_sub[2, spam_idx]
    weight_sub_rare = mat_sub[2, rare_idx]

    print(f"Document d3 contains 'spam' 10,000 times and 'rareword' 1 time.")
    print(f"Raw TF calculation: count = 10000")
    print(f"Sublinear TF calculation: 1 + log(10000) = {1 + math.log(10000):.2f}")

    print(f"\nWeight of 'spam' in d3 (Raw TF):       {weight_default_spam:.4f}")
    print(f"Weight of 'spam' in d3 (Sublinear TF): {weight_sub_spam:.4f}")
    print(f"Weight of 'rareword' in d3 (Sublinear TF): {weight_sub_rare:.4f}")

    # Cross-check with sklearn
    sk_vec = SkV(min_df=1, max_df=1.0, sublinear_tf=True)
    sk_mat = sk_vec.fit_transform([d.text for d in docs]).toarray()
    diff = np.abs(mat_sub.toarray() - sk_mat).max()
    print(f"\nMax absolute difference vs sklearn (sublinear_tf=True): {diff:.2e}")
    print("Observation: Sublinear TF shrinks the massive 'spam' count to a manageable")
    print("logarithmic scale, allowing 'rareword' to receive a proportionally larger")
    print("share of the L2-normalized vector weight.")


# ===========================================================================
# EXERCISE 3: Cosine Similarity Retrieval
# ===========================================================================
@dataclass
class RetrievalResult:
    """Container for a single search result."""
    doc_id: str
    score: float

class DocumentRetriever:
    """A simple search engine using TF-IDF vectors and Cosine Similarity.

    Because our TfidfVectorizer applies L2 normalization by default, the
    dot product between any two vectors is exactly equal to their cosine
    similarity. This makes retrieval extremely fast.
    """

    def __init__(self, vectorizer: TfidfVectorizer, documents: Sequence[Document]) -> None:
        if not vectorizer._is_fit:
            raise RuntimeError("Vectorizer must be fit before passing to retriever.")
        self._vectorizer = vectorizer
        self._documents = documents
        # Precompute and store the document matrix
        self._doc_matrix = vectorizer.transform(documents)

    def retrieve(self, query: str, top_k: int = 1) -> list[RetrievalResult]:
        """Transforms a query and returns the top_k most similar documents.

        Parameters
        ----------
        query : str
            The free-text search query.
        top_k : int
            Number of top results to return.

        Returns
        -------
        list[RetrievalResult]
            List of retrieved documents and their similarity scores, sorted descending.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        # 1. Wrap query in our Document dataclass and transform
        query_doc = Document("query", query)
        query_vec = self._vectorizer.transform([query_doc])  # Shape: (1, n_features)

        # 2. Compute Cosine Similarity
        # Since both query_vec and _doc_matrix are L2-normalized,
        # cosine_similarity(A, B) == dot_product(A, B)
        # Matrix multiplication: (n_docs, n_features) @ (n_features, 1) -> (n_docs, 1)
        scores = self._doc_matrix.dot(query_vec.T).toarray().flatten()

        # 3. Get Top-K indices
        # argsort() returns ascending order, so we take the last top_k elements and reverse
        top_k = min(top_k, len(self._documents))
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 4. Package results
        results = [
            RetrievalResult(doc_id=self._documents[i].doc_id, score=float(scores[i]))
            for i in top_indices
        ]
        return results

def exercise_3_cosine_retrieval() -> None:
    """Demonstrates building a basic search engine over a toy corpus."""
    print("\n" + "=" * 70)
    print("EXERCISE 3: Cosine Similarity Retrieval")
    print("=" * 70)

    docs = [
        Document("d1", "the cat sat on the mat"),
        Document("d2", "the dog sat on the log"),
        Document("d3", "cats and dogs are pets"),
    ]

    # 1. Fit vectorizer and build retriever
    vec = TfidfVectorizer(TfidfConfig(min_df=1))
    vec.fit(docs)
    retriever = DocumentRetriever(vec, docs)

    # 2. Test query
    query = "dog log"
    print(f"Query: '{query}'")
    print("Document Corpus:")
    for d in docs:
        print(f"  - {d.doc_id}: '{d.text}'")

    # 3. Retrieve
    results = retriever.retrieve(query, top_k=2)

    print("\nTop 2 Retrieval Results:")
    for res in results:
        print(f"  Doc ID: {res.doc_id} | Similarity Score: {res.score:.4f}")

    print("\nObservation: d2 ('the dog sat on the log') scores highest because it")
    print("contains both 'dog' and 'log', which are relatively rare in this 3-document")
    print("corpus, giving them high IDF weights.")


# ===========================================================================
# Main Runner
# ===========================================================================
if __name__ == "__main__":
    exercise_1_stop_words()
    exercise_2_sublinear_tf()
    exercise_3_cosine_retrieval()

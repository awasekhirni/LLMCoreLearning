"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
text_classification_exercises.py
=================================

A complete, self-contained Python module implementing solutions for the
three exercises from the Chapter 4 Text Classification lesson.

EXERCISE 1: Add a new evaluation metric (per_class_metrics)
EXERCISE 2: Improve the zero-shot classifier with Sentence-Transformers
EXERCISE 3: Handle the unseen-class edge case with a confidence threshold

Design Principles:
  - DRY: Shared base class and data containers.
  - Input validation and error handling.
  - Reproducibility (RANDOM_SEED = 42).
  - Graceful degradation (if sentence-transformers is missing, falls back
    to TF-IDF but prints a warning).

Author: Senior AI Engineer & Educator
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Any, Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ===========================================================================
# 1. DATA CONTAINERS & BASE CLASS (DRY)
# ===========================================================================

@dataclass
class TextSample:
    """A single piece of text plus its (optional) gold label."""
    text: str
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}")
        if not self.text.strip():
            raise ValueError("text must not be empty or whitespace-only.")
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError(f"label must be str or None, got {type(self.label).__name__}")


@dataclass
class ClassificationResult:
    """Outcome of classifying one TextSample."""
    predicted_label: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)


class TextClassifier:
    """Common interface for every classifier in this module."""
    name: str = "BaseClassifier"

    def fit(self, train_samples: Sequence[TextSample]) -> "TextClassifier":
        raise NotImplementedError

    def predict(self, sample: TextSample) -> ClassificationResult:
        raise NotImplementedError

    def predict_batch(self, samples: Sequence[TextSample]) -> List[ClassificationResult]:
        return [self.predict(s) for s in samples]


# ===========================================================================
# 2. EXERCISE 3: Handle the unseen-class edge case
# ===========================================================================

class TfidfEmbeddingClassifier(TextClassifier):
    """Embedding Model + Classifier (Supervised) with Confidence Thresholding.

    Pipeline: text -> TF-IDF vector -> LogisticRegression -> class label.

    Modification for Ex 3:
    If the maximum probability of the predicted class is below a configurable
    `confidence_threshold`, the model returns "UNKNOWN" instead of guessing.
    """

    name = "TfidfEmbeddingClassifier"

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        random_state: int = RANDOM_SEED,
        confidence_threshold: float = 0.0,  # Ex 3: New parameter
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.confidence_threshold = confidence_threshold

        self.vectorizer_ = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
        )
        self.head_ = LogisticRegression(
            max_iter=1000, random_state=random_state
        )
        self.classes_: Optional[List[str]] = None

    def fit(self, train_samples: Sequence[TextSample]) -> "TfidfEmbeddingClassifier":
        if not train_samples:
            raise ValueError("Cannot fit on an empty training set.")
        if any(s.label is None for s in train_samples):
            raise ValueError("Every training sample must have a label.")

        texts = [s.text for s in train_samples]
        labels = [s.label for s in train_samples]
        X = self.vectorizer_.fit_transform(texts)
        self.head_.fit(X, labels)
        self.classes_ = list(self.head_.classes_)
        return self

    def predict(self, sample: TextSample) -> ClassificationResult:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")

        X = self.vectorizer_.transform([sample.text])
        probs = self.head_.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        max_prob = float(probs[idx])

        scores = {
            self.classes_[i]: float(probs[i]) for i in range(len(self.classes_))
        }

        # Ex 3: Threshold check
        if max_prob < self.confidence_threshold:
            return ClassificationResult(
                predicted_label="UNKNOWN",
                confidence=max_prob,
                all_scores=scores,
            )

        return ClassificationResult(
            predicted_label=self.classes_[idx],
            confidence=max_prob,
            all_scores=scores,
        )


# ===========================================================================
# 3. EXERCISE 2: Improve the zero-shot classifier
# ===========================================================================

class SentenceTransformerZeroShotClassifier(TextClassifier):
    """Zero-Shot Classifier using Sentence-Transformers.

    Replaces the TF-IDF embedder with a real Sentence-Transformer model
    (`all-MiniLM-L6-v2`). Computes cosine similarity between text embeddings
    and label description embeddings.

    Fallback: If `sentence-transformers` is not installed, it reverts to
    TF-IDF to keep the script runnable.
    """

    name = "SentenceTransformerZeroShotClassifier"

    def __init__(self, label_descriptions: Dict[str, str]) -> None:
        if not label_descriptions:
            raise ValueError("label_descriptions must not be empty.")
        self.label_descriptions = dict(label_descriptions)
        self._fitted = False
        self._use_real_st = False
        self._model = None
        self._label_vecs: Dict[str, np.ndarray] = {}

        # Attempt to load real model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_real_st = True
        except ImportError:
            warnings.warn(
                "sentence-transformers not installed. Falling back to TF-IDF. "
                "To see the real improvement, run: pip install sentence-transformers"
            )
            self.vectorizer_ = TfidfVectorizer(stop_words="english")

    def fit(self, train_samples: Sequence[TextSample] = ()) -> "SentenceTransformerZeroShotClassifier":
        if self._use_real_st:
            # Pre-compute label embeddings for efficiency
            descs = list(self.label_descriptions.values())
            vecs = self._model.encode(descs, convert_to_numpy=True)
            for lbl, vec in zip(self.label_descriptions.keys(), vecs):
                self._label_vecs[lbl] = vec
        else:
            # Fallback fit
            corpus = [s.text for s in train_samples] if train_samples else list(self.label_descriptions.values())
            self.vectorizer_.fit(corpus)
        self._fitted = True
        return self

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
        return float(np.dot(a, b) / denom)

    def predict(self, sample: TextSample) -> ClassificationResult:
        if not self._fitted:
            self.fit([])

        if self._use_real_st:
            text_vec = self._model.encode(sample.text, convert_to_numpy=True)
        else:
            text_vec = self.vectorizer_.transform([sample.text]).toarray()[0]

        scores: Dict[str, float] = {}
        for label, desc in self.label_descriptions.items():
            if self._use_real_st:
                desc_vec = self._label_vecs[label]
            else:
                desc_vec = self.vectorizer_.transform([desc]).toarray()[0]
            scores[label] = max(0.0, self._cosine(text_vec, desc_vec))

        best_label = max(scores, key=scores.get)
        return ClassificationResult(
            predicted_label=best_label,
            confidence=scores[best_label],
            all_scores=scores,
        )


# ===========================================================================
# 4. EVALUATION & EXERCISE 1: Add a new evaluation metric
# ===========================================================================

def evaluate_classifier(
    clf: TextClassifier, test_samples: Sequence[TextSample]
) -> Dict[str, Any]:
    """Compute standard metrics. Updated to return y_true and y_pred for Ex 1."""
    y_true = [s.label for s in test_samples]
    y_pred = [clf.predict(s).predicted_label for s in test_samples]
    classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        ),
        "confusion_matrix": cm,
        "classes": classes,
        # Added for Exercise 1:
        "y_true": y_true,
        "y_pred": y_pred,
    }

def per_class_metrics(metrics_dict: Dict[str, Any]) -> pd.DataFrame:
    """Exercise 1: Returns per-class precision, recall, F1 as a DataFrame.

    Drops 'accuracy' row and 'support' column.
    """
    y_true = metrics_dict["y_true"]
    y_pred = metrics_dict["y_pred"]

    # Generate report as a dict
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()

    # Drop the 'accuracy' row and 'support' column as requested
    # Also dropping 'macro avg' and 'weighted avg' to leave ONLY per-class metrics
    rows_to_drop = ['accuracy', 'macro avg', 'weighted avg']
    df = df.drop(index=rows_to_drop, errors='ignore')
    df = df.drop(columns=['support'], errors='ignore')

    return df


# ===========================================================================
# 5. DATA LOADING
# ===========================================================================

def load_dataset(
    categories: Sequence[str] = (
        "rec.sport.baseball",
        "sci.space",
        "talk.politics.mideast",
        "comp.graphics",
    ),
    train_size: int = 100,
    test_size: int = 50,
) -> tuple[list[TextSample], list[TextSample]]:
    """Load a 4-category subset of 20newsgroups and subsample for speed."""
    train_bunch = fetch_20newsgroups(
        subset="train", categories=list(categories),
        remove=("headers", "footers", "quotes"), random_state=RANDOM_SEED,
    )
    test_bunch = fetch_20newsgroups(
        subset="test", categories=list(categories),
        remove=("headers", "footers", "quotes"), random_state=RANDOM_SEED,
    )
    train_full = [
        TextSample(text=t, label=train_bunch.target_names[i])
        for t, i in zip(train_bunch.data, train_bunch.target) if t.strip()
    ]
    test_full = [
        TextSample(text=t, label=test_bunch.target_names[i])
        for t, i in zip(test_bunch.data, test_bunch.target) if t.strip()
    ]
    rng = np.random.RandomState(RANDOM_SEED)
    train_idx = rng.choice(len(train_full), size=min(train_size, len(train_full)), replace=False)
    test_idx = rng.choice(len(test_full), size=min(test_size, len(test_full)), replace=False)
    return [train_full[i] for i in train_idx], [test_full[i] for i in test_idx]


# ===========================================================================
# 6. SCENARIO RUNNERS
# ===========================================================================

def run_exercise_1() -> None:
    """Demonstrates per_class_metrics."""
    print("\n" + "="*60)
    print("EXERCISE 1: Per-Class Evaluation Metrics")
    print("="*60)

    train, test = load_dataset()
    clf = TfidfEmbeddingClassifier()
    clf.fit(train)

    metrics = evaluate_classifier(clf, test)
    df_per_class = per_class_metrics(metrics)

    print("\nPer-Class Metrics DataFrame:")
    print("-"*40)
    print(df_per_class.round(3))
    print("-"*40)
    print(f"DataFrame shape: {df_per_class.shape}")
    print(f"Columns: {list(df_per_class.columns)}")


def run_exercise_2() -> None:
    """Demonstrates SentenceTransformerZeroShotClassifier."""
    print("\n" + "="*60)
    print("EXERCISE 2: Improved Zero-Shot Classifier")
    print("="*60)

    _, test = load_dataset()
    label_descriptions = {
        "rec.sport.baseball": "an article about baseball, bats, pitchers, innings",
        "sci.space": "an article about space, NASA, rockets, planets, astronauts",
        "talk.politics.mideast": "an article about Middle East politics, Israel, Palestine",
        "comp.graphics": "an article about computer graphics, GPUs, rendering, pixels",
    }

    clf = SentenceTransformerZeroShotClassifier(label_descriptions)
    clf.fit([])

    metrics = evaluate_classifier(clf, test)
    print(f"\nZero-Shot Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print("\nNOTE: If you see the fallback warning above, install sentence-transformers")
    print("to see the real improvement over TF-IDF.")


def run_exercise_3() -> None:
    """Demonstrates confidence thresholding for unseen classes."""
    print("\n" + "="*60)
    print("EXERCISE 3: Handling Unseen Classes (Confidence Threshold)")
    print("="*60)

    # Train on only 3 of the 4 classes
    train = [
        TextSample("rocket launch to the moon", "sci.space"),
        TextSample("mars exploration rover", "sci.space"),
        TextSample("home run in the ninth inning", "rec.sport.baseball"),
        TextSample("pitcher threw a fastball", "rec.sport.baseball"),
        TextSample("peace talks in the middle east", "talk.politics.mideast"),
    ]

    # Test with the unseen 4th class
    test_unseen = [
        TextSample("new graphics card renders pixels fast", "comp.graphics"),
        TextSample("GPU rendering pipeline", "comp.graphics"),
    ]

    # Model WITHOUT threshold (default behavior)
    clf_standard = TfidfEmbeddingClassifier()
    clf_standard.fit(train)

    # Model WITH threshold
    clf_threshold = TfidfEmbeddingClassifier(confidence_threshold=0.45)
    clf_threshold.fit(train)

    print("\nPredicting on unseen 'comp.graphics' class:")
    for sample in test_unseen:
        res_std = clf_standard.predict(sample)
        res_thr = clf_threshold.predict(sample)
        print(f"\n  Text: '{sample.text}'")
        print(f"  Standard Model:    predicted='{res_std.predicted_label}' (confidence={res_std.confidence:.3f})")
        print(f"  Threshold Model:   predicted='{res_thr.predicted_label}' (confidence={res_thr.confidence:.3f})")


# ===========================================================================
# 7. ENTRYPOINT
# ===========================================================================

def main() -> None:
    print("="*60)
    print("TEXT CLASSIFICATION EXERCISES MODULE")
    print("="*60)

    run_exercise_1()
    run_exercise_2()
    run_exercise_3()

    print("\n" + "="*60)
    print("All exercises completed successfully.")
    print("="*60)


if __name__ == "__main__":
    main()

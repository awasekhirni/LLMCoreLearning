"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#
#
text_classification_module.py
=============================

A complete, self-contained, runnable learning module for TEXT CLASSIFICATION
using pretrained Language Models (Chapter 4).

Covers (per the technical card):
  Pipeline A — Representation models:
      1. Task-Specific Model (BERT + head)        -> simulated with TF-IDF + MLPClassifier
      2. Embedding Model + Classifier (Supervised)-> TF-IDF + LogisticRegression
      3. Zero-Shot Classification (Unsupervised)   -> TF-IDF + cosine similarity
  Pipeline B — Generative models:
      4. Encoder-Decoder (Flan-T5)                 -> simulated with instruction + keyword scoring
      5. Decoder-Only / API (ChatGPT)              -> simulated; optional real API hook
  Evaluation:
      Confusion matrix, precision, recall, F1, accuracy

Design principles:
  * DRY: one abstract TextClassifier; concrete subclasses share the interface.
  * Input validation lives in the TextSample dataclass.
  * Every public function has a docstring; non-obvious steps are commented.
  * Reproducibility: module-level RANDOM_SEED = 42.

Author: Senior AI Engineer & Educator
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)              # Reproducibility for any numpy-based op
rcParams["figure.figsize"] = (8, 5)
rcParams["axes.grid"] = True
rcParams["figure.dpi"] = 100


# ===========================================================================
# 1. DATA CONTAINERS  (input validation lives here so the rest is DRY)
# ===========================================================================

@dataclass
class TextSample:
    """A single piece of text plus its (optional) gold label.

    Parameters
    ----------
    text : str
        Raw unstructured text. Must be a non-empty, non-whitespace string.
    label : str | None
        Ground-truth class label, or None for unlabeled / inference samples.

    Raises
    ------
    TypeError
        If `text` or `label` has the wrong Python type.
    ValueError
        If `text` is empty or whitespace-only.
    """

    text: str
    label: Optional[str] = None

    def __post_init__(self) -> None:
        # WHY: validate once at construction so downstream code can trust inputs.
        if not isinstance(self.text, str):
            raise TypeError(
                f"text must be a str, got {type(self.text).__name__}"
            )
        if not self.text.strip():
            raise ValueError("text must not be empty or whitespace-only.")
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError(
                f"label must be str or None, got {type(self.label).__name__}"
            )


@dataclass
class ClassificationResult:
    """Outcome of classifying one TextSample.

    Attributes
    ----------
    predicted_label : str
        The class chosen by the model.
    confidence : float
        Probability or similarity score in [0, 1].
    all_scores : dict[str, float]
        Mapping of every candidate label to its score (prob or similarity).
    """

    predicted_label: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)


# ===========================================================================
# 2. ABSTRACT BASE CLASSIFIER  (DRY interface)
# ===========================================================================

class TextClassifier:
    """Common interface for every classifier in this module.

    Subclasses must implement `fit` and `predict`. `predict_batch` is
    provided for free; override it in subclasses if batched inference
    matters for performance.
    """

    name: str = "BaseClassifier"

    def fit(self, train_samples: Sequence[TextSample]) -> "TextClassifier":
        raise NotImplementedError

    def predict(self, sample: TextSample) -> ClassificationResult:
        raise NotImplementedError

    def predict_batch(
        self, samples: Sequence[TextSample]
    ) -> List[ClassificationResult]:
        """Default batched prediction: loop over `predict`."""
        return [self.predict(s) for s in samples]


# ===========================================================================
# 3. CONCRETE CLASSIFIERS
# ===========================================================================

class TfidfEmbeddingClassifier(TextClassifier):
    """Embedding Model + Classifier (Supervised).

    In production you would use `sentence-transformers` to embed text into
    a dense 384-/768-dim vector and then train a Logistic Regression on top.
    For runnable-on-CPU pedagogy we substitute TF-IDF (a sparse embedding);
    the *interface* is identical so swapping is one-line.

    Pipeline: text -> TF-IDF vector -> LogisticRegression -> class label.
    """

    name = "TfidfEmbeddingClassifier"

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        # WHY: fit_transform on train, transform on test -> identical vocab.
        self.vectorizer_ = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
        )
        self.head_ = LogisticRegression(
            max_iter=1000, random_state=random_state
        )
        self.classes_: Optional[List[str]] = None

    def fit(
        self, train_samples: Sequence[TextSample]
    ) -> "TfidfEmbeddingClassifier":
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
        scores = {
            self.classes_[i]: float(probs[i]) for i in range(len(self.classes_))
        }
        return ClassificationResult(
            predicted_label=self.classes_[idx],
            confidence=float(probs[idx]),
            all_scores=scores,
        )


class ZeroShotEmbeddingClassifier(TextClassifier):
    """Zero-Shot Classification (Unsupervised).

    No labeled training data required. The user supplies natural-language
    descriptions of each candidate label (e.g., 'a positive movie review').
    The classifier embeds both the input text and each label description,
    then assigns the label with the highest cosine similarity.

    Pipeline: cosine_similarity(text_vec, label_vecs) -> argmax -> label.
    """

    name = "ZeroShotEmbeddingClassifier"

    def __init__(
        self,
        label_descriptions: Dict[str, str],
        max_features: int = 5000,
    ) -> None:
        if not label_descriptions:
            raise ValueError("label_descriptions must not be empty.")
        self.label_descriptions = dict(label_descriptions)
        self.vectorizer_ = TfidfVectorizer(
            max_features=max_features, stop_words="english"
        )
        self._fitted = False

    def fit(
        self, train_samples: Sequence[TextSample] = ()
    ) -> "ZeroShotEmbeddingClassifier":
        """Fit the TF-IDF vocabulary. Labels are NOT used."""
        # If no train corpus is provided, fit on label descriptions only.
        corpus = (
            [s.text for s in train_samples]
            if train_samples
            else list(self.label_descriptions.values())
        )
        self.vectorizer_.fit(corpus)
        self._fitted = True
        return self

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity in [0, 1] (clamped; TF-IDF vectors are non-negative)."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
        return float(np.dot(a, b) / denom)

    def predict(self, sample: TextSample) -> ClassificationResult:
        if not self._fitted:
            self.fit([])  # auto-fit on label descriptions
        text_vec = self.vectorizer_.transform([sample.text]).toarray()[0]
        scores: Dict[str, float] = {}
        for label, desc in self.label_descriptions.items():
            desc_vec = self.vectorizer_.transform([desc]).toarray()[0]
            scores[label] = max(0.0, self._cosine(text_vec, desc_vec))
        best_label = max(scores, key=scores.get)
        return ClassificationResult(
            predicted_label=best_label,
            confidence=scores[best_label],
            all_scores=scores,
        )


class TaskSpecificHeadClassifier(TextClassifier):
    """Task-Specific Model (simulated).

    In production this would be BERT/RoBERTa with a classification head,
    fine-tuned end-to-end. We approximate the architecture with TF-IDF
    features feeding an MLPClassifier (one hidden layer of 128 units),
    which is small enough to train on CPU in seconds.

    Pipeline: text -> TF-IDF -> MLP(128) -> softmax -> class.
    """

    name = "TaskSpecificHeadClassifier"

    def __init__(
        self, max_features: int = 5000, random_state: int = RANDOM_SEED
    ) -> None:
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer_ = TfidfVectorizer(
            max_features=max_features, stop_words="english"
        )
        self.head_ = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=50,
            random_state=random_state,
            early_stopping=True,
        )
        self.classes_: Optional[List[str]] = None

    def fit(
        self, train_samples: Sequence[TextSample]
    ) -> "TaskSpecificHeadClassifier":
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
        scores = {
            self.classes_[i]: float(probs[i])
            for i in range(len(self.classes_))
        }
        return ClassificationResult(
            predicted_label=self.classes_[idx],
            confidence=float(probs[idx]),
            all_scores=scores,
        )


class GenerativeInstructionClassifier(TextClassifier):
    """Generative Instruction-Based Classifier (simulated Flan-T5 / ChatGPT).

    Mimics how an encoder-decoder (Flan-T5) or decoder-only (GPT) model
    would classify text by *generating the label as text* given an
    instruction prompt. We simulate generation with keyword scoring for
    CPU-runnable pedagogy; the optional `use_real_llm=True` path loads a
    real Flan-T5 via `transformers` if installed.

    Pipeline: instruction + text -> LLM -> generated text -> parse -> label.
    """

    name = "GenerativeInstructionClassifier"

    INSTRUCTION_TEMPLATE = (
        "Classify the following text into one of: {labels}. Text: {text}"
    )

    def __init__(
        self,
        candidate_labels: Sequence[str],
        use_real_llm: bool = False,
        model_name: Optional[str] = None,
    ) -> None:
        self.candidate_labels = list(candidate_labels)
        self.use_real_llm = use_real_llm
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        if use_real_llm:
            try:
                from transformers import (
                    AutoTokenizer,
                    AutoModelForSeq2SeqLM,
                )
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            except Exception as exc:  # pragma: no cover - optional path
                warnings.warn(
                    f"Could not load real LLM ({exc}); falling back to simulation."
                )
                self.use_real_llm = False

    def fit(
        self, train_samples: Sequence[TextSample] = ()
    ) -> "GenerativeInstructionClassifier":
        """Generative models are zero-shot: fit is a no-op (interface symmetry)."""
        return self

    def _simulate_generation(self, text: str) -> str:
        """Approximate LLM generation by keyword-overlap scoring."""
        text_lower = text.lower()
        scores = {lbl: 0 for lbl in self.candidate_labels}
        for lbl in self.candidate_labels:
            for token in re.findall(r"\w+", lbl.lower()):
                scores[lbl] += text_lower.count(token)
        if max(scores.values()) == 0:
            return self.candidate_labels[0]
        return max(scores, key=scores.get)

    def predict(self, sample: TextSample) -> ClassificationResult:
        if self.use_real_llm and self._model is not None:  # pragma: no cover
            prompt = self.INSTRUCTION_TEMPLATE.format(
                labels=", ".join(self.candidate_labels), text=sample.text
            )
            inputs = self._tokenizer(prompt, return_tensors="pt")
            out = self._model.generate(**inputs, max_new_tokens=8)
            generated = self._tokenizer.decode(
                out[0], skip_special_tokens=True
            ).strip()
            predicted = max(
                self.candidate_labels,
                key=lambda lbl: sum(c in generated.lower() for c in lbl.lower()),
            )
            return ClassificationResult(
                predicted_label=predicted, confidence=1.0, all_scores={predicted: 1.0}
            )
        predicted = self._simulate_generation(sample.text)
        return ClassificationResult(
            predicted_label=predicted,
            confidence=1.0,
            all_scores={predicted: 1.0},
        )


# ===========================================================================
# 4. EVALUATION
# ===========================================================================

def evaluate_classifier(
    clf: TextClassifier, test_samples: Sequence[TextSample]
) -> Dict[str, Union[float, np.ndarray, List[str]]]:
    """Compute accuracy, macro precision/recall/F1, and confusion matrix.

    The confusion matrix counts:
        TP = true positive (correctly predicted this class)
        FP = false positive (incorrectly predicted this class)
        FN = false negative (this class missed)
        TN = true negative (correctly predicted something else)

    Precision = TP / (TP + FP)  -- 'of items we labeled X, how many really were X?'
    Recall    = TP / (TP + FN)  -- 'of items that really were X, how many did we catch?'
    F1        = 2 * (P*R) / (P+R) -- harmonic mean
    Accuracy  = (TP+TN) / (TP+TN+FP+FN) -- overall correctness
    """
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
    }


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
    train_size: int = 400,
    test_size: int = 200,
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
        for t, i in zip(train_bunch.data, train_bunch.target)
        if t.strip()  # drop empties from header stripping
    ]
    test_full = [
        TextSample(text=t, label=test_bunch.target_names[i])
        for t, i in zip(test_bunch.data, test_bunch.target)
        if t.strip()
    ]
    rng = np.random.RandomState(RANDOM_SEED)
    train_idx = rng.choice(len(train_full), size=train_size, replace=False)
    test_idx = rng.choice(len(test_full), size=test_size, replace=False)
    return [train_full[i] for i in train_idx], [test_full[i] for i in test_idx]


# ===========================================================================
# 6. SCENARIOS  (happy path + edge cases)
# ===========================================================================

def scenario_happy_path() -> dict:
    """Happy path: train TfidfEmbeddingClassifier, evaluate, return metrics."""
    print("\n--- SCENARIO: Happy Path (Supervised Embedding Classifier) ---")
    train, test = load_dataset()
    print(f"Loaded {len(train)} train / {len(test)} test samples.")

    clf = TfidfEmbeddingClassifier()
    clf.fit(train)
    print(f"Trained {clf.name}. Classes: {clf.classes_}")

    metrics = evaluate_classifier(clf, test)
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        print(f"  {k}: {metrics[k]:.4f}")

    # Single-sample inference demo
    sample = test[0]
    result = clf.predict(sample)
    print(f"  Sample prediction: predicted={result.predicted_label} "
          f"(true={sample.label}, confidence={result.confidence:.3f})")
    return {"clf": clf, "train": train, "test": test, "metrics": metrics}


def scenario_zero_shot() -> dict:
    """Zero-shot: classify the same test set WITHOUT using train labels."""
    print("\n--- SCENARIO: Zero-Shot Classification ---")
    _, test = load_dataset()
    label_descriptions = {
        "rec.sport.baseball": "an article about baseball, bats, pitchers, innings",
        "sci.space": "an article about space, NASA, rockets, planets, astronauts",
        "talk.politics.mideast": "an article about Middle East politics, Israel, Palestine",
        "comp.graphics": "an article about computer graphics, GPUs, rendering, pixels",
    }
    clf = ZeroShotEmbeddingClassifier(label_descriptions)
    clf.fit([])  # fit only on label descriptions; no labels needed
    metrics = evaluate_classifier(clf, test)
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        print(f"  {k}: {metrics[k]:.4f}")
    return {"clf": clf, "test": test, "metrics": metrics}


def scenario_task_specific() -> dict:
    """Task-Specific Model (simulated BERT+head)."""
    print("\n--- SCENARIO: Task-Specific Head Classifier ---")
    train, test = load_dataset(train_size=300, test_size=150)
    clf = TaskSpecificHeadClassifier()
    clf.fit(train)
    metrics = evaluate_classifier(clf, test)
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        print(f"  {k}: {metrics[k]:.4f}")
    return {"clf": clf, "train": train, "test": test, "metrics": metrics}


def scenario_generative() -> dict:
    """Generative instruction-based classifier (simulated)."""
    print("\n--- SCENARIO: Generative Instruction Classifier ---")
    _, test = load_dataset()
    clf = GenerativeInstructionClassifier(
        candidate_labels=[
            "rec.sport.baseball", "sci.space",
            "talk.politics.mideast", "comp.graphics",
        ],
        use_real_llm=False,  # set True + install transformers for real LLM
    )
    metrics = evaluate_classifier(clf, test)
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        print(f"  {k}: {metrics[k]:.4f}")
    return {"clf": clf, "test": test, "metrics": metrics}


def scenario_edge_cases() -> None:
    """Demonstrate input validation and failure modes."""
    print("\n--- SCENARIO: Edge Cases ---")

    # Edge 1: empty / whitespace-only text
    try:
        TextSample(text="   ", label="x")
    except ValueError as e:
        print(f"  [Edge 1 Empty text] caught ValueError: {e}")

    # Edge 2: wrong type for text
    try:
        TextSample(text=12345, label="x")  # type: ignore[arg-type]
    except TypeError as e:
        print(f"  [Edge 2 Wrong type] caught TypeError: {e}")

    # Edge 3: missing label in supervised training
    try:
        TfidfEmbeddingClassifier().fit([TextSample("hello world", None)])
    except ValueError as e:
        print(f"  [Edge 3 Missing label] caught ValueError: {e}")

    # Edge 4: predict before fit
    try:
        TfidfEmbeddingClassifier().predict(TextSample("hello"))
    except RuntimeError as e:
        print(f"  [Edge 4 Not fitted] caught RuntimeError: {e}")

    # Edge 5: extreme input — very long text
    long_text = "data science " * 5000
    sample = TextSample(long_text, label="comp.graphics")
    print(f"  [Edge 5 Long text] accepted (length={len(sample.text)} chars)")

    # Edge 6: unicode / emoji
    emoji_text = "I love this! 😍👍 #rocket 🚀"
    sample2 = TextSample(emoji_text, label="sci.space")
    print(f"  [Edge 6 Emoji/unicode] accepted: '{sample2.text[:40]}...'")

    # Edge 7: class mismatch between train and test (rare class dropped)
    train = [
        TextSample("rocket launch", "sci.space"),
        TextSample("baseball game", "rec.sport.baseball"),
    ]
    test = [TextSample("graphic card", "comp.graphics")]  # unseen in train
    clf = TfidfEmbeddingClassifier(max_features=200)
    clf.fit(train)
    result = clf.predict(test[0])
    print(f"  [Edge 7 Unseen class] predicted={result.predicted_label} "
          f"(true={test[0].label}) — model can only choose among train classes.")


# ===========================================================================
# 7. VISUALIZATION
# ===========================================================================

def make_visualizations(state: dict) -> None:
    """Produce three figures: input distribution, PCA of embeddings, confusion matrix."""
    clf = state["clf"]
    train = state["train"]
    metrics = state["metrics"]

    labels_sorted = sorted(set(s.label for s in train))
    short = lambda l: l.split(".")[-1]

    # ---- Figure 1: input distribution -----------------------------------
    counts = [sum(1 for s in train if s.label == l) for l in labels_sorted]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([short(l) for l in labels_sorted], counts, color="steelblue")
    ax.set_title("Figure 1: Training Sample Counts per Class")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("fig1_input_distribution.png", dpi=120)
    plt.close()

    # ---- Figure 2: PCA of TF-IDF embeddings -----------------------------
    X = clf.vectorizer_.transform([s.text for s in train]).toarray()
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X2 = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for l in labels_sorted:
        mask = np.array([s.label == l for s in train])
        ax.scatter(
            X2[mask, 0], X2[mask, 1],
            label=short(l), alpha=0.6, s=25, edgecolors="none",
        )
    ax.set_title("Figure 2: 2D PCA of TF-IDF Embeddings (Training Samples)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(title="Class")
    plt.tight_layout()
    plt.savefig("fig2_embeddings_pca.png", dpi=120)
    plt.close()

    # ---- Figure 3: confusion matrix ------------------------------------
    classes = metrics["classes"]
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["confusion_matrix"],
        display_labels=[short(c) for c in classes],
    )
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Figure 3: Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig("fig3_confusion_matrix.png", dpi=120)
    plt.close()

    print("  Saved: fig1_input_distribution.png, "
          "fig2_embeddings_pca.png, fig3_confusion_matrix.png")


# ===========================================================================
# 8. ENTRYPOINT
# ===========================================================================

def main() -> None:
    print("=" * 72)
    print("TEXT CLASSIFICATION LEARNING MODULE  (Chapter 4)")
    print("=" * 72)

    happy = scenario_happy_path()
    scenario_zero_shot()
    scenario_task_specific()
    scenario_generative()
    scenario_edge_cases()

    print("\n--- Generating visualizations ---")
    make_visualizations(happy)

    print("\nDone. Review the PNG files saved to your working directory.")


if __name__ == "__main__":
    main()

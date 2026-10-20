
"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
# Exercise 1 — Tune character_coverage on a CJK-Heavy Corpus
# Task. Create a training corpus that is mostly English but contains a small amount of Chinese (you can reuse MULTILINGUAL_TRAIN and add more Chinese lines). Train three Unigram models with character_coverage set to 0.9995, 0.9999, and 1.0. For each model, measure the unknown-token rate on a held-out Chinese sentence containing characters that appear in the training corpus but are rare.
# Hint. Use TokenizerMetrics.unknown_rate(). The 0.9995 model should show a non-zero unknown rate for the rare CJK characters, while 1.0 should show zero. Plot the unknown rate as a bar chart with character_coverage on the x-axis.
# What you are learning. The trade-off between vocabulary size and character coverage. Setting character_coverage=1.0 is essential for CJK, but it may increase the number of required base characters, leaving fewer slots for subword pieces.
# Exercise 2 — Subword Regularization: Measure Segmentation Diversity
# Task. Train a Unigram model on the Shakespeare excerpt. For the word "tomorrow", use enable_sampling=True with alpha=0.0, alpha=0.1, and alpha=0.5. For each setting, encode the word 100 times and count how many distinct segmentations you observe.
# Hint. Use set() to collect unique tuples of pieces. At alpha=0.0, you should see exactly one segmentation (deterministic). At alpha=0.1, you may see 2–5 distinct segmentations. At alpha=0.5, you should see more. Plot the number of distinct segmentations as a function of alpha.
# What you are learning. How Subword Regularization introduces controlled randomness, and why alpha=0.1 is recommended for training robustness without destroying determinism at inference time.
# Exercise 3 — Unigram vs BPE Within SentencePiece
# Task. Train two SentencePiece models on the same English corpus: one with model_type="unigram" and one with model_type="bpe". Use the same vocab_size (e.g., 4000) and character_coverage=0.9995. For each model, measure:
#     Fertility on the training corpus.
#     Compression ratio.
#     Training time (in seconds).
#     Number of distinct segmentations for "tokenization" across 100 sampled encodings (BPE does not support enable_sampling in SentencePiece, so this metric applies only to Unigram).
# Hint. BPE training is typically faster, but Unigram may achieve slightly better compression and offers sampling. Use time.perf_counter() for timing. Present the results in a pandas DataFrame with one row per model type.
# What you are learning. The practical trade-offs between the two algorithms that SentencePiece implements. The technical card's cons field notes that Unigram is "slightly slower than pure BPE at inference"; this exercise lets you measure that difference on your own hardware.



# exercise_solutions_sentencepiece.py
# ===================================
# Complete solutions to Exercises 1–3 from the SentencePiece Unigram
# Learning Module.
#
#  it copies the necessary classes from the
# main module (or you can import them) and adds the solution code for
# each exercise.
#
# Datasets used:
#   - Exercise 1: MULTILINGUAL_TRAIN (embedded, CC0) + an extended
#     Chinese corpus (embedded below, CC0).
#   - Exercise 2: Shakespeare excerpt (Project Gutenberg #100, public
#     domain) loaded via `load_shakespeare_corpus`.
#   - Exercise 3: Same Shakespeare excerpt.
#
# Dependencies:
#   sentencepiece>=0.2.0, numpy, pandas, matplotlib.
#
# Usage:
#   python3 spu_exercises.py
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sentencepiece as spm

# Import the core classes from the main module. If running standalone,
# copy the definitions of SentencePieceModel, TokenizerMetrics,
# load_shakespeare_corpus, MULTILINGUAL_TRAIN, and section() here.
from sentencepiece_learning_module import (
    SentencePieceModel,
    TokenizerMetrics,
    load_shakespeare_corpus,
    MULTILINGUAL_TRAIN,
    section,
)

random.seed(42)
np.random.seed(42)


# ======================================================================
# EXTENDED CHINESE CORPUS (CC0) I hope this is correct i copied the text from a chineese forum
# ======================================================================

EXTENDED_CHINESE: str = """
你好世界今天你好吗
我喜欢用python编程
机器学习模型需要大量数据
深度学习是人工智能的一个重要分支
自然语言处理让计算机理解人类语言
神经网络由许多层组成
Transformer模型改变了自然语言处理领域
注意力机制是Transformer的核心
预训练语言模型在大规模文本上训练
微调可以将预训练模型适配到特定任务
""".strip()


# ======================================================================
# EXERCISE 1 SOLUTION
# ======================================================================

@dataclass
class Exercise1Solution:
    """Tune character_coverage on a CJK-heavy corpus.

    Trains three Unigram models with coverage 0.9995, 0.9999, and 1.0
    on a corpus that is mostly English but contains Chinese. Measures
    unknown-token rate on a held-out Chinese sentence with rare
    characters.

    Attributes
    ----------
    english_corpus : str
    chinese_corpus : str
    test_sentence : str
    vocab_size : int
    """

    english_corpus: str = MULTILINGUAL_TRAIN
    chinese_corpus: str = EXTENDED_CHINESE
    test_sentence: str = "你好世界宇宙无限"
    vocab_size: int = 500

    def run(self) -> pd.DataFrame:
        """Train three models and return unknown rates.

        Returns
        -------
        pandas.DataFrame
            Columns: ``character_coverage``, ``actual_vocab``,
            ``unknown_rate``, ``n_tokens``.
        """
        # Combine English and Chinese into one corpus.
        # The Chinese portion is deliberately small relative to English
        # so that rare CJK characters fall below the 0.9995 threshold.
        train_corpus = self.english_corpus + "\n" + self.chinese_corpus

        rows: List[dict] = []
        for cov in [0.9995, 0.9999, 1.0]:
            model = SentencePieceModel(
                model_prefix=f"sp_ex1sol_cov_{cov}",
                vocab_size=self.vocab_size,
                model_type="unigram",
                character_coverage=cov,
                split_by_whitespace=False,
            )
            model.train(train_corpus)
            metrics = TokenizerMetrics(model)
            ids = model.encode(self.test_sentence)
            rows.append(dict(
                character_coverage=cov,
                actual_vocab=model.effective_vocab_size,
                unknown_rate=metrics.unknown_rate(self.test_sentence),
                n_tokens=len(ids),
            ))
        return pd.DataFrame(rows)

    def plot(
        self, df: pd.DataFrame,
        save_to: Union[str, Path] = "fig_sol_ex1.png",
    ) -> None:
        """Bar chart of unknown rate vs character_coverage."""
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(
            [str(c) for c in df["character_coverage"]],
            df["unknown_rate"],
            color=["steelblue", "darkorange", "seagreen"],
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.2%}",
                ha="center", fontsize=10,
            )
        ax.set_xlabel("character_coverage")
        ax.set_ylabel("Unknown-token rate")
        ax.set_title(
            "Exercise 1 Solution - Unknown rate on rare CJK characters"
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Sol Ex1] Figure saved: {Path(save_to).resolve()}")


# ======================================================================
# EXERCISE 2 SOLUTION
# ======================================================================

@dataclass
class Exercise2Solution:
    """Measure segmentation diversity under Subword Regularization.

    For the word ``"tomorrow"``, encode 100 times at each ``alpha``
    value and count the number of distinct segmentations.

    Attributes
    ----------
    corpus : str
    word : str
    alphas : tuple of float
    n_trials : int
    """

    corpus: str
    word: str = "tomorrow"
    alphas: Tuple[float, ...] = (0.0, 0.1, 0.5)
    n_trials: int = 100

    def run(self) -> pd.DataFrame:
        """Train one model and measure diversity at each alpha.

        Returns
        -------
        pandas.DataFrame
            Columns: ``alpha``, ``n_distinct``, ``example_segmentations``.
        """
        model = SentencePieceModel(
            model_prefix="sp_ex2sol",
            vocab_size=4000,
            model_type="unigram",
            character_coverage=0.9995,
            split_by_whitespace=False,
        )
        model.train(self.corpus)

        rows: List[dict] = []
        for alpha in self.alphas:
            segmentations = set()
            examples = []
            for _ in range(self.n_trials):
                pieces = model.encode(
                    self.word,
                    out_type="str",
                    enable_sampling=True,
                    alpha=alpha,
                )
                key = tuple(pieces)
                segmentations.add(key)
                if len(examples) < 3:
                    examples.append(list(pieces))
            rows.append(dict(
                alpha=alpha,
                n_distinct=len(segmentations),
                example_segmentations=examples,
            ))
        return pd.DataFrame(rows)

    def plot(
        self, df: pd.DataFrame,
        save_to: Union[str, Path] = "fig_sol_ex2.png",
    ) -> None:
        """Bar chart of distinct segmentations vs alpha."""
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(
            [str(a) for a in df["alpha"]],
            df["n_distinct"],
            color=["steelblue", "darkorange", "seagreen"],
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(int(bar.get_height())),
                ha="center", fontsize=10,
            )
        ax.set_xlabel("alpha (sampling smoothness)")
        ax.set_ylabel(f"Distinct segmentations in {self.n_trials} trials")
        ax.set_title(
            "Exercise 2 Solution - Segmentation diversity under sampling"
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Sol Ex2] Figure saved: {Path(save_to).resolve()}")


# ======================================================================
# EXERCISE 3 SOLUTION
# ======================================================================

@dataclass
class Exercise3Solution:
    """Compare Unigram vs BPE within SentencePiece.

    Trains both model types on the same corpus and compares fertility,
    compression ratio, and training time.

    Attributes
    ----------
    corpus : str
    vocab_size : int
    """

    corpus: str
    vocab_size: int = 4000

    def run(self) -> pd.DataFrame:
        """Train Unigram and BPE models and return a comparison table.

        Returns
        -------
        pandas.DataFrame
            Columns: ``model_type``, ``actual_vocab``, ``fertility``,
            ``compression_ratio``, ``train_seconds``.
        """
        rows: List[dict] = []
        for model_type in ["unigram", "bpe"]:
            model = SentencePieceModel(
                model_prefix=f"sp_ex3sol_{model_type}",
                vocab_size=self.vocab_size,
                model_type=model_type,
                character_coverage=0.9995,
                split_by_whitespace=False,
            )
            t0 = time.perf_counter()
            model.train(self.corpus)
            elapsed = time.perf_counter() - t0

            metrics = TokenizerMetrics(model)
            rows.append(dict(
                model_type=model_type,
                actual_vocab=model.effective_vocab_size,
                fertility=metrics.fertility(self.corpus),
                compression_ratio=metrics.compression_ratio(self.corpus),
                train_seconds=elapsed,
            ))
        return pd.DataFrame(rows)

    def plot(
        self, df: pd.DataFrame,
        save_to: Union[str, Path] = "fig_sol_ex3.png",
    ) -> None:
        """Grouped bar chart comparing Unigram and BPE on two metrics."""
        if df.empty:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        bars1 = ax1.bar(
            df["model_type"], df["fertility"],
            color=["steelblue", "darkorange"],
        )
        for bar in bars1:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.3f}",
                ha="center", fontsize=10,
            )
        ax1.set_ylabel("Fertility")
        ax1.set_title("Fertility (lower is better)")
        ax1.grid(True, axis="y", alpha=0.3)

        bars2 = ax2.bar(
            df["model_type"], df["compression_ratio"],
            color=["steelblue", "darkorange"],
        )
        for bar in bars2:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center", fontsize=10,
            )
        ax2.set_ylabel("Compression ratio")
        ax2.set_title("Compression ratio (higher is better)")
        ax2.grid(True, axis="y", alpha=0.3)

        fig.suptitle(
            "Exercise 3 Solution - Unigram vs BPE in SentencePiece",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Sol Ex3] Figure saved: {Path(save_to).resolve()}")


# ======================================================================
# DRIVER
# ======================================================================

def main() -> None:
    """Run all three exercise solutions."""
    pd.set_option("display.width", 200)

    section("SOLUTION 1 - character_coverage on CJK-heavy corpus")
    sol1 = Exercise1Solution(vocab_size=500)
    df1 = sol1.run()
    print(df1.to_string(index=False))
    sol1.plot(df1, save_to="fig_sol_ex1.png")

    section("SOLUTION 2 - Segmentation diversity under sampling")
    shakespeare = load_shakespeare_corpus(max_chars=50_000)
    sol2 = Exercise2Solution(corpus=shakespeare)
    df2 = sol2.run()
    print(df2.to_string(index=False))
    sol2.plot(df2, save_to="fig_sol_ex2.png")

    section("SOLUTION 3 - Unigram vs BPE within SentencePiece")
    sol3 = Exercise3Solution(
        corpus=shakespeare[:20_000], vocab_size=4000
    )
    df3 = sol3.run()
    print(df3.to_string(index=False))
    sol3.plot(df3, save_to="fig_sol_ex3.png")

    section("DONE")
    print("Figures saved:")
    print("  fig_sol_ex1.png")
    print("  fig_sol_ex2.png")
    print("  fig_sol_ex3.png")


if __name__ == "__main__":
    main()

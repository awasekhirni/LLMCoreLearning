# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#
# SentencePiece Unigram Learning Module
# =====================================
# This file is a self-contained educational module for the SentencePiece
# tokenizer with a focus on the Unigram language model. It covers:
#
#   - Training directly on raw text (no pre-tokenization).
#   - Lossless round-trip encoding/decoding using the ▁ meta-symbol.
#   - Subword regularization via sampling.
#   - Character coverage tuning for CJK and multilingual corpora.
#   - Viterbi-based deterministic segmentation.
#
# The module is designed for complete beginners. Every function has a
# docstring, and non-obvious steps are commented with "why" explanations.
#
# Installation:
# python 3 -m venv .   (alternative you can use the new approach using poetry )
#   pip3 install sentencepiece matplotlib numpy pandas
#
# Optional (for Exercise 2 comparison):
#   pip3 install tokenizers
#
# Usage:
#   python3 sentencepiece_learning_module.py
#
# Figures saved:
#   fig_sp_ex1.png  - Fertility vs vocabulary size (English)
#   fig_sp_ex2.png  - Token counts: character_coverage 0.9995 vs 1.0 (multilingual)
#   fig_sp_ex3.png  - Compression ratio (UTF-8 bytes per token)
#
# Dependencies:
#   Python 3.10+, sentencepiece>=0.2.0, numpy, pandas, matplotlib.
#   HuggingFace tokenizers is optional.

from __future__ import annotations

import os
import random
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SentencePiece is the core dependency.
import sentencepiece as spm

# HuggingFace tokenizers is optional; used only for comparison.
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE as HFBPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    HF_AVAILABLE = False
    print("[Warning] HuggingFace `tokenizers` not installed. "
          "Install with: pip install tokenizers")

# Fixed seeds for reproducibility.
random.seed(42)
np.random.seed(42)


# ======================================================================
# CONSTANTS
# ======================================================================

GUTENBERG_URL = "https://www.gutenberg.org/files/100/100-0.txt"

# Public-domain Shakespeare excerpt (~750 words). Used as an offline
# fallback if the Gutenberg download fails.
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
And so he plays his part. The sixth age shifts
Into the lean and slipper'd pantaloon,
With spectacles on nose and pouch on side,
His youthful hose, well saved, a world too wide
For his shrunk shank; and his big manly voice,
Turning again toward childish treble, pipes
And whistles in his sound. Last scene of all,
That ends this strange eventful history,
Is second childishness and mere oblivion,
Sans teeth, sans eyes, sans taste, sans everything.

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

Is this a dagger which I see before me,
The handle toward my hand? Come, let me clutch thee.
I have thee not, and yet I see thee still.
Art thou not, fatal vision, sensible
To feeling as to sight? or art thou but
A dagger of the mind, a false creation,
Proceeding from the heat-oppressed brain?
I see thee yet, in form as palpable
As this which now I draw.

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date;
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.

Let me not to the marriage of true minds
Admit impediments. Love is not love
Which alters when it alteration finds,
Or bends with the remover to remove.
O no! it is an ever-fixed mark
That looks on tempests and is never shaken;
It is the star to every wandering bark,
Whose worth's unknown, although his height be taken.
""".strip()

# Small multilingual training corpus. Deliberately under-represents
# Chinese so that the character_coverage effect is observable in
# Exercise 2.
MULTILINGUAL_TRAIN: str = """
the quick brown fox jumps over the lazy dog
the lazy dog sleeps all day
hello world how are you today
i love programming in python
data science is fun and challenging
machine learning models need lots of data
el rapido zorro marron salta sobre el perro perezoso
el perro perezoso duerme todo el dia
hola mundo como estas hoy
me encanta programar en python
la ciencia de datos es divertida y desafiante
你好世界今天你好吗
我喜欢用python编程
机器学习模型需要大量数据
def foo x return x plus one
print hello
""".strip()

# Held-out test sentences for Exercise 2.
MULTILINGUAL_TEST: List[Tuple[str, str]] = [
    ("english (seen)", "hello world how are you today"),
    ("spanish (seen)", "hola mundo como estas hoy"),
    ("chinese (mixed)", "你好世界宇宙无限"),  # 宇宙无限 are unseen
    ("code (seen)", "def foo x return x plus one"),
    ("emoji (unseen)", "hello 😊🚀 world"),
    ("hindi (unseen)", "नमस्ते दुनिया"),
]


# ======================================================================
# SHARED UTILITIES
# ======================================================================

def load_shakespeare_corpus(
    max_chars: Optional[int] = 50_000,
    prefer_download: bool = True,
    url: str = GUTENBERG_URL,
    timeout_s: float = 10.0,
) -> str:
    """Load a Shakespeare corpus, preferring the full Gutenberg text.

    Parameters
    ----------
    max_chars : int or None
        Truncate the corpus to this many characters. ``None`` disables
        truncation.
    prefer_download : bool
        Try the network first, fall back to the embedded excerpt on any
        error. An offline run therefore always succeeds.
    url : str
        Gutenberg URL for *The Complete Works of William Shakespeare*.
    timeout_s : float
        Network timeout in seconds.

    Returns
    -------
    str
        Raw text, guaranteed non-empty.
    """
    text: Optional[str] = None
    if prefer_download:
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                text = resp.read().decode("utf-8")
        except Exception as exc:  # noqa: BLE001 - fallback is intended
            print(f"[Info] Download failed ({exc}); using embedded excerpt.")
    if text is None:
        text = EMBEDDED_SHAKESPEARE
    return text[:max_chars] if max_chars is not None else text


def section(title: str) -> None:
    """Print a visually distinct section header."""
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


# ======================================================================
# CORE - SentencePiece wrapper (dataclass)
# ======================================================================

@dataclass
class SentencePieceModel:
    """A thin, well-documented wrapper around SentencePiece training and
    inference.

    This class hides the file-based API of ``sentencepiece`` behind a
    stateful object. It supports:

    - Training directly from a raw string (written to a temporary file
      because ``SentencePieceTrainer.train`` requires a file path).
    - Deterministic Viterbi encoding (the default).
    - Probabilistic sampling via ``enable_sampling=True`` and ``alpha``.
    - Lossless decoding.
    - Metric computation via the companion ``TokenizerMetrics`` class.

    Attributes
    ----------
    model_prefix : str
        Prefix for the generated ``.model`` and ``.vocab`` files.
    vocab_size : int
        Target vocabulary size.
    model_type : str
        ``"unigram"`` (default) or ``"bpe"``.
    character_coverage : float
        Fraction of characters to include in the vocabulary. Use ``1.0``
        for CJK corpora; ``0.9995`` is standard for English.
    split_by_whitespace : bool
        If ``False``, whitespace is not used as a hard pre-tokenization
        boundary. This is the key setting that makes SentencePiece
        language-agnostic.
    normalization_rule_name : str
        Unicode normalization rule. ``"nmt_nfkc"`` is the standard
        choice for NMT and LLM tokenizers.
    _processor : Optional[spm.SentencePieceProcessor]
        The underlying SentencePiece processor, populated after
        :meth:`train` or :meth:`load`.
    """

    model_prefix: str = "sp_model"
    vocab_size: int = 8000
    model_type: str = "unigram"
    character_coverage: float = 0.9995
    split_by_whitespace: bool = False
    normalization_rule_name: str = "nmt_nfkc"
    _processor: Optional[spm.SentencePieceProcessor] = field(
        default=None, repr=False, init=False
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, corpus: str, verbose: bool = False) -> None:
        """Train a SentencePiece model on ``corpus`` (a raw string).

        The corpus is written to a temporary file because the
        SentencePiece trainer requires a file path. The temporary file
        is deleted automatically on exit.

        Parameters
        ----------
        corpus : str
            Raw training text. No pre-tokenization is applied.
        verbose : bool
            If ``True``, pass ``minloglevel=0`` to the trainer so it
            prints progress.
        """
        if not isinstance(corpus, str):
            raise TypeError(
                f"corpus must be str, got {type(corpus).__name__}"
            )
        if not corpus.strip():
            raise ValueError("corpus is empty after stripping whitespace.")

        # Write to a temporary file. `delete=False` is used so that the
        # trainer can open the file after the `with` block exits; we
        # delete it explicitly in `finally`.
        tmp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        )
        try:
            tmp.write(corpus)
            tmp.close()

            spm.SentencePieceTrainer.train(
                input=tmp.name,
                model_prefix=self.model_prefix,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
                character_coverage=self.character_coverage,
                split_by_whitespace=self.split_by_whitespace,
                normalization_rule_name=self.normalization_rule_name,
                # `minloglevel=2` silences the trainer; 0 shows progress.
                minloglevel=0 if verbose else 2,
            )

            # Load the trained model into the processor.
            self._processor = spm.SentencePieceProcessor(
                model_file=f"{self.model_prefix}.model"
            )
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def load(self, model_file: str) -> None:
        """Load a pre-trained ``.model`` file.

        Parameters
        ----------
        model_file : str
            Path to a SentencePiece ``.model`` file (e.g.,
            ``"m.model"``).
        """
        if not Path(model_file).is_file():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        self._processor = spm.SentencePieceProcessor(model_file=model_file)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        out_type: str = "int",
        enable_sampling: bool = False,
        alpha: float = 0.1,
    ) -> Union[List[int], List[str]]:
        """Encode ``text`` into subword pieces or integer IDs.

        Parameters
        ----------
        text : str
            Input text. May contain whitespace, punctuation, and any
            Unicode script.
        out_type : str
            ``"int"`` returns vocabulary IDs; ``"str"`` returns piece
            strings (including the ``▁`` meta-symbol).
        enable_sampling : bool
            If ``True``, sample a segmentation from the Unigram lattice
            instead of using Viterbi search. This implements Subword
            Regularization.
        alpha : float
            Sampling smoothness. ``0.0`` = deterministic (no sampling);
            ``0.1`` = slight variation; higher = more randomness.
            Ignored when ``enable_sampling=False``.

        Returns
        -------
        list of int or list of str
        """
        self._require_processor()
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text.strip():
            return []

        return self._processor.encode(
            text,
            out_type=spm.SentencePieceProcessor.out_type_to_id(out_type),
            enable_sampling=enable_sampling,
            alpha=alpha,
        )

    def decode(self, ids: Union[Sequence[int], Sequence[str]]) -> str:
        """Decode token IDs or piece strings back to text.

        SentencePiece guarantees lossless detokenization **with respect
        to the normalized form**: ``decode(encode(text))`` reproduces
        the text after Unicode normalization, with ``▁`` restored to
        spaces.

        Parameters
        ----------
        ids : sequence of int or sequence of str
            Vocabulary IDs (if ints) or piece strings (if strs).

        Returns
        -------
        str
        """
        self._require_processor()
        if not ids:
            return ""
        return self._processor.decode(ids)

    def nbest_encode(
        self, text: str, nbest_size: int = 5
    ) -> List[Tuple[List[str], float]]:
        """Return the top-``nbest_size`` segmentations with log-probabilities.

        This is only meaningful for the **Unigram** model type. It exposes
        the probabilistic nature of Unigram segmentation and is useful
        for ensemble tokenization, uncertainty estimation, and debugging.

        Parameters
        ----------
        text : str
        nbest_size : int

        Returns
        -------
        list of (pieces, log_prob)
        """
        self._require_processor()
        if self.model_type != "unigram":
            raise ValueError(
                "nbest_encode is only supported for model_type='unigram'."
            )
        return self._processor.nbest_encode(
            text, nbest_size=nbest_size, out_type=str
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def effective_vocab_size(self) -> int:
        """Actual vocabulary size of the loaded model."""
        self._require_processor()
        return self._processor.get_piece_size()

    @property
    def unk_id(self) -> int:
        """ID of the ``<unk>`` token (``0`` by default in SentencePiece)."""
        self._require_processor()
        return self._processor.unk_id()

    def _require_processor(self) -> None:
        """Raise a clear error if the model has not been trained or loaded."""
        if self._processor is None:
            raise RuntimeError(
                "No model loaded. Call `.train(corpus)` or `.load(file)` first."
            )


# ======================================================================
# METRICS - DRY: one implementation, reused everywhere
# ======================================================================

@dataclass
class TokenizerMetrics:
    """Compute standard tokenizer metrics for any ``SentencePieceModel``.

    All metrics operate on raw text. No pre-tokenization is assumed.

    Attributes
    ----------
    model : SentencePieceModel
        A trained or loaded model.
    """

    model: SentencePieceModel

    def token_count(self, text: str) -> int:
        """Number of tokens produced by deterministic Viterbi encoding."""
        return len(self.model.encode(text))

    def fertility(self, corpus: str) -> float:
        """Tokens per whitespace-separated word.

        This is the conventional "fertility" metric. For scripts without
        spaces (CJK), use :meth:`tokens_per_char` instead.
        """
        n_words = len(corpus.split())
        return self.token_count(corpus) / n_words if n_words else 0.0

    def tokens_per_char(self, text: str) -> float:
        """Tokens per Unicode code point. Script-agnostic alternative."""
        return self.token_count(text) / len(text) if text else 0.0

    def compression_ratio(self, text: str) -> float:
        """UTF-8 bytes per token. Higher = more information per token.

        UTF-8 **bytes** (not ``len(text)``) is used because a CJK
        character occupies 3 bytes in memory, and that is what actually
        determines model efficiency.
        """
        n_tokens = self.token_count(text)
        return len(text.encode("utf-8")) / n_tokens if n_tokens else 0.0

    def unknown_rate(self, text: str) -> float:
        """Fraction of tokens that are ``<unk>``.

        A well-configured SentencePiece model should have an unknown
        rate of ``0.0``. If it is non-zero, increase
        ``character_coverage`` or enable ``byte_fallback``.
        """
        ids = self.model.encode(text)
        if not ids:
            return 0.0
        unk = self.model.unk_id
        return sum(1 for i in ids if i == unk) / len(ids)


# ======================================================================
# EXERCISE 1 - Vocabulary-size tuning (English)
# ======================================================================

@dataclass
class Exercise1Solution:
    """Train SentencePiece Unigram at several vocabulary sizes on
    English text and measure fertility and compression.

    Unlike BPE, SentencePiece **cannot** snapshot a single training run
    at multiple vocabulary sizes: the Unigram pruning schedule is
    internal to the C++ trainer. This exercise therefore trains a
    separate model for each target size. A 50,000-character Shakespeare
    excerpt keeps each run under a few seconds.

    Attributes
    ----------
    corpus : str
        Raw English training text.
    vocab_targets : tuple of int
        Vocabulary sizes to evaluate.
    """

    corpus: str
    vocab_targets: Tuple[int, ...] = (1000, 2000, 4000, 8000)

    def run(self) -> pd.DataFrame:
        """Train one model per target size and return a metrics table.

        Returns
        -------
        pandas.DataFrame
            Columns: ``vocab_size``, ``actual_vocab``, ``fertility``,
            ``tokens_per_char``, ``compression_ratio``,
            ``unknown_rate``, ``train_seconds``.
        """
        rows: List[dict] = []
        for target in sorted(self.vocab_targets):
            prefix = f"sp_ex1_{target}"
            model = SentencePieceModel(
                model_prefix=prefix,
                vocab_size=target,
                model_type="unigram",
                character_coverage=0.9995,  # English-dominant
                split_by_whitespace=False,
            )
            t0 = time.perf_counter()
            try:
                model.train(self.corpus)
            except RuntimeError as exc:
                # Training can fail if the target vocab_size is too
                # small relative to the number of required characters.
                # Record the failure and continue.
                print(f"[Ex1] vocab_size={target} failed: {exc}")
                rows.append(dict(
                    vocab_size=target, actual_vocab=np.nan,
                    fertility=np.nan, tokens_per_char=np.nan,
                    compression_ratio=np.nan, unknown_rate=np.nan,
                    train_seconds=np.nan,
                ))
                continue
            elapsed = time.perf_counter() - t0

            metrics = TokenizerMetrics(model)
            rows.append(dict(
                vocab_size=target,
                actual_vocab=model.effective_vocab_size,
                fertility=metrics.fertility(self.corpus),
                tokens_per_char=metrics.tokens_per_char(self.corpus),
                compression_ratio=metrics.compression_ratio(self.corpus),
                unknown_rate=metrics.unknown_rate(self.corpus),
                train_seconds=elapsed,
            ))
        return pd.DataFrame(rows)

    def plot(
        self, df: pd.DataFrame, save_to: Union[str, Path] = "fig_sp_ex1.png"
    ) -> None:
        """Plot fertility and compression ratio vs vocabulary size."""
        ok = df.dropna(subset=["fertility"])
        if ok.empty:
            print("[Ex1] No successful runs - nothing to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(ok["actual_vocab"], ok["fertility"],
                 marker="o", color="steelblue")
        ax1.set_title("Exercise 1 - Fertility vs Vocabulary Size")
        ax1.set_xlabel("Actual vocabulary size")
        ax1.set_ylabel("Fertility (tokens / whitespace word)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(ok["actual_vocab"], ok["compression_ratio"],
                 marker="s", color="darkorange")
        ax2.set_title("Exercise 1 - Compression Ratio vs Vocabulary Size")
        ax2.set_xlabel("Actual vocabulary size")
        ax2.set_ylabel("Compression ratio (UTF-8 bytes / token)")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex1] Figure saved: {Path(save_to).resolve()}")


# ======================================================================
# EXERCISE 2 - Character coverage: 0.9995 vs 1.0 (multilingual)
# ======================================================================

@dataclass
class Exercise2Solution:
    """Train two Unigram models on the same multilingual corpus—one with
    ``character_coverage=0.9995`` and one with ``character_coverage=1.0``
    —and compare their behavior on held-out sentences.

    This exercise makes the CJK character-coverage lesson concrete. With
    ``0.9995`` on a corpus containing Chinese, some rare CJK characters
    may be dropped and mapped to ``<unk>``. With ``1.0``, every character
    in the training corpus is covered.

    Attributes
    ----------
    train_corpus : str
        Raw multilingual training text.
    test_set : list of (label, text)
        Held-out sentences for evaluation.
    vocab_size : int
        Target vocabulary size for both models.
    """

    train_corpus: str = MULTILINGUAL_TRAIN
    test_set: List[Tuple[str, str]] = field(
        default_factory=lambda: list(MULTILINGUAL_TEST)
    )
    vocab_size: int = 500

    def run(self) -> pd.DataFrame:
        """Train both models and encode every test case.

        Returns
        -------
        pandas.DataFrame
            One row per test case, with token counts and unknown-token
            flags for both coverage settings.
        """
        # -- Train the low-coverage model (0.9995) --
        low_model = SentencePieceModel(
            model_prefix="sp_ex2_low",
            vocab_size=self.vocab_size,
            model_type="unigram",
            character_coverage=0.9995,
            split_by_whitespace=False,
        )
        low_model.train(self.train_corpus)

        # -- Train the full-coverage model (1.0) --
        high_model = SentencePieceModel(
            model_prefix="sp_ex2_high",
            vocab_size=self.vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            split_by_whitespace=False,
        )
        high_model.train(self.train_corpus)

        rows: List[dict] = []
        for label, text in self.test_set:
            low_ids = low_model.encode(text)
            low_decoded = low_model.decode(low_ids)
            low_unk = low_model.unk_id in low_ids
            low_lossless = (low_decoded == text)

            high_ids = high_model.encode(text)
            high_decoded = high_model.decode(high_ids)
            high_unk = high_model.unk_id in high_ids
            high_lossless = (high_decoded == text)

            rows.append(dict(
                case=label,
                text=text,
                low_tokens=len(low_ids),
                low_decoded=low_decoded,
                low_has_unk=low_unk,
                low_lossless=low_lossless,
                high_tokens=len(high_ids),
                high_decoded=high_decoded,
                high_has_unk=high_unk,
                high_lossless=high_lossless,
            ))
        return pd.DataFrame(rows)

    def plot(
        self, df: pd.DataFrame, save_to: Union[str, Path] = "fig_sp_ex2.png"
    ) -> None:
        """Grouped bar chart of token counts per test case."""
        if df.empty:
            return
        x = np.arange(len(df))
        w = 0.4
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.bar(x - w / 2, df["low_tokens"], w,
               label="character_coverage=0.9995", color="steelblue")
        ax.bar(x + w / 2, df["high_tokens"], w,
               label="character_coverage=1.0", color="seagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(df["case"], rotation=25, ha="right")
        ax.set_ylabel("Number of tokens")
        ax.set_title("Exercise 2 - Token counts by character coverage (Unigram)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex2] Figure saved: {Path(save_to).resolve()}")


# ======================================================================
# EXERCISE 3 - Compression ratio (UTF-8 bytes / token)
# ======================================================================

@dataclass
class Exercise3Solution:
    """Compute the compression ratio of a trained SentencePiece model
    on the same corpus used for training.

    ``compression_ratio = utf8_bytes(text) / token_count``. Higher is
    better: each token carries more information, so sequences are shorter
    and inference is cheaper.

    Attributes
    ----------
    train_corpus : str
    test_corpus : str or None
        Defaults to ``train_corpus`` if ``None``.
    vocab_size : int
    """

    train_corpus: str
    test_corpus: Optional[str] = None
    vocab_size: int = 8000

    def __post_init__(self) -> None:
        if self.test_corpus is None:
            self.test_corpus = self.train_corpus

    def run(self) -> pd.DataFrame:
        """Train a Unigram model and return its compression ratio.

        Returns
        -------
        pandas.DataFrame
            One row with ``model_type``, ``vocab_size``, ``n_tokens``,
            ``n_bytes``, and ``compression_ratio``.
        """
        model = SentencePieceModel(
            model_prefix="sp_ex3",
            vocab_size=self.vocab_size,
            model_type="unigram",
            character_coverage=0.9995,
            split_by_whitespace=False,
        )
        model.train(self.train_corpus)
        metrics = TokenizerMetrics(model)

        rows: List[dict] = [dict(
            model_type="unigram",
            vocab_size=model.effective_vocab_size,
            n_tokens=metrics.token_count(self.test_corpus),
            n_bytes=len(self.test_corpus.encode("utf-8")),
            compression_ratio=metrics.compression_ratio(self.test_corpus),
        )]

        # Optional: compare with HuggingFace byte-level BPE if available.
        if HF_AVAILABLE:
            hf = self._train_hf_bytelevel()
            hf_n_tokens = len(hf.encode(self.test_corpus).ids)
            rows.append(dict(
                model_type="hf-bytelevel-bpe",
                vocab_size=hf.get_vocab_size(),
                n_tokens=hf_n_tokens,
                n_bytes=len(self.test_corpus.encode("utf-8")),
                compression_ratio=(
                    len(self.test_corpus.encode("utf-8")) / hf_n_tokens
                    if hf_n_tokens else 0.0
                ),
            ))
        return pd.DataFrame(rows)

    def _train_hf_bytelevel(self) -> Tokenizer:
        """Train a HuggingFace byte-level BPE for comparison (optional)."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        )
        try:
            tmp.write(self.train_corpus)
            tmp.close()
            tok = Tokenizer(HFBPE(unk_token="<unk>"))
            tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["<unk>"],
                min_frequency=2,
                show_progress=False,
            )
            tok.train(files=[tmp.name], trainer=trainer)
            return tok
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def plot(
        self, df: pd.DataFrame, save_to: Union[str, Path] = "fig_sp_ex3.png"
    ) -> None:
        """Bar chart comparing compression ratios."""
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(df["model_type"], df["compression_ratio"],
                      color=["steelblue", "seagreen"][: len(df)])
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center", fontsize=10,
            )
        ax.set_ylabel("Compression ratio (UTF-8 bytes / token)")
        ax.set_title("Exercise 3 - Compression Ratio (higher is better)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex3] Figure saved: {Path(save_to).resolve()}")


# ======================================================================
# EDGE CASES AND FAILURE SCENARIOS
# ======================================================================

def edge_case_empty_input(model: SentencePieceModel) -> None:
    """Empty and whitespace-only input should return an empty list, not
    crash."""
    print("\n[Edge] Empty input:")
    for text in ["", "   ", "\n\t"]:
        result = model.encode(text)
        assert result == [], f"Expected [], got {result!r} for {text!r}"
        print(f"  {text!r} -> {result} (OK)")


def edge_case_wrong_type(model: SentencePieceModel) -> None:
    """Passing a non-string should raise ``TypeError`` with a clear
    message."""
    print("\n[Edge] Wrong type:")
    for bad in [42, None, ["hello"]]:
        try:
            model.encode(bad)
        except TypeError as exc:
            print(f"  {bad!r} -> TypeError: {exc} (OK)")
        else:
            raise AssertionError(f"Expected TypeError for {bad!r}")


def edge_case_extreme_vocab(corpus: str) -> None:
    """Very small and very large vocabulary sizes should either succeed
    or fail gracefully with a clear error message."""
    print("\n[Edge] Extreme vocabulary sizes:")
    for vs in [4, 1_000_000]:
        model = SentencePieceModel(
            model_prefix=f"sp_edge_{vs}",
            vocab_size=vs,
            model_type="unigram",
            character_coverage=1.0,  # avoid dropping characters
            split_by_whitespace=False,
        )
        try:
            model.train(corpus[:2000])  # small slice for speed
            print(f"  vocab_size={vs} -> trained, "
                  f"effective={model.effective_vocab_size} (OK)")
        except RuntimeError as exc:
            print(f"  vocab_size={vs} -> RuntimeError: {exc} (handled)")


def edge_case_unk_rate() -> None:
    """A model trained without a script should produce <unk> tokens for
    that script. This is the failure mode that ``character_coverage=1.0``
    and ``byte_fallback`` are designed to prevent."""
    print("\n[Edge] Unknown-token rate on unseen script:")
    # Train only on English; test on Hindi.
    english_only = "the quick brown fox jumps over the lazy dog"
    model = SentencePieceModel(
        model_prefix="sp_edge_unk",
        vocab_size=100,
        model_type="unigram",
        character_coverage=1.0,  # cover all *training* characters
        split_by_whitespace=False,
    )
    model.train(english_only)
    metrics = TokenizerMetrics(model)
    hindi = "नमस्ते दुनिया"
    rate = metrics.unknown_rate(hindi)
    print(f"  Hindi text with English-only model: unknown_rate={rate:.2%}")
    if rate > 0:
        print("  (Expected: unseen characters become <unk>.)")


# ======================================================================
# ELI5 CODE WALKTHROUGH
# ======================================================================

def eli5_walkthrough() -> None:
    """A simplified, heavily narrated script that demonstrates SentencePiece
    Unigram with toy data, as if speaking to a five-year-old.

    Run this function directly to see the step-by-step explanation in the
    console. No figures are generated.
    """
    print("=" * 60)
    print("ELI5 WALKTHROUGH: How SentencePiece Unigram works")
    print("=" * 60)

    # Step 1: A toy corpus with three languages and no spaces in Chinese.
    toy_corpus = """
    hello world
    hello there
    the world is big
    你好世界
    你好朋友
    """
    print("\nStep 1: Here is our tiny training text:")
    print(toy_corpus.strip())
    print("(Notice: Chinese has NO spaces between words!)")

    # Step 2: Train a small model.
    print("\nStep 2: We train a SentencePiece model on this raw text.")
    print("SentencePiece does NOT need us to put spaces between words.")
    model = SentencePieceModel(
        model_prefix="sp_eli5",
        vocab_size=50,
        model_type="unigram",
        character_coverage=1.0,  # cover every character, including CJK
        split_by_whitespace=False,
    )
    model.train(toy_corpus)
    print(f"Done! The model learned {model.effective_vocab_size} pieces.")

    # Step 3: Encode a sentence.
    sentence = "hello world"
    pieces = model.encode(sentence, out_type="str")
    ids = model.encode(sentence, out_type="int")
    print(f"\nStep 3: Let's encode the sentence: {sentence!r}")
    print(f"  As pieces: {pieces}")
    print(f"  As IDs:    {ids}")
    print("(The ▁ symbol means 'there was a space here'.)")

    # Step 4: Decode back.
    decoded = model.decode(ids)
    print(f"\nStep 4: Decode the IDs back: {decoded!r}")
    print(f"  Is it the same? {decoded == sentence}")

    # Step 5: Show a CJK example.
    cjk = "你好世界"
    cjk_pieces = model.encode(cjk, out_type="str")
    print(f"\nStep 5: Encode Chinese (no spaces!): {cjk!r}")
    print(f"  Pieces: {cjk_pieces}")
    print("SentencePiece found the pieces without needing spaces.")

    # Step 6: Show sampling.
    print("\nStep 6: Because Unigram is probabilistic, we can ask for")
    print("different ways to slice the same sentence:")
    for i in range(3):
        sampled = model.encode(
            "hello world", out_type="str",
            enable_sampling=True, alpha=0.5,
        )
        print(f"  Try {i + 1}: {sampled}")
    print("(This randomness helps models become more robust.)")

    print("\n" + "=" * 60)
    print("That's SentencePiece Unigram! It cuts text into pieces—")
    print("even when there are no spaces—by finding the most probable")
    print("way to slice it.")
    print("=" * 60)


# ======================================================================
# DRIVER
# ======================================================================

def main() -> None:
    """Run all exercises, edge cases, and the ELI5 walkthrough."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 55)

    # ---------------------------------------------------------------- ELI5
    eli5_walkthrough()

    # ---------------------------------------------------------------- 1
    section("EXERCISE 1 - Vocabulary-size tuning (English, Unigram)")
    shakespeare = load_shakespeare_corpus(max_chars=50_000, prefer_download=True)
    print(f"Corpus length: {len(shakespeare):,} chars "
          f"({len(shakespeare.split()):,} whitespace words)")

    ex1 = Exercise1Solution(corpus=shakespeare)
    df1 = ex1.run()
    print("\nResults:")
    print(df1.to_string(index=False))
    ex1.plot(df1, save_to="fig_sp_ex1.png")

    # ---------------------------------------------------------------- 2
    section("EXERCISE 2 - Character coverage: 0.9995 vs 1.0 (multilingual)")
    ex2 = Exercise2Solution(vocab_size=500)
    df2 = ex2.run()
    print("\nResults:")
    print(df2.to_string(index=False))
    ex2.plot(df2, save_to="fig_sp_ex2.png")

    # ---------------------------------------------------------------- 3
    section("EXERCISE 3 - Compression ratio (Shakespeare excerpt)")
    ex3 = Exercise3Solution(
        train_corpus=shakespeare[:20_000],
        test_corpus=shakespeare[:20_000],
        vocab_size=8000,
    )
    df3 = ex3.run()
    print("\nResults:")
    print(df3.to_string(index=False))
    ex3.plot(df3, save_to="fig_sp_ex3.png")

    # ---------------------------------------------------------------- Edge cases
    section("EDGE CASES AND FAILURE SCENARIOS")
    # Train a small model for edge-case tests.
    edge_model = SentencePieceModel(
        model_prefix="sp_edge_main",
        vocab_size=500,
        model_type="unigram",
        character_coverage=0.9995,
        split_by_whitespace=False,
    )
    edge_model.train(shakespeare[:10_000])
    edge_case_empty_input(edge_model)
    edge_case_wrong_type(edge_model)
    edge_case_extreme_vocab(shakespeare)
    edge_case_unk_rate()

    section("DONE")
    print("Figures saved:")
    print("  fig_sp_ex1.png")
    print("  fig_sp_ex2.png")
    print("  fig_sp_ex3.png")


if __name__ == "__main__":
    main()


# Create a virtual environment (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip3 install sentencepiece matplotlib numpy pandas

# Optional: install HuggingFace tokenizers for Exercise 3 comparison
pip3 install tokenizers

# Run the module
python3 005_spu_learning_module.py

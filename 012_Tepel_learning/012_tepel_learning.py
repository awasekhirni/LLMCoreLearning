"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
012_tepel_learning.py
============================

Token Embedding + Positional Encoding Lookup.

This is the entry point/front door of every transformer: token IDs (integers) go in,
dense d_model vectors with positional information come out, ready for
attention blocks.

Structure
---------
1.  SimpleWordTokenizer       - build vocab from text, encode text -> IDs
2.  EmbeddingConfig           - dataclass holding every hyperparameter
3.  TokenEmbedding            - the lookup table (NumPy implementation)
4.  SinusoidalPE / LearnedPE / NoPE - positional encodings
5.  InputEmbedding            - token + positional (the actual "front door")
6.  TiedLMHead                - weight-tying demo (input weight = output weight)
7.  EmbeddingMetrics          - norm stability, padding invariance, memory
8.  TinyTrainingDemo          - small SGD loop showing embeddings *learn*
9.  Scenario runners          - happy path + 5 edge cases
10. ELI5Walkthrough           - toy script with plain-language narration
11. EmbeddingVisualizer       - matplotlib figures saved to disk
12. main()                    - orchestration

Design principles
-----------------
- DRY: all metrics live on EmbeddingMetrics; all PE variants implement
  the same `forward(seq_len) -> (seq_len, d_model)` contract; scenarios
  share a single validation helper `_validate_token_ids`.
- Dataclasses everywhere: every stateful component is a @dataclass so
  it is trivially instantiable, inspectable, and reproducible.
- Reproducibility: all randomness flows from `EmbeddingConfig.seed`.
- No hidden dependencies: PyTorch is optional and only used in a
  clearly-labelled reference demo.

Dependencies
------------
    pip install numpy pandas matplotlib
    # Optional, for the PyTorch cross-check only:
    pip install torch

Run
---
    python3 embedding_learning_module.py

Figures saved to the current directory:
    fig_1_input_token_ids.png
    fig_2_embedding_lookup.png
    fig_3_positional_encodings.png
    fig_4_combined_embeddings.png
    fig_5_norm_stability.png
    fig_6_weight_tying.png
"""

from __future__ import annotations

import math
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional dependency: PyTorch is used only for a reference cross-check.
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:                            # pragma: no cover
    TORCH_AVAILABLE = False


# ----------------------------------------------------------------------
# Global reproducibility (per-library seeds; NumPy 2.x prefers rng)
# ----------------------------------------------------------------------
SEED: int = 42
np.random.seed(SEED)


# ======================================================================
# CONSTANTS
# ======================================================================

GUTENBERG_URL: str = "https://www.gutenberg.org/files/100/100-0.txt"

# Public-domain Shakespeare excerpt (~1200 chars). Used as a fallback if
# the Gutenberg download fails, so this module always runs offline.
EMBEDDED_SHAKESPEARE: str = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles.
All the world's a stage, and all the men and women merely players;
They have their exits and their entrances.
Tomorrow, and tomorrow, and tomorrow, creeps in this petty pace from day to day.
Shall I compare thee to a summer's day? Thou art more lovely and more temperate.
Let me not to the marriage of true minds admit impediments.
""".strip()

# Small corpus used for the "training loss contribution" demo.
# Shorter than the Shakespeare corpus so the loop runs in seconds.
TOY_CORPUS: str = """
the cat sat on the mat
the dog sat on the log
a cat and a dog
the mat is on the floor
the dog is on the mat
""".strip()


# ======================================================================
# UTILITIES
# ======================================================================

def load_corpus(
    max_chars: Optional[int] = 20_000,
    prefer_download: bool = True,
    url: str = GUTENBERG_URL,
    timeout_s: float = 10.0,
) -> str:
    """
    Load a Shakespeare corpus, preferring the Gutenberg download.

    Parameters
    ----------
    max_chars : int or None
        Truncate the corpus to this many characters (keeps runtimes
        classroom-friendly). ``None`` disables truncation.
    prefer_download : bool
        Try the network first; fall back to the embedded excerpt on any
        error. An offline run therefore always succeeds.
    url : str
        Gutenberg URL for *The Complete Works of William Shakespeare*.
    timeout_s : float
        Network timeout—never hang a live demo.

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
        except Exception as exc:               # noqa: BLE001 - fallback is intended
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


def embedding_memory_mb(
    vocab_size: int,
    d_model: int,
    bytes_per_param: int = 2,
) -> float:
    """
    Memory footprint (in MB) of an embedding matrix.

    Parameters
    ----------
    vocab_size, d_model : int
        Matrix shape (vocab_size, d_model).
    bytes_per_param : int
        2 for fp16/bf16, 4 for fp32.

    Returns
    -------
    float
        Size in mebibytes (MiB).
    """
    return (vocab_size * d_model * bytes_per_param) / (1024 ** 2)


# ======================================================================
# SIMPLE WORD-LEVEL TOKENIZER
# ======================================================================
# We need integer token IDs to feed the embedding layer. Rather than
# require tiktoken / HuggingFace, we build a tiny whitespace+punct
# word-level tokenizer from the corpus. This keeps the module
# self-contained and makes the ID -> vector mapping visible.
# ======================================================================

@dataclass
class SimpleWordTokenizer:
    """
    A minimal word-level tokenizer.

    Attributes
    ----------
    vocab : dict[str, int]
        Token string -> integer ID. Special tokens occupy IDs 0..3.
    specials : list[str]
        Reserved tokens, always first in the vocabulary.
    """

    specials: List[str] = field(
        default_factory=lambda: ["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    vocab: Dict[str, int] = field(default_factory=dict)

    # ------------------ construction ------------------
    def build(self, corpus: str, max_vocab: Optional[int] = None) -> "SimpleWordTokenizer":
        """
        Build the vocabulary from a corpus.

        Parameters
        ----------
        corpus : str
        max_vocab : int or None
            Cap on vocabulary size. ``None`` = unbounded.

        Returns
        -------
        SimpleWordTokenizer
            ``self``, for chaining.
        """
        self.vocab = {tok: i for i, tok in enumerate(self.specials)}
        counts = Counter(self._split(corpus))
        for word, _freq in counts.most_common():
            if max_vocab is not None and len(self.vocab) >= max_vocab:
                break
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        return self

    @staticmethod
    def _split(text: str) -> List[str]:
        """Lowercase + split into word / single-punctuation tokens."""
        return re.findall(r"\w+|[^\w\s]", text.lower())

    # ------------------ encode / decode ------------------
    def encode(
        self,
        text: str,
        add_special: bool = True,
        max_len: Optional[int] = None,
    ) -> List[int]:
        """
        Encode text -> token IDs.

        Unknown words map to ``<unk>``. If ``add_special`` is True,
        ``<bos>`` and ``<eos>`` are prepended/appended respectively
        (standard for language-modeling sequences).
        """
        ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in self._split(text)]
        if max_len is not None:
            ids = ids[:max_len]
        if add_special:
            ids = [self.vocab["<bos>"]] + ids + [self.vocab["<eos>"]]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Inverse of encode; special tokens are dropped, case is not recovered."""
        inv = {i: t for t, i in self.vocab.items()}
        kept = [
            inv.get(i, "<unk>") for i in ids
            if inv.get(i) not in self.specials
        ]
        return " ".join(kept)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]


# ======================================================================
# CONFIG
# ======================================================================

@dataclass
class EmbeddingConfig:
    """
    All hyperparameters for the embedding front-end.

    Attributes
    ----------
    vocab_size : int
        Number of tokens (rows in the embedding table).
    d_model : int
        Embedding dimension. Must be even for sinusoidal PE.
    padding_idx : int
        Row index used for padding. That row is zeroed and receives no
        gradient. Use ``-1`` for "no padding".
    max_position_embeddings : int
        Cap on sequence length for *learned* / *sinusoidal* encodings.
        Ignored when ``positional_encoding_type == "none"``.
    positional_encoding_type : str
        "sinusoidal" (fixed), "learned" (trainable), or "none"
        (expect RoPE/ALiBi downstream).
    init_std : float
        Standard deviation of the N(0, init_std) initialisation used by
        the original Transformer (Vaswani et al. 2017).
    scale_by_sqrt_d : bool
        Multiply token embeddings by sqrt(d_model) before adding
        positional information. This is what modern LLMs do; set False
        to replicate the original paper's "scaled PE" variant.
    seed : int
        RNG seed for reproducibility.
    """

    vocab_size: int = 1000
    d_model: int = 64
    padding_idx: int = 0
    max_position_embeddings: int = 512
    positional_encoding_type: str = "sinusoidal"
    init_std: float = 0.02
    scale_by_sqrt_d: bool = True
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.d_model < 2:
            raise ValueError(f"d_model must be >= 2, got {self.d_model}")
        if self.positional_encoding_type not in {"sinusoidal", "learned", "none"}:
            raise ValueError(
                f"positional_encoding_type must be 'sinusoidal', 'learned', or 'none'; "
                f"got {self.positional_encoding_type!r}"
            )
        if (
            self.positional_encoding_type == "sinusoidal"
            and self.d_model % 2 != 0
        ):
            raise ValueError(
                f"d_model must be even for sinusoidal PE, got {self.d_model}"
            )
        if self.positional_encoding_type != "none" and self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be > 0")
        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")


# ======================================================================
# VALIDATION HELPER (DRY - used by every forward pass)
# ======================================================================

def validate_token_ids(token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    Coerce ``token_ids`` to an int64 array and validate it.

    Rules enforced:
      * must be numeric (int or float with integral values),
      * every id in ``[0, vocab_size)``,
      * returns a C-contiguous ``int64`` array so callers can index
        directly without further checks.

    Raises
    ------
    TypeError
        If the input is not numeric or contains non-integral floats.
    ValueError
        If any id is out of range.
    """
    arr = np.asarray(token_ids)
    if arr.size == 0:
        return arr.astype(np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.floating) and np.all(arr == arr.astype(np.int64)):
            arr = arr.astype(np.int64)
        else:
            raise TypeError(
                f"token_ids must be integer-valued; got dtype={arr.dtype}"
            )
    if arr.min() < 0 or arr.max() >= vocab_size:
        raise ValueError(
            f"token_ids out of range: min={arr.min()}, max={arr.max()}, "
            f"vocab_size={vocab_size}"
        )
    return arr.astype(np.int64)


# ======================================================================
# TOKEN EMBEDDING
# ======================================================================

@dataclass
class TokenEmbedding:
    """
    Dense lookup table: ``token_id -> vector of size d_model``.

    This mirrors ``torch.nn.Embedding`` exactly (lookup, padding row
    zeroed, optional sqrt(d_model) scaling) but is implemented in pure
    NumPy so the mechanics are visible and the module has no hard
    dependency on PyTorch.
    """

    config: EmbeddingConfig
    weight: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.config.seed)
        self.weight = rng.normal(
            loc=0.0,
            scale=self.config.init_std,
            size=(self.config.vocab_size, self.config.d_model),
        ).astype(np.float64)
        # Padding row is zeroed (as in PyTorch nn.Embedding).
        if 0 <= self.config.padding_idx < self.config.vocab_size:
            self.weight[self.config.padding_idx] = 0.0

    @property
    def shape(self) -> Tuple[int, int]:
        return self.weight.shape

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Look up embeddings for a batch of IDs.

        Parameters
        ----------
        token_ids : array-like of int
            Shape ``(B, L)`` or ``(L,)``.

        Returns
        -------
        np.ndarray
            Shape ``(B, L, d_model)`` or ``(L, d_model)`` matching the
            input, scaled by sqrt(d_model) if configured.
        """
        ids = validate_token_ids(token_ids, self.config.vocab_size)
        out = self.weight[ids]                  # fancy indexing = the "lookup"
        if self.config.scale_by_sqrt_d:
            out = out * math.sqrt(self.config.d_model)
        return out

    # Convenience so `model(ids)` works
    __call__ = forward


# ======================================================================
# POSITIONAL ENCODINGS
# ======================================================================
# All PE variants implement the same tiny contract:
#     forward(seq_len) -> np.ndarray of shape (seq_len, d_model)
# which InputEmbedding simply adds to the token embeddings. This
# uniform interface is the DRY backbone that lets us swap PE types
# without changing the caller.
# ======================================================================

@dataclass
class SinusoidalPE:
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).

        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    Advantages: zero parameters, deterministic, extrapolates to any
    position up to ``max_position_embeddings``.

    The table is precomputed once at construction; ``forward`` returns
    a *slice* (view) so forward passes are O(seq_len) not O(seq_len·d).
    """

    config: EmbeddingConfig
    table: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        d = self.config.d_model
        max_len = self.config.max_position_embeddings
        positions = np.arange(max_len, dtype=np.float64)[:, None]         # (M, 1)
        # div_term[i] = 10000 ^ (-2i / d)
        div_term = np.exp(np.arange(0, d, 2, dtype=np.float64) * (-math.log(10000.0) / d))
        pe = np.zeros((max_len, d), dtype=np.float64)
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        self.table = pe

    def forward(self, seq_len: int) -> np.ndarray:
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )
        return self.table[:seq_len]

    __call__ = forward


@dataclass
class LearnedPE:
    """
    Trainable absolute positional embedding table.

    Same lookup mechanics as token embedding, just indexed by *position*
    instead of token ID. Advantages: task-adaptive. Disadvantage:
    cannot extrapolate beyond ``max_position_embeddings`` at all.
    """

    config: EmbeddingConfig
    weight: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.config.seed + 1)
        self.weight = rng.normal(
            0.0, self.config.init_std,
            size=(self.config.max_position_embeddings, self.config.d_model),
        ).astype(np.float64)

    def forward(self, seq_len: int) -> np.ndarray:
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )
        return self.weight[:seq_len]

    __call__ = forward


@dataclass
class NoPE:
    """
    Identity placeholder: *no positional encoding is added to embeddings*.

    Use this when position information is injected later—e.g. RoPE in
    attention (Llama, Claude) or ALiBi bias on attention logits. The
    embedding layer simply passes token embeddings through untouched.
    """

    config: EmbeddingConfig

    def forward(self, seq_len: int) -> np.ndarray:      # noqa: ARG002
        # Return a zeros array so InputEmbedding can stay branch-free.
        return np.zeros((seq_len, self.config.d_model), dtype=np.float64)

    __call__ = forward


# ======================================================================
# THE "FRONT DOOR": TOKEN EMBEDDING + POSITIONAL ENCODING
# ======================================================================

@dataclass
class InputEmbedding:
    """
    Combine token embeddings and positional encodings.

    Final output shape is ``(B, L, d_model)``—exactly what transformer
    blocks expect. This class is the thing you plug into any model as
    layer zero.

    Also exposes an attention mask derived from ``padding_idx`` so
    downstream attention can ignore padding tokens.
    """

    config: EmbeddingConfig
    token_embedding: TokenEmbedding = field(init=False)
    positional_encoding: Union[SinusoidalPE, LearnedPE, NoPE] = field(init=False)

    def __post_init__(self) -> None:
        self.token_embedding = TokenEmbedding(self.config)
        pe_type = self.config.positional_encoding_type
        if pe_type == "sinusoidal":
            self.positional_encoding = SinusoidalPE(self.config)
        elif pe_type == "learned":
            self.positional_encoding = LearnedPE(self.config)
        elif pe_type == "none":
            self.positional_encoding = NoPE(self.config)
        else:                                   # pragma: no cover - config validated
            raise ValueError(f"Unknown PE type: {pe_type}")

    def forward(
        self,
        token_ids: np.ndarray,
        return_mask: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        token_ids : array-like of int, shape (B, L) or (L,)
        return_mask : bool
            If True, also return an attention mask with 1 = keep, 0 = pad.

        Returns
        -------
        np.ndarray of shape (B, L, d_model) [or (L, d_model)]
            OR (embeddings, attention_mask) if ``return_mask=True``.
        """
        ids = validate_token_ids(token_ids, self.config.vocab_size)
        tok = self.token_embedding(ids)                     # (..., L, D)
        seq_len = ids.shape[-1]
        pos = self.positional_encoding(seq_len)             # (L, D) - broadcast
        out = tok + pos
        if return_mask:
            mask = (ids != self.config.padding_idx).astype(np.float32)
            return out, mask
        return out

    __call__ = forward


# ======================================================================
# WEIGHT-TYING LM HEAD (small but important reference demo)
# ======================================================================

@dataclass
class TiedLMHead:
    """
    A language-model head that *shares* its weight matrix with the input
    embedding table. Saves ``vocab_size × d_model`` parameters (10–20%
    of a typical LLM) and typically matches or slightly improves quality.

    This is what Claude, GPT-2, and most modern LLMs do. The technical
    card explicitly calls out weight tying as a "pro".

    Usage
    -----
        tied = TiedLMHead(embedding)
        logits = tied(hidden_states)                # (..., V)
    """

    embedding: TokenEmbedding

    def __call__(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        hidden_states : np.ndarray, shape (..., d_model)

        Returns
        -------
        np.ndarray, shape (..., vocab_size)
            Un-normalised logits (pre-softmax).
        """
        # h @ W^T  - equivalent to torch.nn.functional.linear(h, W)
        return hidden_states @ self.embedding.weight.T

    @property
    def weight(self) -> np.ndarray:
        """Expose the shared matrix so callers cannot accidentally copy it."""
        return self.embedding.weight


# ======================================================================
# METRICS
# ======================================================================

@dataclass
class EmbeddingMetrics:
    """
    All benchmark metrics referenced by the technical card.

    - ``norm_stability``        : embedding norm stats (mean, std, min, max)
    - ``padding_invariance``    : does the padding row stay zero?
    - ``oov_robustness``        : does ``<unk>`` flow through correctly?
    - ``memory_report``         : vocab_size × d_model memory in MB
    """

    @staticmethod
    def norm_stability(embeddings: np.ndarray) -> Dict[str, float]:
        """
        Per-token L2-norm statistics.

        Healthy embeddings at init have norms concentrated around
        ``sqrt(d_model)·init_std`` (for untied, unscaled) or around
        ``sqrt(d_model)`` after scaling. Divergence between positions
        warns of instability.
        """
        norms = np.linalg.norm(embeddings, axis=-1).ravel()
        return {
            "mean": float(norms.mean()),
            "std":  float(norms.std()),
            "min":  float(norms.min()),
            "max":  float(norms.max()),
        }

    @staticmethod
    def padding_invariance(
        model: InputEmbedding, token_ids: np.ndarray
    ) -> bool:
        """
        Verify that the padding_idx row is exactly zero.

        Invariant: after forward, every position whose ID == padding_idx
        must be the zero vector *if* the PE is "none". For sinusoidal /
        learned PE, padding rows still receive positional signal, which
        is expected—the *token* part is zero. We check the raw table.
        """
        return bool(np.allclose(
            model.token_embedding.weight[model.config.padding_idx], 0.0
        ))

    @staticmethod
    def oov_robustness(
        tokenizer: SimpleWordTokenizer,
        model: InputEmbedding,
        text: str,
    ) -> Dict[str, Union[int, bool, float]]:
        """
        Check that an unknown token is mapped to ``<unk>`` and produces
        a finite embedding (no NaN/Inf leakage).
        """
        ids = tokenizer.encode(text, add_special=False)
        unk_present = tokenizer.unk_id in ids
        emb = model(np.asarray([ids]))
        finite = bool(np.isfinite(emb).all())
        return {
            "num_tokens": len(ids),
            "unk_present": unk_present,
            "embeddings_finite": finite,
        }

    @staticmethod
    def memory_report(config: EmbeddingConfig) -> pd.DataFrame:
        """
        Compare memory footprint with and without weight tying.
        """
        mb = embedding_memory_mb(config.vocab_size, config.d_model)
        return pd.DataFrame([
            dict(variant="untied (input + output)", params=2 * config.vocab_size * config.d_model, memory_mb=2 * mb),
            dict(variant="tied (shared matrix)",   params=config.vocab_size * config.d_model,     memory_mb=mb),
        ])


# ======================================================================
# TINY TRAINING DEMO (shows "embedding training loss contribution")
# ======================================================================

@dataclass
class TinyTrainingDemo:
    """
    A 4-step manual gradient-descent demo proving that *the embedding
    weight matrix is trainable* and that weight-tying makes a real
    difference to the loss.

    Setup (deliberately tiny):
      - vocab_size ≈ 15, d_model = 8, context_len = 4
      - Each training pair: (input_ids[0:4], target_id = input_ids[4])
      - Hidden state = mean of the 4 token embeddings (a mini "pool")
      - Logits = hidden @ W^T where W = embedding.weight  (TIED)
      - Loss = cross-entropy
      - Manual gradient on W, sum of two contributions:
          (a) output side:  dLogits @ hidden^T
          (b) input side:   mean of hidden gradients scattered to source IDs

    This is not a real LM—it is a pedagogical illustration that the loss
    gradient *flows into* the embedding table and improves it.
    """

    corpus: str
    config: EmbeddingConfig
    context_len: int = 4
    steps: int = 200
    lr: float = 0.5

    def _build_pairs(self) -> List[Tuple[np.ndarray, int]]:
        tok = SimpleWordTokenizer().build(self.corpus)
        ids = tok.encode(self.corpus, add_special=False)
        pairs = []
        for i in range(len(ids) - self.context_len):
            ctx = np.asarray(ids[i:i + self.context_len], dtype=np.int64)
            target = ids[i + self.context_len]
            pairs.append((ctx, target))
        # Cache the tokenizer on self so `run` can return it for inspection
        self._tokenizer = tok
        return pairs

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over the last axis."""
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def run(self, verbose_every: Optional[int] = 50) -> pd.DataFrame:
        """
        Train the embeddings with plain SGD (tied LM head).

        Returns
        -------
        pd.DataFrame with columns ``step`` and ``loss``.
        """
        pairs = self._build_pairs()
        emb = TokenEmbedding(self.config)
        tied = TiedLMHead(emb)
        pad = self.config.padding_idx
        V, D = emb.shape
        history: List[dict] = []

        for step in range(1, self.steps + 1):
            # Sample one pair per step to keep gradients as small as possible.
            idx = (step - 1) % len(pairs)
            ctx, target = pairs[idx]

            # --- forward ---------------------------------------------------
            tok_emb = emb(ctx)                         # (L, D)
            if 0 <= pad < len(ctx):
                keep = (ctx != pad).astype(np.float64)[:, None]
                hidden = (tok_emb * keep).sum(axis=0) / max(keep.sum(), 1.0)
            else:
                hidden = tok_emb.mean(axis=0)          # (D,)
            logits = tied(hidden[None, :])[0]          # (V,)
            probs = self._softmax(logits)
            loss = -math.log(max(probs[target], 1e-12))

            # --- backward (tied weights => two contributions to dW) -------
            dlogits = probs.copy()
            dlogits[target] -= 1.0                     # dL/dlogits  (V,)
            # (a) output side: dW += dlogits^T @ hidden
            dW_out = np.outer(dlogits, hidden)         # (V, D)
            # (b) input side: hidden = mean(tok_emb[ctx]) => d_hidden is
            #     dlogits @ W; d_token_emb = d_hidden / L; scatter-add.
            d_hidden = dlogits @ tied.weight           # (D,)
            dW_in = np.zeros_like(emb.weight)
            share = 1.0 / max(np.sum(ctx != pad), 1)
            for tid in ctx:
                if tid == pad:
                    continue
                dW_in[tid] += d_hidden * share

            # --- SGD update -------------------------------------------------
            grad = dW_out + dW_in
            grad[pad] = 0.0                             # never touch <pad>
            emb.weight -= self.lr * grad

            history.append(dict(step=step, loss=loss))
            if verbose_every and step % verbose_every == 0:
                print(f"  step {step:3d}  loss={loss:.4f}")

        return pd.DataFrame(history)


# ======================================================================
# SCENARIO RUNNERS
# ======================================================================

def run_happy_path(
    tokenizer: SimpleWordTokenizer,
    config: EmbeddingConfig,
    sentences: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scenario 1 — happy path.

    Encode 3 sentences (padded to common length), run the embedding,
    return (embeddings, attention_mask).
    """
    print("\n[Scenario 1] Happy path")
    max_len = max(len(tokenizer.encode(s, add_special=False)) for s in sentences)
    batch = np.full((len(sentences), max_len), tokenizer.pad_id, dtype=np.int64)
    for i, s in enumerate(sentences):
        ids = tokenizer.encode(s, add_special=False)
        batch[i, :len(ids)] = ids
    print(f"  batch shape = {batch.shape}, max_len = {max_len}")
    print(f"  batch[0]    = {batch[0].tolist()}")

    model = InputEmbedding(config)
    emb, mask = model(batch, return_mask=True)
    print(f"  embeddings shape = {emb.shape}")
    print(f"  attention mask shape = {mask.shape}, "
          f"kept = {int(mask.sum())}/{mask.size}")

    stats = EmbeddingMetrics.norm_stability(emb)
    print(f"  norm stats: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
          f"min={stats['min']:.3f}, max={stats['max']:.3f}")
    print(f"  padding invariant: {EmbeddingMetrics.padding_invariance(model, batch)}")
    return emb, mask


def run_edge_cases(tokenizer: SimpleWordTokenizer, config: EmbeddingConfig) -> None:
    """Scenario 2–6 — edge cases and failure modes."""
    model = InputEmbedding(config)

    # -------- 2. Empty input --------------------------------------------
    print("\n[Scenario 2] Empty input")
    try:
        out = model(np.empty((0,), dtype=np.int64))
        print(f"  → shape {out.shape} (0 tokens, d_model preserved)")
    except Exception as exc:                # noqa: BLE001
        print(f"  → raised {type(exc).__name__}: {exc}")

    # -------- 3. All-padding batch --------------------------------------
    print("\n[Scenario 3] All-padding batch (padding_idx invariant)")
    padded = np.full((2, 5), tokenizer.pad_id, dtype=np.int64)
    emb, mask = model(padded, return_mask=True)
    # With sinusoidal/learned PE, padding rows are NOT zero (they receive
    # positional signal). The *token* contribution is zero, which is what
    # matters. Assert the token-table row is zero.
    print(f"  token embedding row for padding is all-zero: "
          f"{bool(np.allclose(model.token_embedding.weight[tokenizer.pad_id], 0.0))}")
    print(f"  attention mask sum = {int(mask.sum())} (expected 0)")

    # -------- 4. Out-of-vocabulary token --------------------------------
    print("\n[Scenario 4] Out-of-vocabulary word")
    oov = EmbeddingMetrics.oov_robustness(
        tokenizer, model, "the zorblax frobnicated the widget"
    )
    print(f"  unk_present={oov['unk_present']}, "
          f"embeddings_finite={oov['embeddings_finite']}")

    # -------- 5. Sequence longer than max_position_embeddings -----------
    print("\n[Scenario 5] Sequence longer than max_position_embeddings")
    too_long = np.zeros((1, config.max_position_embeddings + 1), dtype=np.int64)
    try:
        model(too_long)
        print("  → no error (unexpected)")
    except ValueError as exc:
        print(f"  → ValueError as expected: {exc}")

    # -------- 6. Wrong input type ---------------------------------------
    print("\n[Scenario 6] Wrong input type (float without integral values)")
    try:
        model(np.array([[1.0, 2.5, 3.0]]))
        print("  → no error (unexpected)")
    except TypeError as exc:
        print(f"  → TypeError as expected: {exc}")


# ======================================================================
# ELI5 WALKTHROUGH
# ======================================================================

@dataclass
class ELI5Walkthrough:
    """
    A step-by-step, plain-English walkthrough with toy data.

    Every print statement is written as if explaining to a 5-year-old,
    but the underlying computation is identical to the real model.
    """

    config: EmbeddingConfig
    tokenizer: SimpleWordTokenizer
    sentence: str = "the cat sat on the mat"

    def run(self) -> None:
        print()
        print("=" * 78)
        print(" ELI5 WALKTHROUGH: How a sentence becomes a vector")
        print("=" * 78)

        # --- Step 1 -----------------------------------------------------
        print("\nStep 1: Split the sentence into pieces called tokens.")
        tokens = self.tokenizer._split(self.sentence)
        print(f"   '{self.sentence}'")
        print(f"   → tokens = {tokens}")

        # --- Step 2 -----------------------------------------------------
        print("\nStep 2: Turn each token into a number (its ID).")
        ids = self.tokenizer.encode(self.sentence, add_special=False)
        print(f"   → ids = {ids}")

        # --- Step 3 -----------------------------------------------------
        print("\nStep 3: Look up each ID in a big table. "
              "Each row of the table is a list of numbers (a vector).")
        model = InputEmbedding(self.config)
        tok_emb = model.token_embedding(np.asarray(ids))
        print(f"   → token embeddings shape = {tok_emb.shape}")
        print(f"   → first token's first 5 numbers = "
              f"{np.round(tok_emb[0, :5], 3).tolist()}")

        # --- Step 4 -----------------------------------------------------
        print("\nStep 4: The table row only says WHAT the word is. "
              "It doesn't say WHERE the word is.")
        print("        To fix that, we add a special 'position sticky note' "
              "to every word.")
        pos = model.positional_encoding(len(ids))
        print(f"   → position encodings shape = {pos.shape}")
        print(f"   → position 0's first 5 numbers = "
              f"{np.round(pos[0, :5], 3).tolist()}")

        # --- Step 5 -----------------------------------------------------
        print("\nStep 5: Add them together. Now each position has "
              "MEANING + POSITION baked in.")
        final = model(np.asarray(ids))
        print(f"   → final embeddings shape = {final.shape}")
        print(f"   → first token after adding = "
              f"{np.round(final[0, :5], 3).tolist()}")
        print("\nThat's it! The transformer takes these numbers and does "
              "attention, FFN, etc. But it all started HERE.")


# ======================================================================
# VISUALIZER
# ======================================================================

@dataclass
class EmbeddingVisualizer:
    """
    Produce all six figures for this lesson and save them to disk.

    File names (fixed, so learners can verify):
        fig_1_input_token_ids.png
        fig_2_embedding_lookup.png
        fig_3_positional_encodings.png
        fig_4_combined_embeddings.png
        fig_5_norm_stability.png
        fig_6_weight_tying.png
    """

    config: EmbeddingConfig
    tokenizer: SimpleWordTokenizer
    sentences: Sequence[str] = (
        "the cat sat on the mat",
        "a dog slept on the log",
    )

    def _pad_batch(self) -> np.ndarray:
        max_len = max(len(self.tokenizer.encode(s, add_special=False)) for s in self.sentences)
        batch = np.full((len(self.sentences), max_len), self.tokenizer.pad_id, dtype=np.int64)
        for i, s in enumerate(self.sentences):
            ids = self.tokenizer.encode(s, add_special=False)
            batch[i, :len(ids)] = ids
        return batch

    # ------------------------------------------------------------------
    def plot_input(self, path: Path = Path("fig_1_input_token_ids.png")) -> None:
        """Figure (a): what goes in — token IDs for the example sentence."""
        batch = self._pad_batch()
        ids = batch[0]
        words = self.tokenizer.decode(ids).split()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        bars = ax.bar(range(len(ids)), ids, color="steelblue")
        for bar, wid, w in zip(bars, ids, words + [""] * len(ids)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"id={wid}", ha="center", fontsize=8)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(words + [""] * (len(ids) - len(words)) if len(words) < len(ids) else words,
                          rotation=30, ha="right")
        ax.set_ylabel("Token ID")
        ax.set_title(f"Fig 1 — Input token IDs for: '{self.sentences[0]}'")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Viz] Saved {path.resolve()}")

    # ------------------------------------------------------------------
    def plot_processing(self, path: Path = Path("fig_2_embedding_lookup.png")) -> None:
        """Figure (b): token embedding | positional encoding | their sum."""
        model = InputEmbedding(self.config)
        ids = np.asarray(self.tokenizer.encode(self.sentences[0], add_special=False))
        tok = model.token_embedding(ids)                         # (L, D)
        pos = model.positional_encoding(len(ids))                # (L, D)
        combined = tok + pos

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, mat, title in zip(
            axes,
            [tok, pos, combined],
            ["Token embedding lookup",
             "Positional encoding (sinusoidal)",
             "Token + Position (fed to transformer)"],
        ):
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r")
            ax.set_xlabel("d_model (feature dim)")
            ax.set_ylabel("sequence position")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle("Fig 2 — Processing: token + positional = input embeddings", y=1.02)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Viz] Saved {path.resolve()}")

    # ------------------------------------------------------------------
    def plot_positional(self, path: Path = Path("fig_3_positional_encodings.png")) -> None:
        """Compare sinusoidal vs learned PE patterns side-by-side."""
        sin_cfg = EmbeddingConfig(**{**self.config.__dict__, "positional_encoding_type": "sinusoidal"})
        lrn_cfg = EmbeddingConfig(**{**self.config.__dict__, "positional_encoding_type": "learned"})
        pe_sin = SinusoidalPE(sin_cfg).forward(self.config.max_position_embeddings)
        pe_lrn = LearnedPE(lrn_cfg).forward(self.config.max_position_embeddings)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, mat, title in zip(axes, [pe_sin, pe_lrn],
                                  ["Sinusoidal (fixed, generalises)",
                                   "Learned (trainable, capped)"]):
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_xlabel("d_model")
            ax.set_ylabel("position")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Fig 3 — Positional encoding patterns", y=1.02)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Viz] Saved {path.resolve()}")

    # ------------------------------------------------------------------
    def plot_combined(self, path: Path = Path("fig_4_combined_embeddings.png")) -> None:
        """Figure (c): the final (B, L, d_model) tensor, one heatmap per batch row."""
        model = InputEmbedding(self.config)
        batch = self._pad_batch()
        emb = model(batch)
        fig, axes = plt.subplots(1, len(batch), figsize=(7 * len(batch), 4.5))
        if len(batch) == 1:
            axes = [axes]
        for ax, mat, sentence in zip(axes, emb, self.sentences):
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r")
            ax.set_xlabel("d_model")
            ax.set_ylabel("position")
            ax.set_title(f"'{sentence}'")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Fig 4 — Final input embeddings (B, L, d_model)", y=1.02)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Viz] Saved {path.resolve()}")

    # ------------------------------------------------------------------
    def plot_norm_stability(self, path: Path = Path("fig_5_norm_stability.png")) -> None:
        """Per-position norm stability + memory comparison."""
        model = InputEmbedding(self.config)
        batch = self._pad_batch()
        emb = model(batch)
        norms = np.linalg.norm(emb, axis=-1)                      # (B, L)

        df_mem = EmbeddingMetrics.memory_report(self.config)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        for b, sentence in enumerate(self.sentences):
            ax1.plot(norms[b], marker="o", label=f"'{sentence}'")
        ax1.set_xlabel("position")
        ax1.set_ylabel("L2 norm of embedding")
        ax1.set_title("Fig 5a — Embedding norm by position (should be stable)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.bar(df_mem["variant"], df_mem["memory_mb"], color=["indianred", "seagreen"])
        for i, v in enumerate(df_mem["memory_mb"]):
            ax2.text(i, v + 0.02, f"{v:.3f} MB", ha="center", fontsize=9)
        ax2.set_ylabel("Memory (MiB)")
        ax2.set_title("Fig 5b — Weight tying halves embedding memory")
        ax2.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Viz] Saved {path.resolve()}")

    # ------------------------------------------------------------------
    def plot_weight_tying(self, path: Path = Path("fig_6_weight_tying.png")) -> None:
        """Show that tying makes input + output share the same matrix."""
        V, D = self.config.vocab_size, self.config.d_model
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, title, label_in, label_out in [
            (axes[0], "Untied (default): two matrices", "input emb\n(V×D)", "LM head\n(V×D)"),
            (axes[1], "Tied: one shared matrix", "input emb\n(V×D)", "LM head (shared)"),
        ]:
            ax.add_patch(plt.Rectangle((0.1, 0.3), 0.3, 0.4, color="steelblue"))
            ax.text(0.25, 0.5, label_in, ha="center", va="center", color="white", fontsize=10)
            if "Untied" in title:
                ax.add_patch(plt.Rectangle((0.6, 0.3), 0.3, 0.4, color="indianred"))
                ax.text(0.75, 0.5, label_out, ha="center", va="center", color="white", fontsize=10)
            else:
                ax.add_patch(plt.Rectangle((0.6, 0.3), 0.3, 0.4, color="steelblue"))
                ax.text(0.75, 0.5, label_out, ha="center", va="center", color="white", fontsize=8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")
            ax.set_title(title)

        total_untied = 2 * V * D
        total_tied = V * D
        fig.suptitle(
            f"Fig 6 — Weight tying: params {total_untied:,} → {total_tied:,} "
            f"({total_untied/total_tied:.1f}× smaller)",
            y=1.04,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Viz] Saved {path.resolve()}")

    # ------------------------------------------------------------------
    def run_all(self) -> None:
        self.plot_input()
        self.plot_processing()
        self.plot_positional()
        self.plot_combined()
        self.plot_norm_stability()
        self.plot_weight_tying()


# ======================================================================
# OPTIONAL PYTORCH CROSS-CHECK (skip if torch not installed)
# ======================================================================

def torch_cross_check(config: EmbeddingConfig, token_ids: np.ndarray) -> None:
    """Confirm the NumPy implementation matches torch.nn.Embedding numerically."""
    if not TORCH_AVAILABLE:
        print("[PyTorch] not installed — skipping cross-check.")
        return
    print("\n[PyTorch cross-check]")
    emb_np = TokenEmbedding(config)
    # Copy the NumPy weight into an nn.Embedding so lookups are identical.
    emb_t = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.padding_idx)
    with torch.no_grad():
        emb_t.weight.copy_(torch.from_numpy(emb_np.weight))
    ids_t = torch.from_numpy(token_ids)
    out_t = emb_t(ids_t).numpy() * (math.sqrt(config.d_model) if config.scale_by_sqrt_d else 1.0)
    out_np = emb_np(token_ids)
    print(f"  numpy shape={out_np.shape}, torch shape={out_t.shape}, "
          f"max|diff|={np.abs(out_np - out_t).max():.3e}")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 40)

    # 1. Corpus + tokenizer -------------------------------------------------
    section("SETUP — corpus + tokenizer")
    corpus = load_corpus(max_chars=20_000, prefer_download=True)
    tokenizer = SimpleWordTokenizer().build(corpus, max_vocab=200)
    print(f"corpus chars: {len(corpus):,}")
    print(f"tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"special tokens: {tokenizer.specials}")

    # 2. Config ------------------------------------------------------------
    config = EmbeddingConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        padding_idx=tokenizer.pad_id,
        max_position_embeddings=64,
        positional_encoding_type="sinusoidal",
        init_std=0.02,
        scale_by_sqrt_d=True,
        seed=SEED,
    )
    print(f"config: {config}")

    # 3. Scenario runners --------------------------------------------------
    section("SCENARIOS")
    run_happy_path(
        tokenizer, config,
        sentences=["the cat sat on the mat",
                   "a dog slept on the log",
                   "shakespeare wrote many sonnets"],
    )
    run_edge_cases(tokenizer, config)

    # 4. ELI5 walkthrough --------------------------------------------------
    section("ELI5 WALKTHROUGH")
    ELI5Walkthrough(config=config, tokenizer=tokenizer).run()

    # 5. Metrics: memory, OOV, norm stability -----------------------------
    section("METRICS")
    print("\nMemory report:")
    print(EmbeddingMetrics.memory_report(config).to_string(index=False))
    print("\nOOV robustness (unseen words → <unk>):")
    model = InputEmbedding(config)
    print(EmbeddingMetrics.oov_robustness(
        tokenizer, model, "the frobnicator zorblaxed a widget"
    ))

    # 6. Tiny training demo -----------------------------------------------
    section("TRAINING LOSS CONTRIBUTION DEMO")
    toy_tok = SimpleWordTokenizer().build(TOY_CORPUS)
    toy_cfg = EmbeddingConfig(
        vocab_size=toy_tok.vocab_size,
        d_model=8,
        padding_idx=toy_tok.pad_id,
        max_position_embeddings=16,
        positional_encoding_type="none",
        init_std=0.1,
        scale_by_sqrt_d=False,
        seed=SEED + 2,
    )
    demo = TinyTrainingDemo(corpus=TOY_CORPUS, config=toy_cfg, steps=200, lr=0.5)
    history = demo.run(verbose_every=50)
    print(f"  loss: start={history['loss'].iloc[0]:.4f} "
          f"→ end={history['loss'].iloc[-1]:.4f}")

    # 7. PyTorch cross-check ----------------------------------------------
    section("PYTORCH CROSS-CHECK (optional)")
    sample_ids = np.asarray(tokenizer.encode("the cat sat on the mat", add_special=False))
    torch_cross_check(config, sample_ids)

    # 8. Visualizations ---------------------------------------------------
    section("VISUALIZATIONS")
    viz = EmbeddingVisualizer(config=config, tokenizer=tokenizer)
    viz.run_all()

    section("DONE")
    print("Figures written to current directory: fig_1..fig_6_*.png")


if __name__ == "__main__":
    main()


# python3 -m venv .venv && source .venv/bin/activate
# pip install numpy pandas matplotlib
# # Optional:
# pip install torch

# python3 embedding_learning_module.py

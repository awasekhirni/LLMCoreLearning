"""
Author: Awase Khirni Syed, 03/October/2025
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

This module implements a complete, working, decoder-only Transformer LLM from
scratch in PyTorch, covering every major concept discussed in the book:


   Tokens & Embeddings (BPE tokenizer, token embeddings,
             contextualized embeddings, text embeddings)
   Inside LLMs (Transformer blocks, self-attention, KV cache,
             RoPE, RMSNorm, SwiGLU, GQA, parallel token processing,
             sampling/decoding strategies)
   Text Classification (task head)
   Prompt Engineering (sampling controls: temperature, top-k, top-p)
   Advanced Text Generation (memory, chains, agents scaffold)
   Text Embedding Models (contrastive learning)
   Fine-Tuning Representation Models (classification head)
   Fine-Tuning Generation Models (SFT / preference tuning scaffold)



  * Full type hints and NumPy-style docstrings on every class and method
  * Modular, composable class hierarchy
  * Complete KV-cache implementation (Ch. 3) — enables ~4-5x faster
    autoregressive decoding
  * Grouped-Query Attention (GQA) — Ch. 3
  * Rotary Position Embeddings (RoPE) — Ch. 3
  * NTK-aware RoPE scaling for long-context extrapolation
  * SwiGLU feedforward activation — Ch. 3
  * RMSNorm (pre-normalization) — Ch. 3
  * Weight tying between token embedding and LM head (reduces parameters)
  * Multiple sampling strategies (greedy, temperature, top-k, top-p,
    min-p, repetition penalty) — Ch. 6
  * Gradient accumulation, cosine LR schedule with warmup, gradient
    clipping, mixed-precision training (AMP, bf16 on Ampere+)
  * Gradient checkpointing for memory-efficient large-model training
  * Checkpoint save/load (model + optimizer + config + tokenizer)
  * HuggingFace-style `save_pretrained` / `from_pretrained` directory API
  * Batched generation with left-padding
  * Streaming generation (token-by-token yield)
  * Real (non-stub) ClassificationHead and EmbeddingModel that run the
    full backbone
  * Unified attention-mask builder (causal + padding + cache-aware)
  * KVCache container class with reset / detach / to-device
  * Numerical stability safeguards (logit clamping, epsilon handling,
    uniform fallback for degenerate sampling distributions)
  * Reproducibility helpers (seed control with optional cudnn.deterministic)
  * `logging` module integration (no bare `print` calls)
  * Explicit `__all__` public API surface

QUICK START
-----------
    python betaori_llmcore.py

This will train a tiny model on a small corpus so you can see end-to-end
next-token prediction working. Swap in your own corpus / config for real use.

LICENSE: MIT (educational use)
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


# ==============================================================================
# PUBLIC API & LOGGING
# ==============================================================================

__all__ = [
    # Config
    "LLMConfig",
    # Tokenizer
    "BPETokenizer",
    # Normalization
    "RMSNorm", "LayerNorm", "make_norm",
    # RoPE
    "precompute_rope_frequencies", "precompute_rope_frequencies_ntk",
    "apply_rope", "rotate_half",
    # Mask utilities
    "build_attention_mask",
    # KV cache
    "KVCache",
    # Attention / FFN / Block
    "MultiHeadAttention", "FeedForward", "TransformerBlock",
    # Model
    "LLM",
    # Generation
    "TextGenerator",
    # Data / Training
    "TextDataset", "LLMTrainer",
    # Extension heads
    "ClassificationHead", "EmbeddingModel",
    # Utilities
    "set_seed",
    # Entry point
    "demo",
]

logger = logging.getLogger("hands_on_llm")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

@dataclass
class LLMConfig:
    """
    Hyperparameters and architectural choices for the LLM.

    This dataclass centralises every knob that affects the model's shape,
    numerical behaviour, and memory footprint. It follows the design pattern
    used by HuggingFace's ``PretrainedConfig``, so you can serialize/deserialize
    it directly with ``asdict()`` / ``LLMConfig(**d)``.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the tokenizer's vocabulary. Must match the
        tokenizer used at training/inference time.
    max_seq_len : int
        Maximum context length ("context size"). Positions beyond
        this are truncated. RoPE can extrapolate somewhat (especially with
        NTK scaling), but learned positional embeddings strictly require
        ``pos < max_seq_len``.
    hidden_size : int
        Model dimension ``d_model``. Every Transformer block preserves
        this size.
    num_layers : int
        Number of stacked Transformer blocks (Ch. 3, "Transformer blocks").
    num_heads : int
        Number of query heads. ``hidden_size`` must be divisible by this.
    num_kv_heads : Optional[int]
        Number of key/value heads for Grouped-Query Attention (Ch. 3).
        If ``None``, defaults to ``num_heads`` (i.e., standard Multi-Head
        Attention). If less than ``num_heads``, enables GQA, which shrinks
        the KV cache and speeds up generation at a small quality cost.
    intermediate_size : Optional[int]
        Hidden size of the feedforward network. If ``None``, defaults to
        ``4 * hidden_size`` for GELU/ReLU, or ``8/3 * hidden_size``
        (rounded up to a multiple of 64) for SwiGLU -- matching the ratio
        used in Llama.
    dropout : float
        Dropout probability applied inside residual branches.
    attention_dropout : float
        Dropout on attention weights (usually 0.0 during inference).
    use_rope : bool
        If True, use rotary positional embeddings (RoPE, Ch. 3). If False,
        use a learned absolute positional embedding table.
    rope_theta : float
        RoPE base frequency. 10_000 is the original; Llama 3 uses 500_000.
    rope_scaling_factor : float
        NTK-by-parts scaling factor for length extrapolation. 1.0 = no
        scaling. Use a value > 1 to extend context beyond ``max_seq_len``
        without retraining.
    use_swiglu : bool
        If True, use the SwiGLU feedforward . If False, use GELU.
    use_rms_norm : bool
        If True, use RMSNorm . If False, use LayerNorm.
    tie_embeddings : bool
        If True, share weights between the token embedding and the LM head.
        This is standard for GPT-2 / Llama and reduces parameters.
    initializer_range : float
        Standard deviation for the normal initialization of linear weights.
    gradient_checkpointing : bool
        If True, enable gradient checkpointing in Transformer blocks.
        Trades ~30% compute for ~3x memory savings. Use for large models
        that would otherwise OOM.

    Notes
    -----
    The defaults produce a *tiny* model (~a few million parameters) that
    can be trained on CPU in minutes. For a real workload, scale
    ``hidden_size``, ``num_layers``, and ``num_heads`` up by 10-50x and
    use a GPU.
    """

    # Vocabulary and sequence
    vocab_size: int = 1024
    max_seq_len: int = 256

    # Architecture
    hidden_size: int = 192
    num_layers: int = 4
    num_heads: int = 6
    num_kv_heads: Optional[int] = None
    intermediate_size: Optional[int] = None

    # Regularisation
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Architectural choices
    use_rope: bool = True
    rope_theta: float = 10_000.0
    rope_scaling_factor: float = 1.0
    use_swiglu: bool = True
    use_rms_norm: bool = True
    tie_embeddings: bool = True

    # Initialization
    initializer_range: float = 0.02

    # Memory
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        """Validate and fill in defaults after construction."""
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        if self.intermediate_size is None:
            if self.use_swiglu:
                raw = int(8 * self.hidden_size / 3)
                self.intermediate_size = ((raw + 63) // 64) * 64
            else:
                self.intermediate_size = 4 * self.hidden_size

        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads}) for GQA"
        )
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"

    @property
    def head_dim(self) -> int:
        """Per-head dimensionality ``d_k = d_model / num_heads``."""
        return self.hidden_size // self.num_heads

    @property
    def num_kv_groups(self) -> int:
        """How many query heads share each KV head."""
        return self.num_heads // self.num_kv_heads

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for checkpointing)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMConfig":
        """Deserialize from a plain dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ==============================================================================
# SECTION 2: BYTE-LEVEL BPE TOKENIZER
# ==============================================================================

class BPETokenizer:
    """
    A minimal, educational Byte-Pair-Encoding tokenizer.

    This mirrors the *spirit* of the tokenizers
    ("LLM Tokenization"): start from a raw byte vocabulary and iteratively
    merge the most frequent adjacent pair of symbols into a new token, until
    the desired vocabulary size is reached.

    Production tokenizers (GPT-4's ``tiktoken``, Llama's SentencePiece) are
    optimizations of this same idea with additional engineering (regex
    pre-splitting, byte fallback, whitespace markers, etc.). This
    implementation keeps the algorithm bare so you can *see* what is
    happening.

    Design decisions
    --------------------------------------------
    Byte-level: works for any UTF-8 text without an ``<UNK>`` token.
    Greedy merges in learned order**: when encoding, we apply merges in
      the exact order they were learned -- this is the canonical BPE rule.
    Special tokens: ``<|bos|>``, ``<|pad|>``, ``<|eos|>``,
      ``<|unk|>`` are reserved for chat templating

    .. versionchanged:: 1.1
        ``encode(allow_special=True)`` now uses a regex-based splitter
        instead of a NUL-byte sentinel. This fixes corruption of text
        containing NUL bytes and mis-classification of bare-digit
        substrings as special-token IDs.

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size target (including the 256 byte tokens and
        any special tokens).
    special_tokens : Optional[Dict[str, int]]
        Optional mapping of special-token strings to their IDs. If ``None``,
        a sensible default chat set is used.

    Attributes
    ----------
    merges : Dict[Tuple[int, int], int]
        Map from ``(left_id, right_id)`` to the newly created merged token ID.
    vocab : Dict[int, bytes]
        Map from token ID to its byte sequence.
    """

    DEFAULT_SPECIALS = {
        "<|bos|>": 256,
        "<|pad|>": 257,
        "<|eos|>": 258,
        "<|unk|>": 259,
    }

    def __init__(
        self,
        vocab_size: int = 1024,
        special_tokens: Optional[Dict[str, int]] = None,
    ) -> None:
        self.target_vocab_size = vocab_size
        self.special_tokens: Dict[str, int] = (
            dict(special_tokens) if special_tokens else dict(self.DEFAULT_SPECIALS)
        )
        self.specials_inv: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}

        # Vocabulary starts as the 256 byte values, then specials are added.
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        next_id = 256
        for tok_str in self.special_tokens:
            self.vocab[next_id] = tok_str.encode("utf-8")
            next_id += 1

        # Merges learned during `train`
        self.merges: Dict[Tuple[int, int], int] = {}
        self._trained: bool = False
        self._special_pattern: Optional["re.Pattern"] = None

    # ------------------------------------------------------------------ #
    #  Special-token regex
    # ------------------------------------------------------------------ #
    def _compile_special_pattern(self) -> "re.Pattern":
        """Compile a regex that matches any registered special token, longest first."""
        if not self.special_tokens:
            return re.compile(r"(?!)")  # never matches
        sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        escaped = [re.escape(s) for s in sorted_specials]
        return re.compile("|".join(escaped))

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def train(self, text: str, verbose: bool = False) -> None:
        """
        Learn BPE merges from ``text`` until ``target_vocab_size`` is reached.

        Parameters
        ----------
        text : str
            Training corpus. Should be reasonably large (>= 100KB) for
            meaningful merges.
        verbose : bool
            If True, log progress every 100 merges.
        """
        tokens: List[int] = list(text.encode("utf-8"))

        while len(self.vocab) < self.target_vocab_size:
            counts: Counter = Counter(zip(tokens[:-1], tokens[1:]))
            if not counts:
                break

            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break

            new_id = len(self.vocab)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            # Rewrite `tokens`, replacing every occurrence of `pair`.
            new_tokens: List[int] = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]
                ):
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

            if verbose and (len(self.vocab) % 100 == 0):
                logger.info(
                    f"[BPE] vocab={len(self.vocab)} "
                    f"merged {pair} -> {new_id} (freq={freq})"
                )

        self._trained = True
        self._special_pattern = self._compile_special_pattern()

    # ------------------------------------------------------------------ #
    #  Encoding
    # ------------------------------------------------------------------ #
    def encode(self, text: str, allow_special: bool = False) -> List[int]:
        """
        Encode a string to a list of token IDs.

        .. versionchanged:: 1.1
            Uses an explicit regex split for special tokens instead of a
            NUL-byte sentinel. Fixes corruption of text containing NUL
            bytes and mis-classification of bare-digit substrings.

        Parameters
        ----------
        text : str
            Input string.
        allow_special : bool
            If True, occurrences of the registered special-token strings
            (e.g., ``<|eos|>``) are replaced with their reserved IDs before
            BPE. If False, special-token substrings are treated as raw text
            (safer for arbitrary user input).

        Returns
        -------
        List[int]
        """
        if not self._trained:
            return list(text.encode("utf-8"))

        if not allow_special or not self.special_tokens:
            return self._bpe_encode(text)

        # Compile lazily if state was loaded from disk.
        if self._special_pattern is None:
            self._special_pattern = self._compile_special_pattern()

        out: List[int] = []
        for piece in self._special_pattern.split(text):
            if not piece:
                continue
            if piece in self.special_tokens:
                out.append(self.special_tokens[piece])
            else:
                out.extend(self._bpe_encode(piece))
        return out

    def _bpe_encode(self, text: str) -> List[int]:
        """Apply learned merges to a plain string (no specials)."""
        tokens: List[int] = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            pairs = set(zip(tokens[:-1], tokens[1:]))
            candidates = [(self.merges[p], p) for p in pairs if p in self.merges]
            if not candidates:
                break
            # Apply the *earliest-learned* merge first (canonical BPE).
            _, pair = min(candidates)

            new_tokens: List[int] = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]
                ):
                    new_tokens.append(self.merges[pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    # ------------------------------------------------------------------ #
    #  Decoding
    # ------------------------------------------------------------------ #
    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """
        Decode a sequence of token IDs back to a string.

        Parameters
        ----------
        ids : Iterable[int]
        skip_special : bool
            If True, special tokens are removed from the output.
        """
        chunks: List[bytes] = []
        for i in ids:
            if skip_special and i in self.specials_inv:
                continue
            chunks.append(self.vocab.get(i, b""))
        raw = b"".join(chunks)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> None:
        """Pickle the tokenizer's learned state to disk."""
        state = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "target_vocab_size": self.target_vocab_size,
            "_trained": self._trained,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        """Load a tokenizer previously saved with ``.save()``."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        tok = cls(
            vocab_size=state["target_vocab_size"],
            special_tokens=state["special_tokens"],
        )
        tok.vocab = state["vocab"]
        tok.merges = state["merges"]
        tok._trained = state["_trained"]
        tok._special_pattern = tok._compile_special_pattern()
        return tok

    def __len__(self) -> int:
        """Current vocabulary size."""
        return len(self.vocab)

    @property
    def bos_id(self) -> int:
        return self.special_tokens["<|bos|>"]

    @property
    def pad_id(self) -> int:
        return self.special_tokens["<|pad|>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<|eos|>"]

    @property
    def unk_id(self) -> int:
        return self.special_tokens["<|unk|>"]


# ==============================================================================
# SECTION 3: NORMALIZATION LAYERS
# ==============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Ch. 3, "The Transformer Block").

    Unlike LayerNorm, RMSNorm does **not** subtract the mean or apply a
    bias; it only rescales by the root-mean-square of the activations:

    .. math::
        y = x \\cdot \\mathrm{rsqrt}(\\mathrm{mean}(x^2) + \\epsilon) \\cdot w

    This is cheaper to compute and has been shown to match LayerNorm quality
    in modern LLMs (Llama, Mistral, Gemma).

    Parameters
    ----------
    dim : int
        Size of the last dimension of ``x``.
    eps : float
        Numerical stabiliser inside the square root.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `float32` up-cast is important: fp16 RMS can overflow.
        dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * rms).to(dtype) * self.weight


class LayerNorm(nn.Module):
    """
    Standard LayerNorm, provided as a drop-in alternative to RMSNorm.

    Pre-norm (Ch. 3) is the standard placement in modern LLMs. The ``bias``
    is included for compatibility with the original Transformer paper.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


def make_norm(dim: int, use_rms: bool) -> nn.Module:
    """Factory: ``RMSNorm`` if ``use_rms``, else ``LayerNorm``."""
    return RMSNorm(dim) if use_rms else LayerNorm(dim)


# ==============================================================================
# SECTION 4: ROTARY POSITION EMBEDDINGS (RoPE)
# ==============================================================================

def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the cos/sin tables for RoPE ( "Positional Embeddings").

    RoPE encodes position by rotating the (even, odd) pairs of a query/key
    vector by an angle that grows with position. Rotations across many
    frequencies let the model represent both absolute and relative offsets.

    Parameters
    ----------
    head_dim : int
        Per-head dimension (must be even).
    max_seq_len : int
        Number of positions to precompute.
    theta : float
        Base frequency. Larger values better preserve very long contexts.
    device : Optional[torch.device]
        Device on which to place the tables.

    Returns
    -------
    cos, sin : Tuple[torch.Tensor, torch.Tensor]
        Each of shape ``(max_seq_len, head_dim)``.
    """
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"

    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()

    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def precompute_rope_frequencies_ntk(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    scaling_factor: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE cos/sin with NTK-by-parts scaling.

    .. versionadded:: 1.1

    Multiplies the base ``theta`` by
    ``scaling_factor ** (head_dim / (head_dim - 2))``, which stretches
    high-frequency components less than low-frequency ones -- preserving
    local structure while extending reach. This is one of the standard
    techniques used by Llama-style models to extrapolate RoPE beyond the
    trained context length.

    Parameters
    ----------
    head_dim, max_seq_len, theta, device
        Same as :func:`precompute_rope_frequencies`.
    scaling_factor : float
        Ratio of ``desired_context / trained_context``. 1.0 = no scaling.

    Returns
    -------
    cos, sin : Tuple[torch.Tensor, torch.Tensor]
        Each of shape ``(max_seq_len, head_dim)``.
    """
    assert head_dim % 2 == 0
    assert head_dim > 2, "NTK scaling requires head_dim > 2"
    if scaling_factor <= 1.0:
        return precompute_rope_frequencies(head_dim, max_seq_len, theta, device)
    effective_theta = theta * (scaling_factor ** (head_dim / (head_dim - 2)))
    return precompute_rope_frequencies(head_dim, max_seq_len, effective_theta, device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the second half of the last dimension to the front (with a
    sign flip). Used inside :func:`apply_rope`.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to ``q`` and ``k``.

    Parameters
    ----------
    q, k : torch.Tensor
        Shape ``(B, num_heads, S, head_dim)``.
    cos, sin : torch.Tensor
        Shape ``(S, head_dim)`` -- one row per position.

    Returns
    -------
    q_rot, k_rot : same shapes as inputs.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ==============================================================================
# SECTION 5: ATTENTION MASK BUILDER
# ==============================================================================

def build_attention_mask(
    query_length: int,
    key_length: int,
    past_len: int,
    pad_mask: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Build an additive float attention mask of shape ``(B, 1, S_q, S_kv)``
    combining causal masking with optional padding.

    .. versionadded:: 1.1

    The mask is *additive*: ``0.0`` for attended positions and
    ``-inf`` for masked positions. This is the format expected by
    :func:`torch.nn.functional.scaled_dot_product_attention`.

    The mask is *block-causal*: position ``i`` (in the current chunk of
    length ``query_length``) may attend to key position ``j`` (in the
    full key stream of length ``key_length = past_len + query_length``)
    iff ``j <= past_len + i``. This correctly handles three cases:

      1. **Pure prefill** (``past_len == 0, query_length > 1``):
         standard lower-triangular causal mask.
      2. **Pure decode** (``query_length == 1``): no mask needed (returned
         as ``None`` so SDPA can take its fast path).
      3. **Chunked decode** (``past_len > 0, query_length > 1``):
         past is fully visible, current chunk is lower-triangular.

    Padding mask (if provided) is *additively* combined with the causal
    component. The ``pad_mask`` should cover the *full* key stream
    (i.e., length ``key_length``), with ``1`` for valid tokens and
    ``0`` for pad tokens.

    Parameters
    ----------
    query_length : int
        Number of query positions (``S_q``).
    key_length : int
        Total number of key positions (``S_kv = past_len + query_length``).
    past_len : int
        Number of cached past positions.
    pad_mask : Optional[torch.Tensor]
        ``(B, key_length)`` with 1=valid, 0=pad, or None.
    device, dtype : torch.device, torch.dtype
    batch_size : Optional[int]
        Required when ``pad_mask`` is None but a causal-only mask is
        being built.

    Returns
    -------
    Optional[torch.Tensor]
        ``(B, 1, S_q, S_kv)`` additive float mask, or ``None`` when no
        mask is needed (single-token decode without padding).
    """
    if pad_mask is not None:
        B = pad_mask.size(0)
    elif batch_size is not None:
        B = batch_size
    else:
        B = 1

    need_causal = query_length > 1
    need_pad = pad_mask is not None

    if not need_causal and not need_pad:
        return None

    if need_causal:
        i_idx = torch.arange(query_length, device=device).unsqueeze(1)  # (S_q, 1)
        j_idx = torch.arange(key_length, device=device).unsqueeze(0)    # (1, S_kv)
        allowed = j_idx <= (past_len + i_idx)                            # (S_q, S_kv)
        causal = torch.where(allowed, 0.0, float("-inf")).to(dtype)
        mask = causal[None, None, :, :].expand(B, 1, query_length, key_length)
    else:
        mask = torch.zeros(
            (B, 1, query_length, key_length), device=device, dtype=dtype
        )

    if need_pad:
        pad_additive = torch.where(
            pad_mask.bool(), 0.0, float("-inf")
        ).to(dtype)  # (B, S_kv)
        mask = mask + pad_additive[:, None, None, :]

    return mask


# ==============================================================================
# SECTION 6: KV CACHE CONTAINER
# ==============================================================================

class KVCache:
    """
    Per-layer KV cache container.

    .. versionadded:: 1.1

    Replaces the raw ``List[Tuple[Tensor, Tensor]]`` convention with a
    typed container that:

      * Enforces shape invariants per layer.
      * Supports ``reset()``, ``detach()``, ``to(device)``.
      * Is iterable / indexable, so it can be passed through the
        transformer stack cleanly.

    Attributes
    ----------
    caches : List[Optional[Tuple[Tensor, Tensor]]]
        One ``(k, v)`` tuple per layer, or ``None`` for a fresh cache.
    """

    def __init__(self, num_layers: int) -> None:
        self._caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers

    def __len__(self) -> int:
        """Number of cached *positions* (0 if no layer has been populated)."""
        first = self._caches[0]
        return 0 if first is None else first[0].shape[2]

    def __getitem__(self, i: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self._caches[i]

    def __setitem__(self, i: int, val: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self._caches[i] = val

    def __iter__(self) -> Iterator[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        return iter(self._caches)

    def reset(self) -> None:
        """Drop all cached keys/values."""
        self._caches = [None] * len(self._caches)

    def to(self, device: torch.device) -> "KVCache":
        """Move all cached tensors to ``device``."""
        for i, c in enumerate(self._caches):
            if c is not None:
                self._caches[i] = (c[0].to(device), c[1].to(device))
        return self

    def detach(self) -> "KVCache":
        """Return a new cache with all tensors detached (no grad history)."""
        new = KVCache(len(self._caches))
        for i, c in enumerate(self._caches):
            if c is not None:
                new._caches[i] = (c[0].detach(), c[1].detach())
        return new

    @property
    def is_empty(self) -> bool:
        return len(self) == 0


# ==============================================================================
# SECTION 7: MULTI-HEAD ATTENTION  (Chapter 3)
# ==============================================================================

class MultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention with optional Grouped-Query Attention
    (GQA) and KV-cache support .

    The forward pass implements the two conceptual steps of attention
     "Attention is all you need"):

        1. **Relevance scoring** -- ``softmax(Q K^T / sqrt(d_k))``
        2. **Information combining** -- weighted sum over values.

    Enhancements
    ------------
    * **RoPE** applied to Q and K just before scoring (matches Llama).
    * **GQA** via ``repeat_interleave`` of K/V heads.
    * **KV cache** -- the returned per-layer cache should be fed back in
      on the next generation step to avoid re-computing past tokens.
    * **SDPA** -- PyTorch's ``scaled_dot_product_attention`` is used,
      which transparently picks Flash-Attention / Memory-Efficient
      backends on supported GPUs ) "Flash Attention").

    .. versionchanged:: 1.1
        The causal-mask logic has been rewritten to correctly handle
        the *chunked decode* case (``S > 1`` with a non-empty KV cache).
        Previously, multi-token decode steps would silently skip the
        causal mask, leaking future tokens. The new logic uses
        :func:`build_attention_mask` to construct a block-causal mask in
        all cases. The fp16-to-fp32 up-cast (which disabled
        FlashAttention kernels) has also been removed; SDPA handles
        mixed precision natively.

    Parameters
    ----------
    config : LLMConfig
        Supplies ``hidden_size``, ``num_heads``, ``num_kv_heads``, ``dropout``.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.attn_dropout = config.attention_dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, S, hidden_size)``.
        cos, sin : Optional[torch.Tensor]
            RoPE tables of shape ``(S_total, head_dim)``. ``S_total`` must
            cover past cached positions + current ``S``. If None, RoPE is
            skipped (only valid when ``config.use_rope`` is False).
        kv_cache : Optional[Tuple[Tensor, Tensor]]
            Past keys and values of shape
            ``(B, num_kv_heads, S_past, head_dim)``.
        attention_mask : Optional[torch.Tensor]
            **Additive** float mask of shape ``(B, 1, S_q, S_kv)`` (as
            produced by :func:`build_attention_mask`). Use for padding +
            causal control. ``None`` lets SDPA pick its fast path
            (``is_causal=True`` when ``S > 1`` and no cache).

        Returns
        -------
        out : torch.Tensor
            ``(B, S, hidden_size)``.
        new_kv_cache : Tuple[Tensor, Tensor]
            Updated cache -- pass this back on the next step.
        """
        B, S, _ = x.shape
        S_past = 0 if kv_cache is None else kv_cache[0].shape[2]

        # ---- 1. Project Q, K, V -----------------------------------------
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ---- 2. RoPE ----------------------------------------------------
        if cos is not None and sin is not None:
            cos_slice = cos[S_past: S_past + S]
            sin_slice = sin[S_past: S_past + S]
            q, k = apply_rope(q, k, cos_slice, sin_slice)

        # ---- 3. KV cache concat ----------------------------------------
        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)
        new_kv_cache = (k, v)

        S_kv = k.shape[2]

        # ---- 4. GQA: repeat KV heads to match Q heads ------------------
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # ---- 5. SDPA -----------------------------------------------------
        # If a pre-built additive mask is supplied, use it directly.
        # Otherwise let SDPA take its `is_causal=True` fast path when
        # appropriate (training prefill with no padding mask).
        if attention_mask is not None:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            # No mask supplied: pure prefill (no cache) -> causal.
            # Pure decode (cache + S=1) -> no causal needed.
            need_is_causal = (S > 1) and (S_past == 0)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=need_is_causal,
            )

        # ---- 6. Output projection ---------------------------------------
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.resid_dropout(self.o_proj(attn_out))
        return out, new_kv_cache


# ==============================================================================
# SECTION 8: FEEDFORWARD NETWORK
# ---# ==============================================================================

class FeedForward(nn.Module):
    """
    Position-wise feedforward network (Ch. 3, "The feedforward neural
    network at a glance"). This is where the model stores the bulk of its
    learned knowledge.

    Two variants are supported:

    * **GELU** (standard): ``Linear -> GELU -> Linear``.
    * **SwiGLU** (Llama-style): ``(SiLU(W1 x) * W3 x) -> W2``. SwiGLU uses
      three matrices instead of two, which is why ``intermediate_size`` is
      scaled by 8/3 rather than 4.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.use_swiglu = config.use_swiglu

        if self.use_swiglu:
            self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        else:
            self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            gate = F.silu(self.w1(x))
            value = self.w3(x)
            return self.dropout(self.w2(gate * value))
        else:
            return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# ==============================================================================
# SECTION 9: TRANSFORMER BLOCK  (Chapter 3)
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer block (Ch. 3, "The Transformer Block").

    Structure (pre-norm, the modern default):

    .. code-block::

        x = x + Attention(Norm(x))
        x = x + FeedForward(Norm(x))

    Pre-norm (as opposed to the original post-norm) is more stable to
    train and is what modern LLMs (Llama, Mistral, GPT-NeoX) use.

    Residual connections around each sublayer are critical to let gradients
    flow through dozens of stacked blocks.

    .. versionchanged:: 1.1
        Added optional gradient checkpointing (``config.gradient_checkpointing``)
        for memory-efficient training of larger models.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.attn_norm = make_norm(config.hidden_size, config.use_rms_norm)
        self.attn = MultiHeadAttention(config)
        self.ffn_norm = make_norm(config.hidden_size, config.use_rms_norm)
        self.ffn = FeedForward(config)
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        def _inner(x_inner: torch.Tensor):
            h = self.attn_norm(x_inner)
            h, new_cache = self.attn(
                h, cos=cos, sin=sin, kv_cache=kv_cache,
                attention_mask=attention_mask,
            )
            x_out = x_inner + h
            h2 = self.ffn_norm(x_out)
            h2 = self.ffn(h2)
            return x_out + h2, new_cache

        # Only checkpoint during training, with no KV cache (training never
        # uses the cache anyway).
        use_ckpt = (
            self.gradient_checkpointing
            and self.training
            and kv_cache is None
            and x.requires_grad
        )
        if use_ckpt:
            x, new_kv_cache = torch.utils.checkpoint.checkpoint(
                _inner, x, use_reentrant=False
            )
        else:
            x, new_kv_cache = _inner(x)
        return x, new_kv_cache


# ==============================================================================
# SECTION 10: THE LLM
# ==============================================================================

class LLM(nn.Module):
    """
    A decoder-only Transformer language model (GPT-style).

    Architecture
    ------------
    .. code-block::

        input_ids -> TokenEmbed -+
                                  +-> [TransformerBlock] x N -> Norm -> LM Head -> logits
        positions -> (RoPE) -----+

    Because this is an *autoregressive* model, it predicts one token per
    position given all previous tokens, and generation proceeds token by
    token -- optionally accelerated by KV caching.

    .. versionchanged:: 1.1
        * Added ``return_hidden=True`` flag for downstream heads
          (classification / embedding) to skip the LM head.
        * Added :meth:`save_pretrained` / :meth:`from_pretrained`
          directory-based serialization.
        * Added :meth:`generate` convenience method (requires
          :meth:`attach_tokenizer`).
        * The ``attention_mask`` argument now follows the HuggingFace
          convention: a ``(B, S)`` tensor with 1=valid, 0=pad (previously
          an additive float mask). Internally converted via
          :func:`build_attention_mask`.
        * RoPE tables are bounds-checked during long-context generation.
        * Accepts a :class:`KVCache` instance in addition to raw lists.

    Parameters
    ----------
    config : LLMConfig
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.config = config

        # ---- 1. Token embedding -----------------------------------------
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        # ---- 2. Positional embedding (learned fallback) -----------------
        if not config.use_rope:
            self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
            self.register_buffer("rope_cos", None, persistent=False)
            self.register_buffer("rope_sin", None, persistent=False)
        else:
            self.pos_embed = None
            cos, sin = precompute_rope_frequencies_ntk(
                head_dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
                scaling_factor=config.rope_scaling_factor,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        # ---- 3. Transformer blocks --------------------------------------
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # ---- 4. Final norm + LM head ------------------------------------
        self.final_norm = make_norm(config.hidden_size, config.use_rms_norm)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ---- 5. Weight tying (GPT-2 / Llama) ----------------------------
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # ---- 6. Initialization ------------------------------------------
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if (
                name.endswith("o_proj.weight")
                or name.endswith("w2.weight")
                or name.endswith("fc2.weight")
            ):
                nn.init.normal_(
                    p, mean=0.0,
                    std=config.initializer_range / math.sqrt(2 * config.num_layers),
                )

        # Tokenizer is attached lazily for `generate()`.
        self._tokenizer: Optional[BPETokenizer] = None
        self._generator: Optional["TextGenerator"] = None

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #
    def _init_weights(self, module: nn.Module) -> None:
        """Apply the config's normal init to every Linear and Embedding."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[Union[KVCache, List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_kv_caches: bool = False,
        return_hidden: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, KVCache],
    ]:
        """
        Run a forward pass.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Shape ``(B, S)``.
        kv_caches : Optional[Union[KVCache, List]]
            One ``(k, v)`` tuple per layer (or a :class:`KVCache`),
            or ``None`` for a fresh pass.
        attention_mask : Optional[torch.Tensor]
            ``(B, S)`` tensor with 1=valid, 0=pad (HuggingFace convention).
            Internally converted to an additive causal+padding mask.
        return_kv_caches : bool
            If True, also return the updated per-layer KV caches (as a
            :class:`KVCache`).
        return_hidden : bool
            If True, return the final-norm hidden states (skipping the LM
            head). Used by :class:`ClassificationHead` and
            :class:`EmbeddingModel`.

        Returns
        -------
        logits : torch.Tensor
            Shape ``(B, S, vocab_size)`` (or hidden states if
            ``return_hidden=True``).
        kv_caches : KVCache (optional)
        """
        B, S = input_ids.shape
        device = input_ids.device

        # ---- 1. Token embeddings ----------------------------------------
        x = self.embed_tokens(input_ids)

        # ---- 2. Positional embeddings (if not RoPE) ----------------------
        past_len = 0
        if kv_caches is not None:
            # Support both KVCache and raw list.
            first = kv_caches[0] if isinstance(kv_caches, (list, KVCache)) else None
            # KVCache[0] may return None; check.
            if first is not None:
                past_len = first[0].shape[2]

        if self.pos_embed is not None:
            total_len = past_len + S
            if total_len > self.config.max_seq_len:
                logger.warning(
                    f"Position {total_len} exceeds max_seq_len "
                    f"{self.config.max_seq_len}; learned positional "
                    f"embedding will be truncated."
                )
                total_len = self.config.max_seq_len
            positions = torch.arange(
                past_len, total_len, device=device, dtype=torch.long
            )
            x = x + self.pos_embed(positions)[None, :, :]

        x = self.embed_dropout(x)

        # ---- 3. RoPE table for this pass --------------------------------
        cos = sin = None
        if self.pos_embed is None and self.rope_cos is not None:
            total_len = past_len + S
            if total_len > self.config.max_seq_len:
                logger.warning(
                    f"Position {total_len} exceeds max_seq_len "
                    f"{self.config.max_seq_len}; relying on RoPE "
                    f"extrapolation (scaling_factor="
                    f"{self.config.rope_scaling_factor})."
                )
            cos = self.rope_cos
            sin = self.rope_sin

        # ---- 4. Build additive attention mask ---------------------------
        # Convert user-supplied (B, S) padding mask to (B, 1, S, S_kv)
        # additive form using build_attention_mask.
        if attention_mask is not None:
            # Pad mask should cover the full key stream.
            if past_len > 0:
                past_ones = torch.ones(
                    attention_mask.size(0), past_len,
                    device=attention_mask.device, dtype=attention_mask.dtype,
                )
                full_pad_mask = torch.cat([past_ones, attention_mask], dim=1)
            else:
                full_pad_mask = attention_mask
            additive_mask = build_attention_mask(
                query_length=S,
                key_length=past_len + S,
                past_len=past_len,
                pad_mask=full_pad_mask,
                device=device,
                dtype=x.dtype,
            )
        else:
            # No padding mask: build causal-only mask (returns None for
            # single-token decode, letting SDPA fast-path).
            additive_mask = build_attention_mask(
                query_length=S,
                key_length=past_len + S,
                past_len=past_len,
                pad_mask=None,
                device=device,
                dtype=x.dtype,
                batch_size=B,
            )

        # ---- 5. Transformer blocks --------------------------------------
        new_caches = KVCache(num_layers=len(self.blocks))
        for i, block in enumerate(self.blocks):
            layer_cache = None
            if kv_caches is not None:
                layer_cache = kv_caches[i]
            x, new_cache = block(
                x, cos=cos, sin=sin, kv_cache=layer_cache,
                attention_mask=additive_mask,
            )
            new_caches[i] = new_cache

        # ---- 6. Final norm + optional LM head --------------------------
        hidden = self.final_norm(x)
        if return_hidden:
            if return_kv_caches:
                return hidden, new_caches
            return hidden

        logits = self.lm_head(hidden)

        if return_kv_caches:
            return logits, new_caches
        return logits

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of (trainable) parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        """
        Build an AdamW optimizer with proper parameter grouping.

        Weight decay is applied only to matrices that participate in
        matrix multiplication (Linear/Embedding weights) and skipped for
        norms and biases -- a standard practice from nanoGPT / Llama.
        """
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return AdamW(groups, lr=lr, betas=betas)

    # ------------------------------------------------------------------ #
    #  Generation convenience
    # ------------------------------------------------------------------ #
    def attach_tokenizer(self, tokenizer: BPETokenizer) -> None:
        """Attach a tokenizer so :meth:`generate` works."""
        self._tokenizer = tokenizer
        self._generator = None  # invalidate cached generator

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Thin wrapper around :class:`TextGenerator`.

        Requires :meth:`attach_tokenizer` to have been called first.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Call model.attach_tokenizer(tokenizer) before generate()."
            )
        if self._generator is None:
            self._generator = TextGenerator(self, self._tokenizer)
        return self._generator.generate(prompt, **kwargs)

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model weights + config in a single file."""
        torch.save(
            {"config": self.config.to_dict(), "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "LLM":
        """Reconstruct a model from a ``.save_checkpoint()`` file."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = LLMConfig.from_dict(ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model

    def save_pretrained(self, directory: Union[str, Path]) -> None:
        """
        Persist model + config to a directory (HuggingFace-style).

        Layout::

            directory/
                config.json
                model.pt
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2)
        )
        torch.save(self.state_dict(), d / "model.pt")

    @classmethod
    def from_pretrained(
        cls, directory: Union[str, Path], map_location: str = "cpu"
    ) -> "LLM":
        """Load a directory previously saved with :meth:`save_pretrained`."""
        d = Path(directory)
        cfg = LLMConfig.from_dict(
            json.loads((d / "config.json").read_text())
        )
        model = cls(cfg)
        state = torch.load(d / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model


# ==============================================================================
# SECTION 11: SAMPLING & GENERATION
# ==============================================================================

class TextGenerator:
    """
    Handles autoregressive decoding with a suite of sampling strategies.

    Implements every strategy "Choosing a Single Token
    from the Probability Distribution" and "Controlling Model
    Output":

    -Greedy -- argmax (temperature = 0)
    -Temperature scaling -- divides logits by T
    -Top-k filtering-- keep only the k highest-probability tokens
    -Top-p / nucleus sampling -- keep the smallest set whose
      cumulative probability exceeds p
    -min-p sampling -- keep tokens whose probability is at least
      ``min_p * max_prob`` (modern alternative to top-p, more robust on
      long-tail distributions)
    -Repetition penalty -- downweights tokens already generated
    -KV cache- -- avoids recomputing past states, ~4-5x faster

    .. versionchanged:: 1.1
        * Added ``min_p`` sampling.
        * Added logit clamping (``[-1e9, 1e9]``) for numerical stability.
        * Repetition penalty is now functional (no in-place mutation)
          using the canonical HF ``torch.where`` formulation.
        * Added :meth:`generate_batch` for batched, left-padded
          parallel generation.
        * Added :meth:`stream_generate` for token-by-token streaming.

    Parameters
    ----------
    model : LLM
        The language model to sample from.
    tokenizer : BPETokenizer
        Used to decode/encode text.
    """

    def __init__(self, model: LLM, tokenizer: BPETokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------------ #
    #  Filtering helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """
        Downweight tokens that have already appeared.

        .. versionchanged:: 1.1
            Now returns a *new* tensor (no in-place mutation) and uses
            the canonical HF formulation: positive logits are divided
            by ``penalty`` and non-positive logits are multiplied -- both
            push repeated tokens toward lower probability.
        """
        if penalty == 1.0:
            return logits
        out = logits.clone()
        B = out.shape[0]
        for b in range(B):
            seen = torch.unique(generated_ids[b])
            scores = out[b, seen]
            out[b, seen] = torch.where(
                scores > 0, scores / penalty, scores * penalty
            )
        return out

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only the top-k logits; others become -inf."""
        if k <= 0:
            return logits
        topk_vals, _ = torch.topk(logits, min(k, logits.size(-1)))
        cutoff = topk_vals[..., -1, None]
        return torch.where(logits < cutoff, float("-inf"), logits)

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus sampling: keep the smallest set with cumprob >= p."""
        if p >= 1.0:
            return logits
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        # Remove tokens once we've crossed p (keep the first crossing).
        mask = cumprobs - probs > p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        out = torch.empty_like(logits)
        out.scatter_(-1, sorted_idx, sorted_logits)
        return out

    @staticmethod
    def _min_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
        """
        Remove tokens whose softmax probability is below ``p * max_prob``.

        .. versionadded:: 1.1

        ``min_p`` is dynamically adaptive: it scales with the model's
        confidence, so high-confidence steps keep few tokens and
        low-confidence steps keep more. Empirically outperforms top-p on
        long-tail distributions.
        """
        if p <= 0.0:
            return logits
        probs = F.softmax(logits, dim=-1)
        max_prob, _ = probs.max(dim=-1, keepdim=True)
        keep = probs >= (p * max_prob)
        return torch.where(keep, logits, torch.full_like(logits, float("-inf")))

    # ------------------------------------------------------------------ #
    #  Core step
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        generated_ids: torch.Tensor,
        repetition_penalty: float,
        greedy: bool,
    ) -> torch.Tensor:
        """Given logits of shape ``(B, V)``, produce a next-token ID per batch item."""
        # Clamp for numerical stability before any softmax.
        logits = logits.float().clamp(min=-1e9, max=1e9)

        # 1. Repetition penalty.
        logits = self._apply_repetition_penalty(
            logits, generated_ids, repetition_penalty
        )

        # 2. Greedy shortcut.
        if greedy or temperature == 0.0:
            return logits.argmax(dim=-1)

        # 3. Temperature.
        logits = logits / max(temperature, 1e-8)

        # 4. Top-k, top-p, min-p.
        logits = self._top_k_filter(logits, top_k)
        logits = self._top_p_filter(logits, top_p)
        logits = self._min_p_filter(logits, min_p)

        # 5. Sample.
        probs = F.softmax(logits, dim=-1)
        invalid_rows = (probs.sum(-1) == 0) | torch.isnan(probs).any(-1)
        if invalid_rows.any():
            probs[invalid_rows] = 1.0 / probs.size(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # ------------------------------------------------------------------ #
    #  Single-prompt generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
        stop_on_eos: bool = True,
        return_logprobs: bool = False,
    ) -> Union[str, Tuple[str, List[float]]]:
        """
        Generate text from a single string prompt.

        Parameters
        ----------
        prompt : str
        max_new_tokens : int
        temperature : float
            0 -> greedy. > 0 -> stochastic.
        top_k : int
            0 -> disabled.
        top_p : float
            1.0 -> disabled.
        min_p : float
            0.0 -> disabled.
        repetition_penalty : float
            1.0 -> disabled; > 1 downweights repeated tokens.
        do_sample : bool
            If False, greedy.
        use_cache : bool
            Enable KV cache. Disabling makes generation O(S^3) per step.
        stop_on_eos : bool
            Stop once ``<|eos|>`` is produced.
        return_logprobs : bool
            If True, also return per-step log-probabilities of the
            chosen tokens.

        Returns
        -------
        str  or  (str, List[float])
        """
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, allow_special=True)
        if not input_ids:
            input_ids = [self.tokenizer.eos_id]
        input_ids = torch.tensor([input_ids], device=self.device, dtype=torch.long)

        generated = input_ids
        kv_caches: Optional[KVCache] = None
        logprobs: List[float] = []

        for _ in range(max_new_tokens):
            if use_cache and kv_caches is not None and not kv_caches.is_empty:
                model_in = generated[:, -1:]
            else:
                model_in = generated

            if use_cache:
                logits, kv_caches = self.model(
                    model_in,
                    kv_caches=kv_caches,
                    return_kv_caches=True,
                )
            else:
                logits = self.model(model_in)

            next_logits = logits[:, -1, :]  # (1, V)
            next_id = self._sample_next_token(
                logits=next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                generated_ids=generated,
                repetition_penalty=repetition_penalty,
                greedy=(not do_sample),
            )

            if return_logprobs:
                with torch.no_grad():
                    probs = F.softmax(next_logits.float(), dim=-1)
                    lp = torch.log(probs[0, next_id.item()].clamp_min(1e-12)).item()
                    logprobs.append(lp)

            generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)

            if stop_on_eos and next_id.item() == self.tokenizer.eos_id:
                break
            if generated.shape[1] >= self.model.config.max_seq_len:
                break

        text = self.tokenizer.decode(generated[0].tolist())
        if return_logprobs:
            return text, logprobs
        return text

    # ------------------------------------------------------------------ #
    #  Batched generation
    # ------------------------------------------------------------------ #
    def _left_pad(self, sequences: List[List[int]]) -> torch.Tensor:
        """Pad to max length with ``pad_id`` on the LEFT (causal-LM convention)."""
        max_len = max(len(s) for s in sequences)
        padded = []
        for s in sequences:
            n_pad = max_len - len(s)
            padded.append([self.tokenizer.pad_id] * n_pad + s)
        return torch.tensor(padded, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        stop_on_eos: bool = True,
    ) -> List[str]:
        """
        Batched parallel generation.

        .. versionadded:: 1.1

        All prompts are processed in parallel. Uses **left padding** so
        the final position of every sequence is a real token (otherwise
        the LM head would predict from a pad embedding).

        A per-batch ``finished`` flag tracks which sequences have emitted
        EOS; finished sequences keep producing pad tokens until the rest
        catch up.
        """
        self.model.eval()
        sequences: List[List[int]] = []
        for p in prompts:
            ids = self.tokenizer.encode(p, allow_special=True)
            if not ids:
                ids = [self.tokenizer.eos_id]
            sequences.append(ids)

        input_ids = self._left_pad(sequences)
        # (B, S) padding mask: 1 = real token, 0 = pad.
        pad_mask = (input_ids != self.tokenizer.pad_id).long()
        B = input_ids.size(0)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        kv_caches: Optional[KVCache] = None
        step = 0

        while step < max_new_tokens:
            if kv_caches is None or kv_caches.is_empty:
                # Prefill.
                model_in = input_ids
                mask_in = pad_mask
            else:
                # Decode: feed only the last token of each sequence.
                model_in = input_ids[:, -1:]
                # Decode-step mask: pad_mask now must cover past+current.
                # New token is always "valid".
                new_valid = torch.ones(B, 1, device=self.device, dtype=pad_mask.dtype)
                pad_mask = torch.cat([pad_mask, new_valid], dim=1)
                mask_in = pad_mask[:, -1:]  # only the current position's validity
                # But we need the *full* mask covering past+current for the
                # additive mask builder. LLM.forward extends the mask with
                # past_ones automatically when past_len > 0, so passing the
                # full pad_mask works. However, we only have current S=1
                # query; the additive mask builder will use past_len from
                # the cache. So pass the full pad_mask here.
                mask_in = pad_mask  # (B, past_len + 1)

            logits, kv_caches = self.model(
                model_in,
                kv_caches=kv_caches,
                attention_mask=mask_in,
                return_kv_caches=True,
            )
            next_logits = logits[:, -1, :]
            next_id = self._sample_next_token(
                logits=next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                generated_ids=input_ids,
                repetition_penalty=repetition_penalty,
                greedy=(not do_sample),
            )
            # Force finished sequences to output pad.
            next_id = torch.where(
                finished, torch.full_like(next_id, self.tokenizer.pad_id), next_id
            )
            input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=1)

            if stop_on_eos:
                finished = finished | (next_id == self.tokenizer.eos_id)
                if finished.all():
                    break
            if input_ids.size(1) >= self.model.config.max_seq_len:
                break
            step += 1

        # Strip left padding and decode.
        texts: List[str] = []
        for row in input_ids:
            real_mask = (row != self.tokenizer.pad_id)
            first_real = real_mask.nonzero()
            if len(first_real) > 0:
                row = row[first_real[0].item():]
            texts.append(self.tokenizer.decode(row.tolist()))
        return texts

    # ------------------------------------------------------------------ #
    #  Streaming generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        stop_on_eos: bool = True,
    ) -> Iterator[str]:
        """
        Yield decoded tokens one at a time (prompt UIs).

        .. versionadded:: 1.1

        Usage::

            for chunk in generator.stream_generate(prompt):
                print(chunk, end="", flush=True)
        """
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, allow_special=True)
        if not input_ids:
            input_ids = [self.tokenizer.eos_id]
        input_ids = torch.tensor([input_ids], device=self.device, dtype=torch.long)

        kv_caches: Optional[KVCache] = None
        for _ in range(max_new_tokens):
            if kv_caches is None or kv_caches.is_empty:
                model_in = input_ids
            else:
                model_in = input_ids[:, -1:]

            if kv_caches is None or kv_caches.is_empty:
                logits, kv_caches = self.model(
                    model_in, return_kv_caches=True
                )
            else:
                logits, kv_caches = self.model(
                    model_in, kv_caches=kv_caches, return_kv_caches=True
                )

            next_id = self._sample_next_token(
                logits=logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                generated_ids=input_ids,
                repetition_penalty=repetition_penalty,
                greedy=(not do_sample),
            )

            if stop_on_eos and next_id.item() == self.tokenizer.eos_id:
                yield "<|eos|>"
                break

            input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=1)
            yield self.tokenizer.decode([next_id.item()])

            if input_ids.shape[1] >= self.model.config.max_seq_len:
                break


# ==============================================================================
# SECTION 12: DATASET   packing
# ==============================================================================

class TextDataset(Dataset):
    """
    Sliding-window next-token-prediction dataset (Ch. 3, "packing").

    Given a long corpus, we concatenate all token IDs into one stream,
    then emit every contiguous window of length ``seq_len + 1`` (the extra
    token is the target for the last position).

    This is the standard approach for training language models and
    corresponds to the "packing" technique discussed in Ch. 3 for
    efficiently handling short documents.

    Parameters
    ----------
    token_ids : List[int]
    seq_len : int
        Context length per training example.
    stride : Optional[int]
        Step between successive windows. ``stride == seq_len`` gives no
        overlap; smaller values give overlap (more samples, some
        redundancy).
    """

    def __init__(
        self, token_ids: List[int], seq_len: int, stride: Optional[int] = None
    ) -> None:
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride or seq_len

        n = len(self.data)
        max_start = n - seq_len - 1
        if max_start < 0:
            raise ValueError(
                f"Corpus too short ({n} tokens) for seq_len={seq_len}"
            )
        self.starts = list(range(0, max_start + 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.starts[idx]
        window = self.data[s: s + self.seq_len + 1]
        return window[:-1], window[1:]


# ==============================================================================
# SECTION 13: TRAINER
# ==============================================================================

class LLMTrainer:
    """
    Minimal but production-conscious training loop for a generative LLM.

    Implements techniques :

    - Next-token-prediction cross-entropy loss (the objective used in
      supervised fine-tuning).
    -Gradient accumulation for a larger effective batch size.
    - Cosine LR schedule with linear warmup (standard for LLM training).
    = Gradient clipping.
    = Mixed-precision (AMP) on CUDA: bf16 on Ampere+ (A100, H100),
      fp16 on older CUDA GPUs, fp32 on CPU.
    = Periodic evaluation and checkpointing.
    = Loss masking for padding (``ignore_index`` on pad tokens).

    .. versionchanged:: 1.1
        - Replaced deprecated ``torch.cuda.amp.GradScaler`` with
          ``torch.amp.GradScaler``.
        - Replaced ``torch.autocast`` calls with ``torch.amp.autocast``.
        - Default dtype is now bf16 on Ampere+ GPUs (more stable than fp16).
        - Token throughput is now tracked precisely (was previously
          computed via a confusing formula).
        - Uses the :mod:`logging` module instead of ``print``.
        - Saves the GradScaler state in checkpoints.

    For a full SFT / DPO pipeline , replace the loss with the
    appropriate objective -- the loop structure is the same.
    """

    def __init__(
        self,
        model: LLM,
        tokenizer: BPETokenizer,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        grad_accum_steps: int = 1,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        dtype: Optional[torch.dtype] = None,
        log_every: int = 10,
        eval_every: int = 100,
        save_every: int = 500,
        output_dir: str = "./checkpoints",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Choose AMP dtype automatically.
        if dtype is None:
            if self.device.type == "cuda":
                major, _ = torch.cuda.get_device_capability()
                dtype = torch.bfloat16 if major >= 8 else torch.float16
            else:
                dtype = torch.float32
        self.amp_dtype = dtype
        self.use_amp = (
            use_amp and self.device.type == "cuda" and dtype != torch.float32
        )

        # Optimizer.
        self.optimizer = model.configure_optimizers(
            lr=lr, weight_decay=weight_decay
        )

        # LR schedule: linear warmup to `lr`, then cosine decay to 10% of lr.
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # Modern GradScaler (PyTorch 2.0+).
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        self.step = 0
        self.best_eval_loss = float("inf")

    # ------------------------------------------------------------------ #
    #  Train
    # ------------------------------------------------------------------ #
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
    ) -> None:
        """Run the training loop until ``max_steps`` is reached."""
        self.model.train()
        train_iter = iter(train_loader)
        running_loss = 0.0
        running_tokens = 0
        t0 = time.time()

        while self.step < self.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            step_tokens = 0

            for _ in range(self.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                step_tokens += x.numel()

                with torch.amp.autocast(
                    self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    logits = self.model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.reshape(-1),
                        ignore_index=self.tokenizer.pad_id,
                    ) / self.grad_accum_steps

                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.step += 1
            running_loss += accum_loss
            running_tokens += step_tokens

            # Logging.
            if self.step % self.log_every == 0:
                dt = time.time() - t0
                avg_loss = running_loss / self.log_every
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"step {self.step:>5}/{self.max_steps} "
                    f"loss={avg_loss:.4f} lr={lr:.2e} "
                    f"tok/s={running_tokens / max(dt, 1e-6):,.0f}"
                )
                running_loss = 0.0
                running_tokens = 0
                t0 = time.time()

            # Evaluation.
            if eval_loader is not None and self.step % self.eval_every == 0:
                eval_loss = self.evaluate(eval_loader)
                logger.info(f"step {self.step:>5} eval_loss={eval_loss:.4f}")
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save("best.pt")
                self.model.train()

            # Checkpointing.
            if self.step % self.save_every == 0:
                self.save(f"step_{self.step}.pt")

    # ------------------------------------------------------------------ #
    #  Evaluate
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Compute average cross-entropy loss on a validation loader."""
        self.model.eval()
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with torch.amp.autocast(
                self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits = self.model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=self.tokenizer.pad_id,
                )
            total_loss += loss.item()
            n += 1
        return total_loss / max(1, n)

    # ------------------------------------------------------------------ #
    #  Checkpointing
    # ------------------------------------------------------------------ #
    def save(self, name: str) -> None:
        """Save model + optimizer + scheduler + scaler + step."""
        path = self.output_dir / name
        torch.save(
            {
                "config": self.model.config.to_dict(),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.step,
                "best_eval_loss": self.best_eval_loss,
            },
            path,
        )


# ==============================================================================
# SECTION 14: CLASSIFICATION HEAD
# ==============================================================================

def _pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """
    Pool hidden states along the sequence dimension.

    Parameters
    ----------
    hidden : torch.Tensor
        ``(B, S, H)`` -- pre-LM-head hidden states.
    attention_mask : torch.Tensor
        ``(B, S)`` with 1=valid, 0=pad.
    pooling : {"cls", "mean", "max"}
        * ``cls``  -- last non-pad token (acts as a CLS for causal LMs).
        * ``mean`` -- mean of non-pad tokens.
        * ``max``  -- max-pool over non-pad tokens.
    """
    if pooling == "cls":
        # For causal LMs, "last non-pad token" mirrors BERT's [CLS].
        lengths = attention_mask.sum(dim=1) - 1
        idx = lengths.clamp(min=0)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
    mask = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
    if pooling == "mean":
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts
    if pooling == "max":
        neg_inf = torch.finfo(hidden.dtype).min
        masked = hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), neg_inf)
        return masked.max(dim=1).values
    raise ValueError(f"Unknown pooling: {pooling}")


class ClassificationHead(nn.Module):
    """
    Classification head for text classification .

    .. versionchanged:: 1.1
        This is now a *real* implementation that runs the full backbone
        (previously a stub that only used the embedding layer). Supports
        three pooling strategies, optional backbone freezing, and a
        dropout before the classifier.

    The backbone acts as an encoder of contextualised embeddings; the
    pooled representation feeds a linear classifier. This matches the
    "task-specific model" architecture illustrated in Ch. 4, Figure 4-3.

    Parameters
    ----------
    backbone : LLM
    num_labels : int
    pooling : {"cls", "mean", "max"}
        - ``cls``  -- last-token pooling (good for causal LMs).
        - ``mean`` -- mean of non-pad tokens.
        - ``max``  -- max-pool over non-pad tokens.
    dropout : float
    freeze_backbone : bool
        If True, backbone params are not updated; only the classifier is
        fine-tuned (Ch. 11, "feature-extraction" approach).
    """

    POOLINGS = ("cls", "mean", "max")

    def __init__(
        self,
        backbone: LLM,
        num_labels: int,
        pooling: str = "mean",
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        assert pooling in self.POOLINGS, f"pooling must be one of {self.POOLINGS}"
        self.backbone = backbone
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone.config.hidden_size, num_labels)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        hidden = self.backbone(
            input_ids, attention_mask=attention_mask, return_hidden=True
        )
        pooled = self.dropout(_pool_hidden(hidden, attention_mask, self.pooling))
        return self.classifier(pooled)


# ==============================================================================
# SECTION 15: EMBEDDING MODEL
# ==============================================================================

class EmbeddingModel(nn.Module):
    """
    Text-embedding model .

    .. versionchanged:: 1.1
        Now runs the full backbone (previously a stub). Produces
        mean-pooled, L2-normalised embeddings suitable for cosine
        similarity / contrastive losses.

    Usage with a contrastive loss (e.g., InfoNCE)::

        emb = EmbeddingModel(backbone)
        anchor   = emb(input_ids_a, attention_mask_a)
        positive = emb(input_ids_b, attention_mask_b)
        # In-batch negatives:
        sim = F.cosine_similarity(anchor.unsqueeze(1), positive.unsqueeze(0), dim=-1)
        sim = sim / temperature
        loss = F.cross_entropy(sim, torch.arange(B))


    """

    def __init__(
        self,
        backbone: LLM,
        pooling: str = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        hidden = self.backbone(
            input_ids, attention_mask=attention_mask, return_hidden=True
        )
        pooled = self.dropout(_pool_hidden(hidden, attention_mask, self.pooling))
        return F.normalize(pooled, p=2, dim=-1)


# ==============================================================================
# SECTION 16: REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.

    .. versionchanged:: 1.1
        Added ``deterministic`` flag that, when True, forces cuDNN to
        deterministic algorithms and enables
        ``torch.use_deterministic_algorithms``.

    Parameters
    ----------
    seed : int
    deterministic : bool
        If True, force fully deterministic algorithms (slower but
        reproducible). Default False (typical for training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # older PyTorch


# ==============================================================================
# SECTION 17: DEMO -- TRAIN A TINY MODEL END TO END
# ==============================================================================

def demo() -> None:
    """
    End-to-end demonstration:

        1. Train a small BPE tokenizer on a corpus.
        2. Build a small LLM.
        3. Train it with next-token prediction.
        4. Generate with different sampling strategies.
        5. Verify that KV-cached generation matches non-cached generation.
        6. Demonstrate batched generation.
        7. Demonstrate streaming generation.
        8. Demonstrate save_pretrained / from_pretrained round-trip.
    """
    sep = "=" * 78
    logger.info(sep)
    logger.info("HANDS-ON LARGE LANGUAGE MODELS -- end-to-end demo (v1.1)")
    logger.info(sep)

    set_seed(42)

    # ------------------------------------------------------------------
    # 1. Corpus
    # ------------------------------------------------------------------
    corpus = (
        "the quick brown fox jumps over the lazy dog. "
        "the lazy dog sleeps all day. "
        "the quick brown fox runs through the forest. "
        "the fox and the dog are friends. "
        "the dog and the fox play in the park. "
        "a quick brown fox is clever. "
        "a lazy dog is happy. "
        "the quick fox jumps high. "
        "the lazy dog sleeps well. "
        "the fox runs fast and the dog runs slow. "
    ) * 60

    logger.info(f"Corpus: {len(corpus):,} characters")

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    logger.info("[1/8] Training BPE tokenizer ...")
    tokenizer = BPETokenizer(vocab_size=512)
    tokenizer.train(corpus)
    logger.info(f"    vocabulary size: {len(tokenizer)}")

    encoded = tokenizer.encode(corpus, allow_special=False)
    logger.info(
        f"    corpus -> {len(encoded):,} tokens "
        f"({len(encoded) / len(corpus):.3f} tokens/char)"
    )

    # Round-trip sanity check.
    assert tokenizer.decode(encoded) == corpus, "Tokenizer round-trip failed!"

    # Save tokenizer.
    ckpt_dir = Path("./demo_checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = ckpt_dir / "tokenizer.pkl"
    tokenizer.save(tokenizer_path)
    logger.info(f"    saved tokenizer to {tokenizer_path}")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    logger.info("[2/8] Building LLM ...")
    config = LLMConfig(
        vocab_size=len(tokenizer),
        max_seq_len=64,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.1,
        use_rope=True,
        rope_theta=10_000.0,
        rope_scaling_factor=1.0,
        use_swiglu=True,
        use_rms_norm=True,
        tie_embeddings=True,
        gradient_checkpointing=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLM(config).to(device)
    model.attach_tokenizer(tokenizer)
    logger.info(f"    device: {device}")
    logger.info(f"    parameters: {model.num_parameters():,}")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    logger.info("[3/8] Training ...")
    split = int(len(encoded) * 0.9)
    train_ids, val_ids = encoded[:split], encoded[split:]

    train_ds = TextDataset(train_ids, seq_len=config.max_seq_len)
    val_ds = TextDataset(val_ids, seq_len=config.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, drop_last=True)

    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
        lr=3e-4,
        warmup_steps=20,
        max_steps=200,
        grad_accum_steps=1,
        log_every=20,
        eval_every=100,
        save_every=100,
        output_dir=str(ckpt_dir),
    )
    trainer.train(train_loader, eval_loader=val_loader)

    # ------------------------------------------------------------------
    # 5. Generate (single-prompt)
    # ------------------------------------------------------------------
    logger.info("[4/8] Generating text with different sampling strategies ...")
    generator = TextGenerator(model, tokenizer)

    prompts = [
        "the quick brown fox",
        "the lazy dog",
        "the fox and the dog",
    ]

    strategies = [
        ("greedy",        dict(do_sample=False)),
        ("temp=0.7",      dict(do_sample=True, temperature=0.7, top_k=40)),
        ("temp=1.0+k40",  dict(do_sample=True, temperature=1.0, top_k=40)),
        ("temp=0.9+p0.9", dict(do_sample=True, temperature=0.9, top_p=0.9)),
        ("min_p=0.05",    dict(do_sample=True, temperature=0.9, min_p=0.05)),
    ]

    for prompt in prompts:
        logger.info(f"PROMPT: {prompt!r}")
        for name, kwargs in strategies:
            out = generator.generate(
                prompt,
                max_new_tokens=20,
                use_cache=True,
                stop_on_eos=False,
                **kwargs,
            )
            logger.info(f"    [{name:<12}] {out!r}")

    # ------------------------------------------------------------------
    # 6. Verify KV cache correctness
    # ------------------------------------------------------------------
    logger.info("[5/8] Verifying KV-cache correctness ...")
    model.eval()
    test_prompt = "the quick brown fox jumps"
    ids = torch.tensor(
        [tokenizer.encode(test_prompt, allow_special=True)], device=device
    )
    with torch.no_grad():
        logits_nocache = model(ids)[0, -1]
        _, caches = model(ids, return_kv_caches=True)
        logits_cache, _ = model(
            ids[:, -1:], kv_caches=caches, return_kv_caches=True
        )
        logits_cache = logits_cache[0, -1]

    diff = (logits_nocache - logits_cache).abs().max().item()
    logger.info(f"    max |logits_no_cache - logits_with_cache| = {diff:.2e}")
    assert diff < 1e-3, "KV cache mismatch!"
    logger.info("    OK: KV cache matches non-cached forward pass.")

    # ------------------------------------------------------------------
    # 7. Batched generation
    # ------------------------------------------------------------------
    logger.info("[6/8] Batched generation ...")
    batch_prompts = [
        "the quick brown fox",
        "the lazy dog",
        "the fox and the dog are",
    ]
    batch_outputs = generator.generate_batch(
        batch_prompts,
        max_new_tokens=15,
        temperature=0.7,
        top_k=40,
        do_sample=True,
        stop_on_eos=False,
    )
    for p, o in zip(batch_prompts, batch_outputs):
        logger.info(f"    PROMPT: {p!r}  ->  {o!r}")

    # ------------------------------------------------------------------
    # 8. Streaming generation
    # ------------------------------------------------------------------
    logger.info("[7/8] Streaming generation ...")
    logger.info("    PROMPT: 'the quick brown fox'")
    chunks = []
    for chunk in generator.stream_generate(
        "the quick brown fox",
        max_new_tokens=15,
        temperature=0.7,
        top_k=40,
        stop_on_eos=False,
    ):
        chunks.append(chunk)
    logger.info(f"    streamed: {''.join(chunks)!r}")

    # ------------------------------------------------------------------
    # 9. save_pretrained / from_pretrained
    # ------------------------------------------------------------------
    logger.info("[8/8] save_pretrained / from_pretrained round-trip ...")
    model_dir = ckpt_dir / "model"
    model.save_pretrained(model_dir)
    reloaded = LLM.from_pretrained(model_dir)
    reloaded.attach_tokenizer(tokenizer)
    reloaded.to(device)

    # Verify the reloaded model produces the same logits.
    with torch.no_grad():
        logits_orig = model(ids)[0, -1]
        logits_reloaded = reloaded(ids)[0, -1]
    diff2 = (logits_orig - logits_reloaded).abs().max().item()
    logger.info(f"    max |logits_orig - logits_reloaded| = {diff2:.2e}")
    assert diff2 < 1e-5, "Reloaded model mismatch!"
    logger.info("    OK: round-trip succeeds.")

    # Use the convenience generate() method on the model itself.
    out = reloaded.generate("the lazy dog", max_new_tokens=10, do_sample=False)
    logger.info(f"    reloaded.generate('the lazy dog') = {out!r}")

    logger.info(sep)
    logger.info("Demo complete. Model and checkpoints saved under ./demo_checkpoints")
    logger.info(sep)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    demo()

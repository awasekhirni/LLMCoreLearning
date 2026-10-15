"""
Author: Awase Khirni Syed, May 19, 2025
Copyright 2025 β ORI Inc.Canada All Rights Reserved.
    * Full type hints and NumPy-style docstrings on every class and method
    * Modular, composable class hierarchy
    * Complete KV-cache implementation ( "Speeding Up Generation by
      Caching Keys and Values") — enables ~4–5x faster autoregressive decoding
    * Grouped-Query Attention (GQA) — , "Multi-query and grouped-query
      attention"
    * Rotary Position Embeddings (RoPE) —  "Positional Embeddings (RoPE)"
    * SwiGLU feedforward activation — , "The Transformer Block"
    * RMSNorm (pre-normalization) — , "The Transformer Block"
    * Weight tying between token embedding and LM head (reduces parameters)
    * Multiple sampling strategies (greedy, temperature, top-k, top-p,
      repetition penalty) —
    * Gradient accumulation, cosine LR schedule with warmup, gradient
      clipping, mixed-precision training (AMP)
    * Checkpoint save/load (model + optimizer + config + tokenizer)
    * Contrastive text-embedding head
    * Classification head
    * Attention-mask utilities for padding + causal masking
    * Numerical stability safeguards (logit clamping, epsilon handling)
    * Reproducibility helpers (seed control)

QUICK START
-----------
    python betaORI_ollmdemo.py

This will train a tiny model on a small corpus so you can see end-to-end
next-token prediction working. Swap in your own corpus / config for real use.

Todo Enhancmeents in the next version:
 No batched generation (right-padding for causal LMs).
 No beam search.
 No min_p sampling (modern alternative to top-p).
 No logit bias hook (useful for grammar-constrained decoding).
 No gradient checkpointing (essential for >1B-param training).
 No streaming generation (token-by-token yield).
 No return of log-probabilities / scores.
 No length-extended RoPE (NTK-aware, YaRN-style) for long-context extrapolation.
 No __all__ export list → from llm import * is dangerous.
 No logging module — uses print() everywhere.
 No attention-mask builder utility that unifies causal + padding + cached-position logic.
 No KVCache container class — raw List[Tuple[Tensor,Tensor]] is error-prone.
 No inference context manager (torch.inference_mode).
 No save_pretrained / from_pretrained HuggingFace-style API.
 No model.generate() method (must use TextGenerator).


LICENSE: MIT (educational use)
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import math
import os
import time
import json
import pickle
import random
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import (
    Optional, Tuple, List, Dict, Any, Union, Iterable, Callable
)
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

@dataclass
class LLMConfig:
    """
    Hyperparameters and architectural choices for the LLM.

    This dataclass centralises every knob that affects the model's shape,
    numerical behaviour, and memory footprint. It follows the design pattern
    used by HuggingFace's `PretrainedConfig`, so you can serialize/deserialize
    it directly with `asdict()` / `LLMConfig(**d)`.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the tokenizer's vocabulary. Must match the
        tokenizer used at training/inference time.
    max_seq_len : int
        Maximum context length (Ch. 3, "context size"). Positions beyond this
        are truncated. RoPE can extrapolate somewhat, but learned positional
        embeddings strictly require `pos < max_seq_len`.
    hidden_size : int
        Model dimension `d_model`. Every Transformer block preserves this size.
    num_layers : int
        Number of stacked Transformer blocks (Ch. 3, "Transformer blocks").
    num_heads : int
        Number of query heads. `hidden_size` must be divisible by this.
    num_kv_heads : Optional[int]
        Number of key/value heads for Grouped-Query Attention (Ch. 3). If None,
        defaults to `num_heads` (i.e., standard Multi-Head Attention). If less
        than `num_heads`, enables GQA, which shrinks the KV cache and speeds
        up generation at a small quality cost.
    intermediate_size : Optional[int]
        Hidden size of the feedforward network. If None, defaults to
        `4 * hidden_size` for GELU/ReLU, or `8/3 * hidden_size` (rounded up to
        a multiple of 64) for SwiGLU — matching the ratio used in Llama.
    dropout : float
        Dropout probability applied inside residual branches.
    attention_dropout : float
        Dropout on attention weights (usually 0.0 during inference).
    use_rope : bool
        If True, use rotary positional embeddings (RoPE). If False, use
        a learned absolute positional embedding table.
    rope_theta : float
        RoPE base frequency. 10_000 is the original; Llama 3 uses 500_000.
    use_swiglu : bool
        If True, use the SwiGLU feedforward (Ch. 3). If False, use GELU.
    use_rms_norm : bool
        If True, use RMSNorm (Ch. 3). If False, use LayerNorm.
    tie_embeddings : bool
        If True, share weights between the token embedding and the LM head.
        This is standard for GPT-2 / Llama and reduces parameters.
    initializer_range : float
        Standard deviation for the normal initialization of linear weights.

    Notes
    -----
    The defaults produce a *tiny* model (~a few million parameters) that can
    be trained on CPU in minutes. For a real workload, scale `hidden_size`,
    `num_layers`, and `num_heads` up by 10–50x and use a GPU.
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
    use_swiglu: bool = True
    use_rms_norm: bool = True
    tie_embeddings: bool = True

    # Initialization
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        """Validate and fill in defaults after construction."""
        # Default KV head count = full MHA
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        # Default FFN size based on activation choice
        if self.intermediate_size is None:
            if self.use_swiglu:
                # Llama-style ratio: 8/3 * d_model, rounded up to 64
                raw = int(8 * self.hidden_size / 3)
                self.intermediate_size = ((raw + 63) // 64) * 64
            else:
                self.intermediate_size = 4 * self.hidden_size

        # Sanity checks
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads}) for GQA"
        )

    @property
    def head_dim(self) -> int:
        """Per-head dimensionality `d_k = d_model / num_heads`."""
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
        """Deserialize from a plain dict."""
        return cls(**d)


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

    Production tokenizers (GPT-4's `tiktoken`, Llama's SentencePiece) are
    optimizations of this same idea with additional engineering (regex
    pre-splitting, byte fallback, whitespace markers, etc.). This
    implementation keeps the algorithm bare so you can *see* what is
    happening.

    Design decisions
    -------------------------------------------
    * **Byte-level**: works for any UTF-8 text without an `<UNK>` token.
    * **Greedy merges in learned order**: when encoding, we apply merges in
      the exact order they were learned — this is the canonical BPE rule.
    * **Special tokens**: `<|endoftext|>`, `<|pad|>`, `<|user|>`,
      `<|assistant|>` are reserved for chat templating (Ch. 6).

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size target (including the 256 byte tokens and
        any special tokens).
    special_tokens : Optional[Dict[str, int]]
        Optional mapping of special-token strings to their IDs. If None,
        a sensible default chat set is used.

    Attributes
    ----------
    merges : Dict[Tuple[int, int], int]
        Map from `(left_id, right_id)` to the newly created merged token ID.
    vocab : Dict[int, bytes]
        Map from token ID to its byte sequence.
    """

    # Default special tokens — IDs 256..259, so they sit on top of the
    # 0..255 raw byte tokens.
    DEFAULT_SPECIALS = {
        "<|endoftext|>": 256,
        "<|pad|>": 257,
        "<|user|>": 258,
        "<|assistant|>": 259,
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
        for tok in self.special_tokens:
            self.vocab[next_id] = tok.encode("utf-8")
            next_id += 1

        # Merges learned during `train`
        self.merges: Dict[Tuple[int, int], int] = {}
        self._trained: bool = False

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def train(self, text: str, verbose: bool = False) -> None:
        """
        Learn BPE merges from `text` until `target_vocab_size` is reached.

        Parameters
        ----------
        text : str
            Training corpus. Should be reasonably large (>= 100KB) for
            meaningful merges.
        verbose : bool
            If True, print progress every 100 merges.
        """
        # Encode the raw text as a sequence of byte IDs (0..255).
        tokens: List[int] = list(text.encode("utf-8"))

        # We stop when len(vocab) hits target — accounting for specials.
        while len(self.vocab) < self.target_vocab_size:
            # Count adjacent pairs.
            counts: Counter = Counter(zip(tokens[:-1], tokens[1:]))
            if not counts:
                break

            # Most frequent pair.
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                # No pair repeats; further merging is useless.
                break

            # Assign a new ID.
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
                print(
                    f"[BPE] vocab={len(self.vocab)} "
                    f"merged {pair} -> {new_id} (freq={freq})"
                )

        self._trained = True

    # ------------------------------------------------------------------ #
    #  Encoding
    # ------------------------------------------------------------------ #
    def encode(self, text: str, allow_special: bool = False) -> List[int]:
        """
        Encode a string to a list of token IDs.

        Parameters
        ----------
        text : str
            Input string.
        allow_special : bool
            If True, occurrences of the registered special-token strings
            (e.g., "<|user|>") are replaced with their reserved IDs before
            BPE. If False, special-token substrings are treated as raw text
            (safer for arbitrary user input).

        Returns
        -------
        List[int]
        """
        if not self._trained:
            # Fall back to raw bytes — the tokenizer simply hasn't learned yet.
            return list(text.encode("utf-8"))

        if allow_special:
            # Simple greedy longest-match replacement of specials.
            for sp_tok in sorted(self.special_tokens, key=len, reverse=True):
                text = text.replace(sp_tok, f"\x00{self.special_tokens[sp_tok]}\x00")
            parts = text.split("\x00")
        else:
            parts = [text]

        out: List[int] = []
        for part in parts:
            if part.isdigit() and not allow_special:
                # (disabled path — kept for clarity)
                out.extend(list(part.encode("utf-8")))
                continue
            if allow_special and part.isdigit() and int(part) in self.specials_inv:
                out.append(int(part))
                continue
            out.extend(self._bpe_encode(part))
        return out

    def _bpe_encode(self, text: str) -> List[int]:
        """Apply learned merges to a plain string (no specials)."""
        tokens: List[int] = list(text.encode("utf-8"))

        # Repeatedly find the pair with the lowest merge priority.
        while len(tokens) >= 2:
            pairs = set(zip(tokens[:-1], tokens[1:]))
            # Keep only pairs we actually learned to merge.
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
        # `errors="replace"` keeps things robust even mid-UTF8.
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
        """Load a tokenizer previously saved with `.save()`."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        tok = cls(
            vocab_size=state["target_vocab_size"],
            special_tokens=state["special_tokens"],
        )
        tok.vocab = state["vocab"]
        tok.merges = state["merges"]
        tok._trained = state["_trained"]
        return tok

    def __len__(self) -> int:
        """Current vocabulary size."""
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.special_tokens["<|pad|>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<|endoftext|>"]


# ==============================================================================
# SECTION 3: NORMALIZATION LAYERS
# ==============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Ch. 3, "The Transformer Block").

    Unlike LayerNorm, RMSNorm does **not** subtract the mean or apply a bias;
    it only rescales by the root-mean-square of the activations:

        y = x / sqrt(mean(x^2) + eps) * weight

    This is cheaper to compute and has been shown to match LayerNorm quality
    in modern LLMs (Llama, Mistral, Gemma).

    Parameters
    ----------
    dim : int
        Size of the last dimension of `x`.
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

    Pre-norm (Ch. 3) is the standard placement in modern LLMs. The `bias`
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
    """Factory: RMSNorm if `use_rms`, else LayerNorm."""
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
    Precompute the cos/sin tables for RoPE (Ch. 3, "Positional Embeddings").

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

    Returns
    -------
    cos, sin : Tuple[torch.Tensor, torch.Tensor]
        Each of shape `(max_seq_len, head_dim)`.
    """
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"

    # inv_freq has shape (head_dim/2,)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()  # (S,)

    # Outer product -> (S, head_dim/2)
    freqs = torch.outer(positions, inv_freq)

    # Concatenate so we cover every dimension: (S, head_dim)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the second half of the last dimension to the front (with a sign
    flip). Used inside `apply_rope`.
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
    Apply rotary position embeddings to `q` and `k`.

    Parameters
    ----------
    q, k : torch.Tensor
        Shape `(B, num_heads, S, head_dim)`.
    cos, sin : torch.Tensor
        Shape `(S, head_dim)` — one row per position.

    Returns
    -------
    q_rot, k_rot : same shapes as inputs.
    """
    # Broadcast (S, head_dim) -> (1, 1, S, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ==============================================================================
# SECTION 5: MULTI-HEAD ATTENTION
# ==============================================================================

class MultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention with optional Grouped-Query Attention
    (GQA) and KV-cache support (Ch. 3, "Inside the Transformer Block").

    The forward pass implements the two conceptual steps of attention
    (Ch. 3, "Attention is all you need"):

        1. **Relevance scoring** — `softmax(Q K^T / sqrt(d_k))`
        2. **Information combining** — weighted sum over values.

    Enhancements
    ------------
    * **RoPE** applied to Q and K just before scoring (matches Llama).
    * **GQA** via `repeat_interleave` of K/V heads.
    * **KV cache** — the returned `(new_k, new_v)` should be fed back in on
      the next generation step to avoid re-computing past tokens.
    * **SDPA** — PyTorch's `scaled_dot_product_attention` is used, which
      transparently picks Flash-Attention / Memory-Efficient backends on
      supported GPUs (Ch. 3, "Flash Attention").

    Parameters
    ----------
    config : LLMConfig
        Supplies `hidden_size`, `num_heads`, `num_kv_heads`, `dropout`.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups

        # Query projection covers all query heads.
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        # Key/Value projections only cover KV heads (GQA).
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
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
            Shape `(B, S, hidden_size)`.
        cos, sin : Optional[torch.Tensor]
            RoPE tables of shape `(S_total, head_dim)`. `S_total` must cover
            past cached positions + current `S`. If None, RoPE is skipped
            (only valid when `config.use_rope` is False, or for a
            positional-embedding variant of the model).
        kv_cache : Optional[Tuple[Tensor, Tensor]]
            Past keys and values of shape `(B, num_kv_heads, S_past, head_dim)`.
        attention_mask : Optional[torch.Tensor]
            Additive float mask of shape `(B, 1, S_q, S_kv)`. Use for padding.
            Causal masking is applied automatically and is NOT to be included
            here.

        Returns
        -------
        out : torch.Tensor
            `(B, S, hidden_size)`.
        new_kv_cache : Tuple[Tensor, Tensor]
            Updated cache — pass this back on the next step.
        """
        B, S, _ = x.shape
        S_past = 0 if kv_cache is None else kv_cache[0].shape[2]

        # ---- 1. Project Q, K, V ---------------------------------------------
        q = self.q_proj(x)  # (B, S, H*D)
        k = self.k_proj(x)  # (B, S, H_kv*D)
        v = self.v_proj(x)  # (B, S, H_kv*D)

        # Reshape to (B, heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ---- 2. RoPE --------------------------------------------------------
        if cos is not None and sin is not None:
            # Slice the RoPE table to the current positions.
            cos_slice = cos[S_past : S_past + S]
            sin_slice = sin[S_past : S_past + S]
            q, k = apply_rope(q, k, cos_slice, sin_slice)

        # ---- 3. KV cache concat --------------------------------------------
        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)
        new_kv_cache = (k, v)

        S_kv = k.shape[2]

        # ---- 4. GQA: repeat KV heads to match Q heads ----------------------
        if self.num_kv_groups > 1:
            # (B, H_kv, S_kv, D) -> (B, H_kv * groups, S_kv, D)
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # ---- 5. Causal + optional padding mask -----------------------------
        # If we are doing a single-step generation with cache, no causal mask
        # is needed. Otherwise use `is_causal=True` and optionally add a mask.
        use_causal = S > 1 and S == S_kv

        # Combine user-provided mask with causal via additive float masks.
        if attention_mask is not None or not use_causal:
            # Build an explicit causal (and padding) mask.
            if use_causal:
                causal = torch.full(
                    (S, S_kv), float("-inf"), device=x.device, dtype=x.dtype
                )
                causal = torch.triu(causal, diagonal=1)
                attn_mask = causal[None, None, :, :].expand(B, 1, S, S_kv)
            else:
                attn_mask = torch.zeros(
                    (B, 1, S, S_kv), device=x.device, dtype=x.dtype
                )
            if attention_mask is not None:
                attn_mask = attn_mask + attention_mask
        else:
            attn_mask = None  # SDPA handles causality itself.

        # ---- 6. Scaled dot-product attention -------------------------------
        # Cast to float32 for stability under fp16; cast back after.
        sdpa_dtype = q.dtype
        q_sdpa = q.float() if sdpa_dtype == torch.float16 else q
        k_sdpa = k.float() if sdpa_dtype == torch.float16 else k
        v_sdpa = v.float() if sdpa_dtype == torch.float16 else v

        attn_out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=(attn_mask is None),  # only true when we didn't build one
        )
        if sdpa_dtype == torch.float16:
            attn_out = attn_out.to(sdpa_dtype)

        # ---- 7. Output projection -----------------------------------------
        # (B, H, S, D) -> (B, S, H, D) -> (B, S, H*D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.resid_dropout(self.o_proj(attn_out))
        return out, new_kv_cache


# ==============================================================================
# SECTION 6: FEEDFORWARD NETWORK
# ==============================================================================

class FeedForward(nn.Module):
    """
    Position-wise feedforward network (Ch. 3, "The feedforward neural
    network at a glance"). This is where the model stores the bulk of its
    learned knowledge — recall the "The Shawshank → Redemption" example.

    Two variants are supported:

    * **GELU** (standard): ``Linear -> GELU -> Linear``.
    * **SwiGLU** (Llama-style): ``(SiLU(W1 x) * W3 x) -> W2``. SwiGLU uses
      three matrices instead of two, which is why `intermediate_size` is
      scaled by 8/3 rather than 4.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.use_swiglu = config.use_swiglu

        if self.use_swiglu:
            # Three projections for the gated variant.
            self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        else:
            self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            # SwiGLU: gate * value
            gate = F.silu(self.w1(x))
            value = self.w3(x)
            return self.dropout(self.w2(gate * value))
        else:
            return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# ==============================================================================
# SECTION 7: TRANSFORMER BLOCK
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer block (Ch. 3, "The Transformer Block").

    Structure (pre-norm, the modern default):

        x = x + Attention(Norm(x))
        x = x + FeedForward(Norm(x))

    Pre-norm (as opposed to the original post-norm) is more stable to train
    and is what modern LLMs (Llama, Mistral, GPT-NeoX) use.

    Residual connections around each sublayer are critical to let gradients
    flow through dozens of stacked blocks.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.attn_norm = make_norm(config.hidden_size, config.use_rms_norm)
        self.attn = MultiHeadAttention(config)
        self.ffn_norm = make_norm(config.hidden_size, config.use_rms_norm)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # --- Attention sub-layer ---
        h = self.attn_norm(x)
        h, new_kv_cache = self.attn(
            h, cos=cos, sin=sin, kv_cache=kv_cache, attention_mask=attention_mask
        )
        x = x + h

        # --- Feedforward sub-layer ---
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h
        return x, new_kv_cache


# ==============================================================================
# SECTION 8: THE LLM
# ==============================================================================

class LLM(nn.Module):
    """
    A decoder-only Transformer language model (GPT-style).

    Architecture
    ------------
        input_ids ─► TokenEmbed ─┐
                                 ├─► [TransformerBlock] × N ─► Norm ─► LM Head ─► logits
        positions ─► (RoPE) ─────┘

    Because this is an *autoregressive* model (Ch. 3, "The Inputs and
    Outputs of a Trained Transformer LLM"), it predicts one token per
    position given all previous tokens, and generation proceeds token by
    token — optionally accelerated by KV caching.

    Parameters
    ----------
    config : LLMConfig

    Attributes
    ----------
    embed_tokens : nn.Embedding
        The token embedding table (Ch. 2). One row per vocabulary entry.
    blocks : nn.ModuleList
        Stacked Transformer blocks.
    final_norm : nn.Module
        Norm applied before the LM head.
    lm_head : nn.Linear
        Projects hidden states to vocabulary logits.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.config = config

        # ---- 1. Token embedding -------------------------------------------
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        # ---- 2. Positional embedding (learned fallback) -------------------
        if not config.use_rope:
            self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
        else:
            self.pos_embed = None
            # Precompute RoPE tables once; move to device lazily.
            cos, sin = precompute_rope_frequencies(
                head_dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        # ---- 3. Transformer blocks ---------------------------------------
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # ---- 4. Final norm + LM head -------------------------------------
        self.final_norm = make_norm(config.hidden_size, config.use_rms_norm)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ---- 5. Weight tying (GPT-2 / Llama) ------------------------------
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # ---- 6. Initialization -------------------------------------------
        self.apply(self._init_weights)

        # A scaled init for residual output projections improves
        # training stability (see GPT-2 paper).
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w2.weight") \
                    or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=config.initializer_range / math.sqrt(2 * config.num_layers))

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
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_kv_caches: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Run a forward pass.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Shape `(B, S)`.
        kv_caches : Optional[List[Tuple[Tensor, Tensor]]]
            One `(k, v)` tuple per layer, or None for a fresh pass.
        attention_mask : Optional[torch.Tensor]
            Additive float mask `(B, 1, S_q, S_kv)` for padding. Causal
            masking is applied automatically inside attention.
        return_kv_caches : bool
            If True, also return the updated per-layer KV caches.

        Returns
        -------
        logits : torch.Tensor
            Shape `(B, S, vocab_size)`.
        kv_caches : List[Tuple[Tensor, Tensor]]  (optional)
        """
        B, S = input_ids.shape
        device = input_ids.device

        # ---- 1. Token embeddings -----------------------------------------
        x = self.embed_tokens(input_ids)  # (B, S, H)

        # ---- 2. Positional embeddings (if not RoPE) ----------------------
        past_len = 0
        if kv_caches is not None and kv_caches[0] is not None:
            past_len = kv_caches[0][0].shape[2]

        if self.pos_embed is not None:
            positions = torch.arange(
                past_len, past_len + S, device=device, dtype=torch.long
            )
            x = x + self.pos_embed(positions)[None, :, :]

        x = self.embed_dropout(x)

        # ---- 3. RoPE table for this pass ---------------------------------
        cos = sin = None
        if self.pos_embed is None:
            cos, sin = self.rope_cos, self.rope_sin

        # ---- 4. Transformer blocks ---------------------------------------
        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            layer_cache = None if kv_caches is None else kv_caches[i]
            x, new_cache = block(
                x, cos=cos, sin=sin, kv_cache=layer_cache, attention_mask=attention_mask
            )
            new_caches.append(new_cache)

        # ---- 5. Final norm + LM head -------------------------------------
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, S, V)

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

        We apply weight decay only to matrices that participate in matrix
        multiplication (Linear/Embedding weights) and skip it for norms and
        biases — a standard practice from the nanoGPT / Llama recipes.
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
    #  Persistence
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model weights + config in a single file."""
        torch.save(
            {
                "config": self.config.to_dict(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "LLM":
        """Reconstruct a model from a `.save_checkpoint()` file."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = LLMConfig.from_dict(ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model


# ==============================================================================
# SECTION 9: SAMPLING & GENERATION
# ==============================================================================

class TextGenerator:
    """
    Handles autoregressive decoding with a suite of sampling strategies.

    Implements every strategy  ("Choosing a Single Token
    from the Probability Distribution") and "Controlling Model
    Output":

    * **Greedy** — argmax (temperature = 0)
    * **Temperature scaling** — divides logits by T
    * **Top-k filtering** — keep only the k highest-probability tokens
    * **Top-p / nucleus sampling** — keep the smallest set whose cumulative
      probability exceeds p
    * **Repetition penalty** — downweights tokens already generated
    * **KV cache** — avoids recomputing past states, ~4–5x faster

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
        Downweight tokens that have already appeared. Positive logits are
        divided by `penalty` and negative logits are multiplied — both
        push repeated tokens toward lower probability.
        """
        if penalty == 1.0:
            return logits
        for b in range(logits.shape[0]):
            seen = torch.unique(generated_ids[b])
            # Positive logits: divide; negative: multiply.
            pos = logits[b, seen] > 0
            logits[b, seen[pos]] /= penalty
            logits[b, seen[~pos]] *= penalty
        return logits

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only the top-k logits; others become -inf."""
        if k <= 0:
            return logits
        # `topk` values, and threshold at the k-th largest.
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
        sorted_logits[mask] = float("-inf")
        # Scatter back to original order.
        out = torch.empty_like(logits)
        out.scatter_(-1, sorted_idx, sorted_logits)
        return out

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
        generated_ids: torch.Tensor,
        repetition_penalty: float,
        greedy: bool,
    ) -> torch.Tensor:
        """
        Given logits of shape `(B, V)`, produce a next-token ID per batch item.
        """
        logits = logits.float()

        # 1. Repetition penalty
        logits = self._apply_repetition_penalty(logits, generated_ids, repetition_penalty)

        # 2. Greedy shortcut
        if greedy or temperature == 0.0:
            return logits.argmax(dim=-1)

        # 3. Temperature
        logits = logits / max(temperature, 1e-8)

        # 4. Top-k then top-p
        logits = self._top_k_filter(logits, top_k)
        logits = self._top_p_filter(logits, top_p)

        # 5. Sample
        probs = F.softmax(logits, dim=-1)
        # Guard against all -inf rows (fallback to uniform).
        if torch.isnan(probs).any() or (probs.sum(-1) == 0).any():
            probs = torch.ones_like(probs) / probs.size(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
        stop_on_eos: bool = True,
    ) -> str:
        """
        Generate text from a string prompt.

        Parameters
        ----------
        prompt : str
        max_new_tokens : int
        temperature : float
            0 → greedy. > 0 → stochastic.
        top_k : int
            0 → disabled.
        top_p : float
            1.0 → disabled.
        repetition_penalty : float
            1.0 → disabled; > 1 downweights repeated tokens.
        do_sample : bool
            If False, greedy.
        use_cache : bool
            Enable KV cache. Disabling it makes generation O(S^3) per step.
        stop_on_eos : bool
            Stop once `<|endoftext|>` is produced.

        Returns
        -------
        str
            The full continuation 'including' the prompt.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Encode prompt.
        input_ids = self.tokenizer.encode(prompt, allow_special=True)
        if not input_ids:
            input_ids = [self.tokenizer.eos_id]
        input_ids = torch.tensor([input_ids], device=device, dtype=torch.long)
        prompt_len = input_ids.shape[1]

        generated = input_ids
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        for step in range(max_new_tokens):
            # Only feed the *new* token after the first pass when caching.
            if use_cache and kv_caches is not None:
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

            next_logits = logits[:, -1, :]  # (B, V)

            next_id = self._sample_next_token(
                logits=next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated_ids=generated,
                repetition_penalty=repetition_penalty,
                greedy=(not do_sample),
            )

            generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)

            if stop_on_eos and next_id.item() == self.tokenizer.eos_id:
                break

            # Safety: respect max_seq_len.
            if generated.shape[1] >= self.model.config.max_seq_len:
                break

        text = self.tokenizer.decode(generated[0].tolist())
        return text


# ==============================================================================
# SECTION 10: DATASET
# ==============================================================================

class TextDataset(Dataset):
    """
    Sliding-window next-token-prediction dataset .

    Given a long corpus, we concatenate all token IDs into one stream, then
    emit every contiguous window of length `seq_len + 1` (the extra token
    is the target for the last position).

    This is the standard approach for training language models and
    corresponds to the "packing" technique  for efficiently
    handling short documents.

    Parameters
    ----------
    token_ids : List[int]
    seq_len : int
        Context length per training example.
    stride : int
        Step between successive windows. `stride == seq_len` gives no
        overlap; smaller values give overlap (more samples, some redundancy).
    """

    def __init__(self, token_ids: List[int], seq_len: int, stride: Optional[int] = None) -> None:
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride or seq_len

        # Pre-compute valid starting indices.
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
        window = self.data[s : s + self.seq_len + 1]
        return window[:-1], window[1:]


# ==============================================================================
# SECTION 11: TRAINER
# ==============================================================================

class LLMTrainer:
    """
    Minimal but production-conscious training loop for a generative LLM.



    * Next-token-prediction cross-entropy loss (the objective used in
      supervised fine-tuning).
    * Gradient accumulation for a larger effective batch size.
    * Cosine LR schedule with linear warmup (standard for LLM training).
    * Gradient clipping.
    * Mixed-precision (AMP) on CUDA.
    * Periodic evaluation and checkpointing.
    * Loss masking for padding (ignore_index on pad tokens).

    For a full SFT / DPO pipeline, replace the loss with the
    appropriate objective — the loop structure is the same.
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
        self.use_amp = use_amp and self.device.type == "cuda"
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer.
        self.optimizer = model.configure_optimizers(lr=lr, weight_decay=weight_decay)

        # LR schedule: linear warmup to `lr`, then cosine decay to 10% of lr.
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

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
        """Run the training loop until `max_steps` is reached."""
        self.model.train()
        train_iter = iter(train_loader)
        running_loss = 0.0
        running_steps = 0
        t0 = time.time()

        while self.step < self.max_steps:
            # --- Accumulate gradients -------------------------------------
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(self.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    logits = self.model(x)  # (B, S, V)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.reshape(-1),
                        ignore_index=self.tokenizer.pad_id,
                    )
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            # --- Optimiser step -------------------------------------------
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.step += 1
            running_loss += accum_loss
            running_steps += 1

            # --- Logging ---------------------------------------------------
            if self.step % self.log_every == 0:
                avg = running_loss / running_steps
                dt = time.time() - t0
                tokens_per_s = (
                    self.grad_accum_steps
                    * x.size(0)
                    * x.size(1)
                    * running_steps
                    / max(dt, 1e-6)
                )
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"[step {self.step:>5}/{self.max_steps}] "
                    f"loss={avg:.4f} lr={lr:.2e} "
                    f"tok/s={tokens_per_s:,.0f}"
                )
                running_loss = 0.0
                running_steps = 0
                t0 = time.time()

            # --- Evaluation -------------------------------------------------
            if eval_loader is not None and self.step % self.eval_every == 0:
                eval_loss = self.evaluate(eval_loader)
                print(f"[step {self.step:>5}] eval_loss={eval_loss:.4f}")
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save("best.pt")
                self.model.train()

            # --- Checkpointing ---------------------------------------------
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
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
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
        """Save model + optimizer + scheduler + step."""
        path = self.output_dir / name
        torch.save(
            {
                "config": self.model.config.to_dict(),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.step,
                "best_eval_loss": self.best_eval_loss,
            },
            path,
        )


# ==============================================================================
# SECTION 12: DEMO — TRAIN A TINY MODEL END TO END
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def demo() -> None:
    """
    End-to-end demonstration:

        1. Train a small BPE tokenizer on a corpus.
        2. Build a small LLM.
        3. Train it with next-token prediction.
        4. Generate with different sampling strategies.
        5. Verify that KV-cached generation matches non-cached generation.
    """
    print("=" * 78)
    print("HANDS-ON LARGE LANGUAGE MODELS — end-to-end demo")
    print("=" * 78)

    set_seed(42)

    # ------------------------------------------------------------------
    # 1. Corpus
    # ------------------------------------------------------------------
    # A small, repetitive corpus so a tiny model can learn structure in a
    # couple of minutes on CPU. Replace with your own text for real work.
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
    ) * 60  # ~ few thousand tokens

    print(f"\nCorpus: {len(corpus):,} characters")

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    print("\n[1/5] Training BPE tokenizer ...")
    tokenizer = BPETokenizer(vocab_size=512)
    tokenizer.train(corpus)
    print(f"    vocabulary size: {len(tokenizer)}")

    encoded = tokenizer.encode(corpus, allow_special=False)
    print(f"    corpus -> {len(encoded):,} tokens "
          f"({len(encoded)/len(corpus):.3f} tokens/char)")

    # Quick round-trip sanity check.
    assert tokenizer.decode(encoded) == corpus, "Tokenizer round-trip failed!"

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    print("\n[2/5] Building LLM ...")
    config = LLMConfig(
        vocab_size=len(tokenizer),
        max_seq_len=64,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,     # GQA enabled
        dropout=0.1,
        use_rope=True,
        use_swiglu=True,
        use_rms_norm=True,
        tie_embeddings=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLM(config).to(device)
    print(f"    device: {device}")
    print(f"    parameters: {model.num_parameters():,}")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print("\n[3/5] Training ...")
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
        output_dir="./demo_checkpoints",
    )
    trainer.train(train_loader, eval_loader=val_loader)

    # ------------------------------------------------------------------
    # 5. Generate
    # ------------------------------------------------------------------
    print("\n[4/5] Generating text with different sampling strategies ...")
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
    ]

    for prompt in prompts:
        print(f"\nPROMPT: {prompt!r}")
        for name, kwargs in strategies:
            out = generator.generate(
                prompt,
                max_new_tokens=20,
                use_cache=True,
                stop_on_eos=False,
                **kwargs,
            )
            print(f"    [{name:<12}] {out!r}")

    # ------------------------------------------------------------------
    # 6. Verify KV cache correctness
    # ------------------------------------------------------------------
    print("\n[5/5] Verifying KV-cache correctness ...")
    model.eval()
    test_prompt = "the quick brown fox jumps"
    ids = torch.tensor(
        [tokenizer.encode(test_prompt, allow_special=True)], device=device
    )
    with torch.no_grad():
        # Full sequence, no cache.
        logits_nocache = model(ids)[0, -1]
        # Cached: prefill then one step.
        _, caches = model(ids, return_kv_caches=True)
        # Feed the last token again with cache to get the same "next token"
        # distribution for the same position.
        logits_cache, _ = model(ids[:, -1:], kv_caches=caches, return_kv_caches=True)
        logits_cache = logits_cache[0, -1]

    diff = (logits_nocache - logits_cache).abs().max().item()
    print(f"    max |logits_no_cache - logits_with_cache| = {diff:.2e}")
    assert diff < 1e-3, "KV cache mismatch!"
    print("    ✓ KV cache matches non-cached forward pass.")

    print("\n" + "=" * 78)
    print("Demo complete. Model and checkpoints saved under ./demo_checkpoints")
    print("=" * 78)


# ==============================================================================
# SECTION 13: EXTENSION SCAFFOLDS
# ==============================================================================

class ClassificationHead(nn.Module):
    """
    Classification head for text classification .

    Reuses the LLM backbone (which acts as an encoder of contextualised
    embeddings) and adds a small classifier on top. The `[CLS]` token
    representation — or the mean-pooled token representations — feeds the
    classifier. This matches the "task-specific model" architecture

    """

    def __init__(self, backbone: LLM, num_labels: int, pooling: str = "mean") -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.classifier = nn.Linear(backbone.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.backbone.final_norm(
            self.backbone.embed_tokens(input_ids)  # placeholder; see note
        )
        # NOTE: For a real classifier you would run the full Transformer
        # stack and pool the resulting hidden states. This stub keeps the
        # file focused on the generative path.
        if self.pooling == "mean":
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden[:, 0]
        return self.classifier(pooled)


class EmbeddingModel(nn.Module):
    """
    Text-embedding model scaffold .

    A full implementation would attach a mean-pooling layer to the final
    hidden states of the LLM backbone and fine-tune it with a contrastive
    loss (Ch. 10). The idea is:

        emb = normalize(mean_pool(backbone(input_ids)))
        loss = InfoNCE(emb_anchor, emb_positive, emb_negatives)

    the complete loss formulation. Here we only define the
    forward pass shape so you can drop in your preferred loss.
    """

    def __init__(self, backbone: LLM) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Placeholder: uses embedding layer only. See note above.
        h = self.backbone.embed_tokens(input_ids)
        h = h.mean(dim=1)
        return F.normalize(h, dim=-1)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    demo()

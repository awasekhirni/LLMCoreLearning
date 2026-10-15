"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.
llm_internals.py
=========================
NumPy implementation of the core Transformer architecture used
in modern Large Language Models (Llama / Phi / Mistral family).

Learning goals (mapped to the technical card):
    step_2_embedding          -> class TokenEmbedding
    step_3_transformer_stack  -> class TransformerBlock (with sub-modules)
    step_4_lm_head            -> class LMHead
    step_5_sampling           -> class Sampler
    step_6_autoregressive_loop -> class MiniLLM.generate()

Transformer block anatomy:
    RMSNorm        -> class RMSNorm
    GQA + RoPE     -> class GroupedQueryAttention + class RoPE
    SwiGLU FFN     -> class SwiGLUFFN
    Residuals      -> inside TransformerBlock.forward

Key optimizations:
    KV Cache       -> field on GroupedQueryAttention
    GQA            -> num_kv_heads < num_heads
    RoPE           -> class RoPE
    Flash Attn     -> simulated by chunked loop in _flash_attention_sim()

Supplementary concepts (from technical card):
    Softmax, Linear Projections, Causal Masking,
    RMSNorm, SwiGLU — all implemented in plain NumPy.

Design principles:
    - DRY: every reusable op (softmax, matmul, RMSNorm) is a class.
    - Each class has __init__ + forward(...) (a callable callable).
    - Input validation raises ValueError with actionable messages.
    - Random state controlled by a single seeded RNG passed in __init__.


# 1. Install packages
pip install numpy pandas matplotlib scikit-learn
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. GLOBAL CONFIGURATION (single source of truth — DRY)
# ============================================================================

@dataclass
class TransformerConfig:
    """All hyper-parameters for the Mini-LLM in one place.

    The defaults are deliberately tiny so the entire forward pass fits
    on a single screen and runs in milliseconds on a laptop CPU.
    """
    vocab_size: int = 20          # toy vocabulary size
    hidden_dim: int = 32          # embedding / d_model
    num_heads: int = 4            # query heads
    num_kv_heads: int = 2         # key/value heads (GQA: < num_heads)
    ffn_hidden_dim: int = 64      # SwiGLU intermediate size
    num_layers: int = 2           # transformer block stack depth
    max_seq_len: int = 64         # RoPE precompute length
    rope_theta: float = 10000.0   # RoPE base frequency
    rms_eps: float = 1e-6         # RMSNorm numerical stability
    seed: int = 42                # reproducibility


# ============================================================================
# 2. SUPPLEMENTARY PRIMITIVES — each as its own small class
# ============================================================================

class Softmax:
    """Numerically stable softmax along the last axis.

    Used in: attention weight computation, final token probability.
    Numerical stability trick: subtract max(x) before exp() so the largest
    exponent is exp(0)=1, preventing overflow for logits up to ~700.
    """

    @staticmethod
    def forward(x: np.ndarray, axis: int = -1) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Softmax expects np.ndarray, got {type(x)}")
        if x.size == 0:
            raise ValueError("Softmax received empty array")
        x_max = np.max(x, axis=axis, keepdims=True)
        # Subtract max for stability (see class docstring).
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class LinearProjection:
    """y = x @ W.T + b  (the matrix-multiply primitive).

    Used in: Q/K/V/O projections, FFN's three projections, LM head.
    Weight shape is (out_features, in_features) — the standard PyTorch
    convention — so that y = x @ W.T is a clean batched matmul.
    """

    def __init__(self, in_features: int, out_features: int,
                 use_bias: bool, rng: np.random.Generator):
        # Kaiming-style init scaled down for toy size.
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = (rng.standard_normal((out_features, in_features)) * std).astype(np.float32)
        self.bias = (rng.standard_normal(out_features) * std).astype(np.float32) if use_bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim < 2:
            # promote 1-D vector to (1, d) so matmul works uniformly
            x = x[None, :]
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class CausalMask:
    """Builds the upper-triangular -inf mask used in autoregressive attention.

    Why -inf and not 0?  Because softmax(exp(-inf)) = 0 exactly, so future
    tokens get *zero* attention weight, guaranteeing the model cannot
    "cheat" by peeking at future tokens during training or generation.
    """

    @staticmethod
    def build(seq_len: int, neg_inf_value: float = -1e9) -> np.ndarray:
        # np.triu(k=1) gives the strict upper triangle (above the diagonal).
        # We fill that with a very negative number; diagonal and below stay 0.
        mask = np.triu(np.full((seq_len, seq_len), neg_inf_value, dtype=np.float32), k=1)
        return mask


class RMSNorm:
    """Root Mean Square Normalization (Llama-style, no mean subtraction).

    Why not LayerNorm?  RMSNorm skips the mean-centering step, saving one
    full tensor reduction per layer. Empirically matches LayerNorm quality.
    """

    def __init__(self, dim: int, eps: float = 1e-6, rng: Optional[np.random.Generator] = None):
        # Learnable scale; initialized to ones (PyTorch convention).
        self.weight = np.ones(dim, dtype=np.float32)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"RMSNorm expected last dim {self.weight.shape[0]}, got {x.shape[-1]}"
            )
        # Compute root mean square along feature axis.
        ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
        norm = 1.0 / np.sqrt(ms + self.eps)
        return (x * norm) * self.weight


class SwiGLUFFN:
    """SwiGLU feed-forward sub-layer:  y = W2 * (silu(W1 x) * (W3 x))

    Why three matrices instead of two?  The "gate" * (silu(W1 x) * W3 x)
    pattern lets the network learn to selectively pass or suppress each
    feature, improving capacity vs plain ReLU/GELU FFNs.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, rng: np.random.Generator):
        # W1 = gate projection, W3 = up projection, W2 = down projection.
        self.w1 = LinearProjection(hidden_dim, ffn_dim, use_bias=False, rng=rng)
        self.w2 = LinearProjection(ffn_dim, hidden_dim, use_bias=False, rng=rng)
        self.w3 = LinearProjection(hidden_dim, ffn_dim, use_bias=False, rng=rng)

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        # SiLU = x * sigmoid(x)  (a.k.a. Swish)
        return x * (1.0 / (1.0 + np.exp(-x)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Gating: gate path silu-activated, multiplied by the up path, then down-projected.
        gate = self._silu(self.w1.forward(x))
        up = self.w3.forward(x)
        return self.w2.forward(gate * up)


# ============================================================================
# 3. KEY OPTIMIZATIONS — RoPE, GQA (with KV Cache), "Flash" simulation
# ============================================================================

class RoPE:
    """Rotary Positional Embeddings.

    Rotates consecutive pairs of feature dimensions of Q and K by an angle
    that depends on the token's position. Because rotation is multiplicative,
    the dot product <Q_m, K_n> depends on (m - n), giving *relative* position
    encoding for free — no learned positional embedding table needed.
    """

    def __init__(self, hidden_dim: int, max_seq_len: int, theta: float = 10000.0):
        half = hidden_dim // 2
        # Inverse frequencies: theta^(-2i/d) for i = 0..d/2-1
        inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / hidden_dim))
        # Position index grid
        positions = np.arange(max_seq_len, dtype=np.float32)
        # angles[pos, i] = pos * inv_freq[i]
        angles = np.outer(positions, inv_freq)
        # Precompute sin/cos tables of shape (max_seq_len, hidden_dim)
        # We duplicate each frequency across two consecutive dims (original RoPE layout).
        self.cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
        self.sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)

    def apply(self, x: np.ndarray, seq_len_offset: int = 0) -> np.ndarray:
        """Apply rotation to x (assumed shape [..., seq_len, hidden_dim])."""
        seq_len = x.shape[-2]
        cos = self.cos[seq_len_offset:seq_len_offset + seq_len]   # (seq_len, hidden_dim)
        sin = self.sin[seq_len_offset:seq_len_offset + seq_len]
        # Build rotated x by splitting into even/odd pairs.
        x1 = x[..., 0::2]   # even indices
        x2 = x[..., 1::2]  # odd  indices
        # rotate_pair: (x1*cos - x2*sin, x1*sin + x2*cos)
        rot1 = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
        rot2 = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        # Interleave back.
        out = np.empty_like(x)
        out[..., 0::2] = rot1
        out[..., 1::2] = rot2
        return out


def _flash_attention_sim(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                         mask: np.ndarray, chunk_size: int = 16) -> np.ndarray:
    """Simulate Flash Attention by computing scores in row-chunks.

    Real Flash Attention fuses the softmax + matmul into a single kernel
    that streams through GPU SRAM. We cannot replicate that in NumPy, but
    we CAN demonstrate the *algorithmic* idea: process the key dimension
    in chunks so peak memory is O(chunk_size * seq_len) instead of O(seq_len^2).
    """
    seq_len_q, head_dim = q.shape[-2], q.shape[-1]
    seq_len_k = k.shape[-2]
    # We keep an output buffer and an online running max for stability.
    out = np.zeros_like(q)
    for i in range(0, seq_len_q, chunk_size):
        i_end = min(i + chunk_size, seq_len_q)
        qi = q[i:i_end]                         # (chunk, head_dim)
        scores = qi @ k.T                       # (chunk, seq_len_k)
        # Apply causal mask only for rows in this chunk.
        chunk_mask = mask[i:i_end]
        scores = scores + chunk_mask
        weights = Softmax.forward(scores, axis=-1)
        out[i:i_end] = weights @ v
    return out


class GroupedQueryAttention:
    """GQA layer with KV Cache.

    num_heads   : number of query heads H_q
    num_kv_heads: number of key/value heads H_kv  (H_kv <= H_q for GQA)
    head_dim    : hidden_dim // num_heads

    The KV Cache stores previously computed K and V so we only recompute
    them for *new* tokens during autoregressive generation.
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int,
                 max_seq_len: int, rope: RoPE, rng: np.random.Generator):
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.rope = rope
        self.group_size = num_heads // num_kv_heads
        # Projections. Note Q has dim=hidden, K/V have dim = num_kv_heads * head_dim.
        self.wq = LinearProjection(hidden_dim, num_heads * self.head_dim, use_bias=False, rng=rng)
        self.wk = LinearProjection(hidden_dim, num_kv_heads * self.head_dim, use_bias=False, rng=rng)
        self.wv = LinearProjection(hidden_dim, num_kv_heads * self.head_dim, use_bias=False, rng=rng)
        self.wo = LinearProjection(num_heads * self.head_dim, hidden_dim, use_bias=False, rng=rng)
        # KV Cache: lists that grow as tokens are processed.
        self._cache_k: List[np.ndarray] = []
        self._cache_v: List[np.ndarray] = []
        self._cache_len: int = 0

    def _split_heads(self, x: np.ndarray, num_heads: int) -> np.ndarray:
        """(seq_len, num_heads*head_dim) -> (num_heads, seq_len, head_dim)."""
        s = x.shape[0]
        x = x.reshape(s, num_heads, self.head_dim)
        return np.transpose(x, (1, 0, 2))

    def _repeat_kv(self, x: np.ndarray) -> np.ndarray:
        """Broadcast K/V from num_kv_heads to num_heads (GQA broadcast)."""
        # x: (num_kv_heads, seq_len, head_dim)
        if self.group_size == 1:
            return x
        return np.repeat(x, self.group_size, axis=0)

    def reset_cache(self) -> None:
        self._cache_k.clear()
        self._cache_v.clear()
        self._cache_len = 0

    def forward(self, x: np.ndarray, use_cache: bool = False,
                use_flash: bool = False) -> np.ndarray:
        """
        x: (seq_len, hidden_dim)
        returns: (seq_len, hidden_dim)
        """
        if x.ndim != 2 or x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"GQA expects (seq_len, {self.hidden_dim}), got {x.shape}"
            )
        seq_len = x.shape[0]
        # 1) Project to Q, K, V.
        q = self.wq.forward(x)                  # (seq_len, num_heads*head_dim)
        k = self.wk.forward(x)
        v = self.wv.forward(x)
        # 2) Split into heads.
        q = self._split_heads(q, self.num_heads)         # (H_q, seq_len, head_dim)
        k = self._split_heads(k, self.num_kv_heads)      # (H_kv, seq_len, head_dim)
        v = self._split_heads(v, self.num_kv_heads)
        # 3) Apply RoPE to Q and K (NOT to V — V is content, not position).
        offset = self._cache_len if use_cache else 0
        q = self.rope.apply(q, seq_len_offset=offset)
        k = self.rope.apply(k, seq_len_offset=offset)
        # 4) KV Cache: concatenate previous K/V with new ones.
        if use_cache and len(self._cache_k) > 0:
            prev_k = np.concatenate(self._cache_k, axis=1)   # (H_kv, prev_len, head_dim)
            prev_v = np.concatenate(self._cache_v, axis=1)
            k_full = np.concatenate([prev_k, k], axis=1)
            v_full = np.concatenate([prev_v, v], axis=1)
        else:
            k_full = k
            v_full = v
        # Update cache.
        if use_cache:
            self._cache_k.append(k)
            self._cache_v.append(v)
            self._cache_len += seq_len
        # 5) GQA: broadcast K/V from num_kv_heads up to num_heads.
        k_full_b = self._repeat_kv(k_full)     # (H_q, total_len, head_dim)
        v_full_b = self._repeat_kv(v_full)
        # 6) Attention per head.
        total_len = k_full_b.shape[1]
        # Causal mask must cover (seq_len, total_len) when caching is on.
        # Rows correspond to new query positions (offset..offset+seq_len).
        mask_full = CausalMask.build(seq_len)  # (seq_len, seq_len) base
        if use_cache and offset > 0:
            # Pad left with zeros so cached keys can always be attended to.
            pad_left = np.zeros((seq_len, offset), dtype=np.float32)
            mask_full = np.concatenate([pad_left, mask_full], axis=1)
        # Now slice to (seq_len, total_len).
        mask_full = mask_full[:, :total_len]
        outputs = []
        for h in range(self.num_heads):
            qh = q[h]            # (seq_len, head_dim)
            kh = k_full_b[h]     # (total_len, head_dim)
            vh = v_full_b[h]     # (total_len, head_dim)
            if use_flash:
                oh = _flash_attention_sim(qh[None], kh[None], vh[None], mask_full[None])[0]
            else:
                scores = qh @ kh.T                  # (seq_len, total_len)
                scores = scores + mask_full
                weights = Softmax.forward(scores, axis=-1)
                oh = weights @ vh
            outputs.append(oh)
        out = np.stack(outputs, axis=1)         # (seq_len, num_heads, head_dim)
        out = out.reshape(seq_len, self.num_heads * self.head_dim)
        return self.wo.forward(out)


# ============================================================================
# 4. TRANSFORMER BLOCK + EMBEDDING + LM HEAD + SAMPLER + MINI-LLM
# ============================================================================

class TokenEmbedding:
    """Lookup table: token_id -> dense vector of size hidden_dim."""

    def __init__(self, vocab_size: int, hidden_dim: int, rng: np.random.Generator):
        # Normal(0, 1) init; scaled down by sqrt(hidden_dim) per "Attention is All You Need".
        self.table = (rng.standard_normal((vocab_size, hidden_dim)) / np.sqrt(hidden_dim)
                      ).astype(np.float32)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        if token_ids.ndim != 1:
            raise ValueError(f"Embedding expects 1-D token IDs, got shape {token_ids.shape}")
        if token_ids.size == 0:
            raise ValueError("Embedding received empty token sequence")
        if (token_ids < 0).any() or (token_ids >= self.table.shape[0]).any():
            raise ValueError(
                f"Token IDs must be in [0, {self.table.shape[0]}); got min={token_ids.min()}, "
                f"max={token_ids.max()}"
            )
        return self.table[token_ids]


class TransformerBlock:
    """One block: RMSNorm -> GQA -> residual -> RMSNorm -> SwiGLU -> residual.

    Pre-normalization (Llama-style): norm is applied BEFORE the sub-layer,
    which makes training more stable than the original post-norm Transformer.
    """

    def __init__(self, cfg: TransformerConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.hidden_dim, cfg.rms_eps)
        self.rope = RoPE(cfg.hidden_dim, cfg.max_seq_len, cfg.rope_theta)
        self.attn = GroupedQueryAttention(
            cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads,
            cfg.max_seq_len, self.rope, rng
        )
        self.norm2 = RMSNorm(cfg.hidden_dim, cfg.rms_eps)
        self.ffn = SwiGLUFFN(cfg.hidden_dim, cfg.ffn_hidden_dim, rng)

    def forward(self, x: np.ndarray, use_cache: bool = False,
                use_flash: bool = False) -> np.ndarray:
        # Pre-norm Attention + residual
        x = x + self.attn.forward(self.norm1.forward(x), use_cache=use_cache, use_flash=use_flash)
        # Pre-norm FFN + residual
        x = x + self.ffn.forward(self.norm2.forward(x))
        return x


class LMHead:
    """Final linear projection: hidden_dim -> vocab_size (produces logits)."""

    def __init__(self, hidden_dim: int, vocab_size: int, rng: np.random.Generator):
        self.proj = LinearProjection(hidden_dim, vocab_size, use_bias=False, rng=rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.proj.forward(x)


class Sampler:
    """Token sampling strategies: greedy, temperature, top-p (nucleus)."""

    @staticmethod
    def greedy(logits: np.ndarray) -> int:
        return int(np.argmax(logits))

    @staticmethod
    def temperature(logits: np.ndarray, temperature: float,
                    rng: np.random.Generator) -> int:
        if temperature <= 0:
            raise ValueError("temperature must be > 0 (use greedy() for T=0)")
        scaled = logits / temperature
        probs = Softmax.forward(scaled)
        return int(rng.choice(len(probs), p=probs))

    @staticmethod
    def top_p(logits: np.ndarray, p: float, temperature: float,
              rng: np.random.Generator) -> int:
        """Nucleus sampling: keep smallest set of tokens whose cumulative prob >= p."""
        if not 0.0 < p <= 1.0:
            raise ValueError("p must be in (0, 1]")
        scaled = logits / temperature
        probs = Softmax.forward(scaled)
        # Sort descending by probability.
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumulative = np.cumsum(sorted_probs)
        # Keep tokens until cumulative >= p (always keep at least one).
        cutoff = np.searchsorted(cumulative, p) + 1
        kept_idx = sorted_idx[:cutoff]
        kept_probs = probs[kept_idx]
        kept_probs = kept_probs / kept_probs.sum()
        return int(rng.choice(kept_idx, p=kept_probs))


class MiniLLM:
    """End-to-end mini LLM: embedding -> N transformer blocks -> LM head."""

    def __init__(self, cfg: TransformerConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.embed = TokenEmbedding(cfg.vocab_size, cfg.hidden_dim, self.rng)
        self.blocks = [TransformerBlock(cfg, self.rng) for _ in range(cfg.num_layers)]
        self.lm_head = LMHead(cfg.hidden_dim, cfg.vocab_size, self.rng)
        self.sampler = Sampler()

    def forward(self, token_ids: np.ndarray, use_cache: bool = False,
                use_flash: bool = False) -> np.ndarray:
        x = self.embed.forward(token_ids)               # (seq_len, hidden_dim)
        for block in self.blocks:
            x = block.forward(x, use_cache=use_cache, use_flash=use_flash)
        logits = self.lm_head.forward(x)                # (seq_len, vocab_size)
        return logits

    def generate(self, prompt: np.ndarray, n_new: int,
                 strategy: str = "greedy", temperature: float = 1.0,
                 top_p: float = 0.9, use_cache: bool = True) -> np.ndarray:
        """Autoregressive generation loop."""
        tokens = prompt.copy()
        # Reset caches at start of generation.
        for blk in self.blocks:
            blk.attn.reset_cache()
        # First pass: process the entire prompt (this populates the cache).
        logits = self.forward(tokens, use_cache=use_cache)
        last_logits = logits[-1]
        for _ in range(n_new):
            if strategy == "greedy":
                next_id = self.sampler.greedy(last_logits)
            elif strategy == "temperature":
                next_id = self.sampler.temperature(last_logits, temperature, self.rng)
            elif strategy == "top_p":
                next_id = self.sampler.top_p(last_logits, top_p, temperature, self.rng)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            tokens = np.append(tokens, next_id)
            # Subsequent passes: only the new token (cache holds the rest).
            new_logits = self.forward(np.array([next_id], dtype=np.int64),
                                       use_cache=use_cache)
            last_logits = new_logits[-1]
        return tokens


# ============================================================================
# 5. SCENARIOS — happy path, edge cases, failures
# ============================================================================

def scenario_1_happy_path() -> dict:
    """Basic forward pass: small prompt, check shapes & determinism."""
    cfg = TransformerConfig()
    rng = np.random.default_rng(0)
    prompt = rng.integers(0, cfg.vocab_size, size=8, dtype=np.int64)
    model = MiniLLM(cfg)
    logits = model.forward(prompt)
    return {
        "prompt": prompt.tolist(),
        "logits_shape": logits.shape,
        "logits_sum": float(logits.sum()),
        "expected_logits_shape": (8, cfg.vocab_size),
    }


def scenario_2_autoregressive_generation() -> dict:
    """Greedy generation with KV cache: deterministic across runs."""
    cfg = TransformerConfig()
    prompt = np.array([1, 4, 2, 7], dtype=np.int64)
    model = MiniLLM(cfg)
    out1 = model.generate(prompt, n_new=5, strategy="greedy", use_cache=True)
    model2 = MiniLLM(cfg)  # same seed => same weights => same output
    out2 = model2.generate(prompt, n_new=5, strategy="greedy", use_cache=True)
    return {
        "output_run1": out1.tolist(),
        "output_run2": out2.tolist(),
        "deterministic": bool(np.array_equal(out1, out2)),
    }


def scenario_3_cache_vs_no_cache() -> dict:
    """KV cache must produce identical logits to the no-cache path."""
    cfg = TransformerConfig()
    prompt = np.array([0, 5, 11, 3, 9, 2, 14, 6], dtype=np.int64)
    model = MiniLLM(cfg)
    # Without cache: full forward pass each step.
    logits_no_cache = model.forward(prompt, use_cache=False)[-1]
    # With cache: feed one at a time.
    for blk in model.blocks:
        blk.attn.reset_cache()
    for i, tok in enumerate(prompt):
        out = model.forward(np.array([tok], dtype=np.int64), use_cache=True)
    logits_cached = out[-1]
    return {
        "max_abs_diff": float(np.max(np.abs(logits_no_cache - logits_cached))),
        "tolerance": 1e-5,
        "match": bool(np.allclose(logits_no_cache, logits_cached, atol=1e-5)),
    }


def scenario_4_empty_input() -> dict:
    """Empty token sequence should raise ValueError, not silently return garbage."""
    cfg = TransformerConfig()
    model = MiniLLM(cfg)
    try:
        model.forward(np.array([], dtype=np.int64))
        return {"error": None, "status": "FAILED: should have raised"}
    except ValueError as e:
        return {"error_type": type(e).__name__, "message": str(e),
                "status": "PASSED: raised ValueError"}


def scenario_5_out_of_vocab() -> dict:
    """Token ID >= vocab_size must raise."""
    cfg = TransformerConfig()
    model = MiniLLM(cfg)
    try:
        model.forward(np.array([cfg.vocab_size + 5], dtype=np.int64))
        return {"error": None, "status": "FAILED"}
    except ValueError as e:
        return {"error_type": type(e).__name__, "message": str(e),
                "status": "PASSED"}


def scenario_6_wrong_type() -> dict:
    """Float token IDs must be rejected."""
    cfg = TransformerConfig()
    model = MiniLLM(cfg)
    try:
        model.forward(np.array([1.5, 2.0, 3.0]))  # float
        return {"error": None, "status": "FAILED"}
    except (ValueError, IndexError, TypeError) as e:
        return {"error_type": type(e).__name__, "message": str(e)[:80],
                "status": "PASSED"}


def scenario_7_extreme_logit_values() -> dict:
    """Softmax must remain numerically stable with extreme logits."""
    extreme = np.array([1e6, -1e6, 0.0, 1.0, -1.0], dtype=np.float32)
    probs = Softmax.forward(extreme)
    return {
        "max_prob": float(probs.max()),
        "min_prob": float(probs.min()),
        "sum": float(probs.sum()),
        "finite": bool(np.all(np.isfinite(probs))),
        "expected_max_prob": 1.0,
    }


def scenario_8_top_p_sampling_distribution() -> dict:
    """Top-p=1.0 should match plain temperature sampling distribution."""
    cfg = TransformerConfig()
    model = MiniLLM(cfg)
    logits = model.forward(np.array([3, 1, 2], dtype=np.int64))[-1]
    counts = np.zeros(cfg.vocab_size, dtype=np.int64)
    n_trials = 2000
    for _ in range(n_trials):
        tok = Sampler.top_p(logits, p=1.0, temperature=1.0,
                            rng=np.random.default_rng(42))
        counts[tok] += 1
    empirical = counts / n_trials
    expected = Softmax.forward(logits)
    return {
        "max_abs_diff": float(np.max(np.abs(empirical - expected))),
        "tolerance": 0.05,
        "match": bool(np.allclose(empirical, expected, atol=0.05)),
    }


def scenario_9_long_sequence_overflow() -> dict:
    """Sequence longer than max_seq_len should fail gracefully (RoPE precompute)."""
    cfg = TransformerConfig(max_seq_len=8)
    model = MiniLLM(cfg)
    long_prompt = np.arange(10, dtype=np.int64) % cfg.vocab_size
    try:
        model.forward(long_prompt)
        return {"error": None, "status": "FAILED: should have raised"}
    except (ValueError, IndexError) as e:
        return {"error_type": type(e).__name__, "message": str(e)[:80],
                "status": "PASSED"}


def scenario_10_flash_vs_naive_attention() -> dict:
    """Flash-simulation must match naive attention numerically."""
    cfg = TransformerConfig()
    model = MiniLLM(cfg)
    prompt = np.array([2, 7, 1, 9, 5], dtype=np.int64)
    naive = model.forward(prompt, use_cache=False, use_flash=False)
    flash = model.forward(prompt, use_cache=False, use_flash=True)
    return {
        "max_abs_diff": float(np.max(np.abs(naive - flash))),
        "tolerance": 1e-5,
        "match": bool(np.allclose(naive, flash, atol=1e-5)),
    }


def run_all_scenarios() -> None:
    """Pretty-print every scenario's outcome."""
    scenarios = [
        ("1. Happy path forward pass", scenario_1_happy_path),
        ("2. Autoregressive generation (greedy)", scenario_2_autoregressive_generation),
        ("3. KV cache equals no-cache", scenario_3_cache_vs_no_cache),
        ("4. Empty input rejected", scenario_4_empty_input),
        ("5. Out-of-vocab rejected", scenario_5_out_of_vocab),
        ("6. Wrong type rejected", scenario_6_wrong_type),
        ("7. Softmax numerical stability", scenario_7_extreme_logit_values),
        ("8. Top-p sampling ≈ temperature distribution", scenario_8_top_p_sampling_distribution),
        ("9. Sequence > max_seq_len rejected", scenario_9_long_sequence_overflow),
        ("10. Flash == naive attention", scenario_10_flash_vs_naive_attention),
    ]
    for name, fn in scenarios:
        print(f"\n--- {name} ---")
        result = fn()
        for k, v in result.items():
            print(f"  {k}: {v}")


# ============================================================================
# 6. MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" Looking Inside Large Language Models")
    print("=" * 70)
    run_all_scenarios()
    # Visualizations are produced by visualize.py .

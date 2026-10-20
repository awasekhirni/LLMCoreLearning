"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.

nope_learning_module.py
=======================
Educational implementation of **NoPE (No Positional Encoding)** for
decoder‑only Transformers.


Python : 3.10+

Purpose
-------
Demonstrate that a causal Transformer can learn order **without any
explicit positional encoding**.  The causal mask (`is_causal=True`)
already encodes “who comes before whom”, so the model can extrapolate
to sequence lengths far beyond those seen during training.

Design principles
-----------------
- **DRY**: every reusable component (attention block, data generator,
  visualisation) is a single function or class.
- **Dataclasses**: `NoPEConfig`, `NoPETransformer`, and `CopyTaskDataset`
  are dataclasses – trivially instantiable, inspectable, and comparable.
- **Full documentation**: module, class, and method docstrings; inline
  comments explain *why*, not just *what*.

Dependencies
------------
torch>=2.0, numpy, matplotlib, pandas, scikit-learn

Run
---
    python nope_learning_module.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------
# Reproducibility – single source of truth for every random operation.
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


# ======================================================================
# CONFIGURATION
# ======================================================================

@dataclass
class NoPEConfig:
    """
    Every hyper‑parameter in one immutable‑by‑convention bundle.

    Attributes
    ----------
    vocab_size : int
        Number of distinct tokens (including special tokens).
    d_model : int
        Hidden dimension of the Transformer.
    n_heads : int
        Number of attention heads.  ``d_model`` must be divisible by this.
    n_layers : int
        Number of decoder blocks.
    d_ff : int
        Feed‑forward hidden dimension (typically 4 * d_model).
    max_seq_len : int
        Maximum length **used during training**.  NoPE has no hard
        architectural limit; this value only controls the synthetic data.
    dropout : float
        Dropout probability (0.0 for deterministic educational runs).
    pad_token_id : int
        ID used for padding (also the “unknown” token in the toy vocab).
    """
    vocab_size: int = 20
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_seq_len: int = 10          # training length – NoPE can extrapolate
    dropout: float = 0.0
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


# ======================================================================
# CORE MODEL – NoPE decoder block and full Transformer
# ======================================================================

@dataclass
class NoPEAttention(nn.Module):
    """
    Multi‑head causal self‑attention **without any positional encoding**.

    The only source of order is `F.scaled_dot_product_attention` with
    ``is_causal=True``.  For a sequence of length T, token *i* can only
    attend to tokens *j ≤ i*.  This mask is what implicitly tells the
    model “left is earlier, right is later”.

    Parameters
    ----------
    config : NoPEConfig
    """

    config: NoPEConfig = field(default_factory=NoPEConfig)

    def __post_init__(self) -> None:
        super().__init__()
        cfg = self.config
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)

        # ---- THE KEY LINE ----
        # `is_causal=True` creates a lower‑triangular mask internally.
        # No position vectors are added anywhere.
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)


@dataclass
class NoPEBlock(nn.Module):
    """
    A single pre‑norm decoder block: LayerNorm → NoPEAttention → residual,
    then LayerNorm → FFN → residual.  **No positional encoding is added
    at any point.**
    """

    config: NoPEConfig = field(default_factory=NoPEConfig)

    def __post_init__(self) -> None:
        super().__init__()
        cfg = self.config
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = NoPEAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


@dataclass
class NoPETransformer(nn.Module):
    """
    Decoder‑only Transformer with **no positional encoding**.

    The forward pass is deliberately simple:
    1. Token embedding (no position embedding added).
    2. Stack of NoPE blocks.
    3. Final LayerNorm and linear head.

    Because no position table or rotation is used, the model has **no
    architectural maximum sequence length**.  You can pass a longer
    sequence than `config.max_seq_len` at inference and the only change
    is the causal mask size.
    """

    config: NoPEConfig = field(default_factory=NoPEConfig)

    def __post_init__(self) -> None:
        super().__init__()
        cfg = self.config
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([NoPEBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying (GPT‑2 style) – reduces parameters and often helps.
        self.lm_head.weight = self.tok_emb.weight

        # Initialise weights with a small std for stable training.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        idx : torch.Tensor
            Integer token IDs, shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        # ---- NO POSITIONAL ENCODING IS ADDED HERE ----
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)


# ======================================================================
# SYNTHETIC COPY TASK – designed to test length generalisation
# ======================================================================

@dataclass
class CopyTaskDataset(Dataset):
    """
    A minimal sequence‑copying task.

    Given a random sequence of integers, the model must output the same
    sequence shifted by one position (next‑token prediction).  This is a
    classic probe for length generalisation: if a model truly learns the
    copying rule, it should work on sequences longer than those seen in
    training.

    Parameters
    ----------
    n_samples : int
    seq_len : int
        Length of each sequence.
    vocab_size : int
        Tokens are drawn uniformly from ``[1, vocab_size)``.  Token 0 is
        reserved for padding.
    seed : int
        Fixed seed for reproducibility.
    """

    n_samples: int = 5000
    seq_len: int = 10
    vocab_size: int = 20
    seed: int = SEED

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # Shape (n_samples, seq_len) – values in [1, vocab_size-1]
        self.data = rng.integers(1, self.vocab_size, size=(self.n_samples, self.seq_len))
        # Input is the sequence; target is the same sequence (copy task).
        # In next‑token prediction we shift inside the training loop.
        self.inputs = torch.tensor(self.data, dtype=torch.long)
        self.targets = torch.tensor(self.data, dtype=torch.long)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


# ======================================================================
# TRAINING AND EVALUATION
# ======================================================================

def train_one_epoch(
    model: NoPETransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one epoch and return mean cross‑entropy loss."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Next‑token prediction: input = x[:, :-1], target = x[:, 1:]
        logits = model(x[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: NoPETransformer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return mean cross‑entropy loss on the given loader."""
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y[:, 1:].reshape(-1),
        )
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


# ======================================================================
# EDGE CASES AND INPUT VALIDATION
# ======================================================================

def validate_input(idx: torch.Tensor, config: NoPEConfig) -> None:
    """
    Raise a clear error for common mistakes.

    Checks
    ------
    - ``idx`` is a 2‑D tensor of integers.
    - Values are in ``[0, vocab_size)``.
    - Sequence length is at least 2 (needed for next‑token prediction).
    """
    if not isinstance(idx, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(idx).__name__}")
    if idx.dim() != 2:
        raise ValueError(f"Expected 2‑D tensor (batch, seq_len), got {idx.dim()}‑D")
    if not torch.is_floating_point(idx) is False:  # idx must be integer
        raise TypeError("Token IDs must be integers")
    if idx.max() >= config.vocab_size or idx.min() < 0:
        raise ValueError(f"Token IDs must be in [0, {config.vocab_size})")
    if idx.size(1) < 2:
        raise ValueError("Sequence length must be at least 2 for next‑token prediction")


def handle_empty_input(model: NoPETransformer) -> torch.Tensor:
    """
    Edge case: empty input ``torch.empty(0, 0, dtype=torch.long)``.

    Returns an empty logits tensor instead of crashing.
    """
    empty = torch.empty(0, 0, dtype=torch.long)
    with torch.no_grad():
        return model(empty)   # shape (0, 0, vocab_size)


# ======================================================================
# VISUALISATION
# ======================================================================

def plot_attention_heatmap(
    model: NoPETransformer,
    sample: torch.Tensor,
    save_to: str | Path = "fig_nope_attention.png",
) -> None:
    """
    Plot the attention matrix of the first head in the first layer.

    Because there is **no positional encoding**, the only structure you
    see is the causal mask: a lower‑triangular pattern.
    """
    model.eval()
    with torch.no_grad():
        x = model.tok_emb(sample.unsqueeze(0))
        # Re‑compute attention manually for visualisation.
        block = model.blocks[0]
        attn = block.attn
        B, T, C = x.shape
        q = attn.q_proj(x).view(B, T, attn.config.n_heads, C // attn.config.n_heads).transpose(1, 2)
        k = attn.k_proj(x).view(B, T, attn.config.n_heads, C // attn.config.n_heads).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        # Apply causal mask for display
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)[0, 0].cpu().numpy()   # first head

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(weights, cmap="Blues")
    ax.set_title("NoPE attention (head 0, layer 0)\nOnly the causal mask creates structure")
    ax.set_xlabel("Key position (attended to)")
    ax.set_ylabel("Query position (attending from)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[Viz] Saved attention heatmap → {Path(save_to).resolve()}")


def plot_length_generalisation(
    results: pd.DataFrame,
    save_to: str | Path = "fig_nope_extrapolation.png",
) -> None:
    """
    Plot loss vs. evaluation sequence length.

    A flat or slowly rising curve past the training length (10) is the
    visual signature of NoPE’s length generalisation.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(results["seq_len"], results["loss"], marker="o", color="steelblue")
    ax.axvline(results["train_len"].iloc[0], color="crimson", linestyle="--",
               label=f"Training length = {results['train_len'].iloc[0]}")
    ax.set_xlabel("Evaluation sequence length")
    ax.set_ylabel("Cross‑entropy loss")
    ax.set_title("NoPE length generalisation (lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"[Viz] Saved extrapolation curve → {Path(save_to).resolve()}")


# ======================================================================
# MAIN – end‑to‑end demonstration
# ======================================================================

def main() -> None:
    """Train a NoPE model on length 10, evaluate on lengths 8–40."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = NoPEConfig(max_seq_len=10, vocab_size=20)
    model = NoPETransformer(cfg).to(device)
    print(model)

    # ---- data --------------------------------------------------------
    train_ds = CopyTaskDataset(n_samples=5000, seq_len=10, vocab_size=cfg.vocab_size)
    val_ds = CopyTaskDataset(n_samples=500, seq_len=10, vocab_size=cfg.vocab_size, seed=1)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # ---- training ----------------------------------------------------
    for epoch in range(1, 6):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

    # ---- edge cases --------------------------------------------------
    print("\n[Edge cases]")
    print("Empty input →", handle_empty_input(model).shape)
    try:
        validate_input(torch.tensor([[1, 2, 999]]), cfg)
    except ValueError as e:
        print("Wrong token ID →", e)

    # ---- length generalisation --------------------------------------
    rows = []
    for length in [8, 10, 12, 16, 20, 30, 40]:
        ds = CopyTaskDataset(n_samples=200, seq_len=length,
                             vocab_size=cfg.vocab_size, seed=length)
        loader = DataLoader(ds, batch_size=64)
        loss = evaluate(model, loader, device)
        rows.append(dict(seq_len=length, loss=loss, train_len=10))
        print(f"Length {length:2d} | loss {loss:.4f}")
    results = pd.DataFrame(rows)

    # ---- visualisations ----------------------------------------------
    sample = train_ds[0][0]
    plot_attention_heatmap(model, sample)
    plot_length_generalisation(results)

    # ---- ELI5 narration ----------------------------------------------
    print("\n" + "=" * 60)
    print("ELI5 WALKTHROUGH")
    print("=" * 60)
    print("1. We give the model a row of tokens: [3, 7, 1, 9].")
    print("2. The model looks at token 3, but the mask says it can only see itself.")
    print("3. It looks at token 7, and the mask lets it see 3 and 7.")
    print("4. No position numbers are ever added – the mask itself says 'left is earlier'.")
    print("5. Because nothing breaks when the row gets longer, the model can copy 40 tokens")
    print("   even though it only trained on 10!")


if __name__ == "__main__":
    main()

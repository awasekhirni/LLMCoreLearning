"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
tepel_exercises.py
======================

Complete solutions to Exercises 1–3 from
`012_tepel_learning.py`.

Run:
    python3 tepel_exercises.py

Every solution is a @dataclass with a single `run()` method returning a
pandas DataFrame so you can compare against the Expected Outcomes in the
main module. All random state flows from SEED for reproducibility.

Datasets
--------
 Project Gutenberg #100 (Shakespeare) — public domain. Loaded via
  `embedding_learning_module.load_corpus()`, which falls back to the
  embedded excerpt on any network error.
 Embedded synthetic toy corpus — generated in this file with a fixed
  seed for the weight-tying and vocab-extension demos.

Exercise 1 — Sinusoidal vs learned positional encodings under distribution shift.
Train two tiny input-embedding models (one sinusoidal, one learned) on a task where token order matters—e.g., predict whether the sequence "a b" or "b a" was given. Evaluate both on sequences at position 10 that never appeared in training. Hint: the learned PE will fail when evaluated at a position ≥ max_position_embeddings; the sinusoidal PE will also degrade but can be evaluated at unseen positions. Plot accuracy vs evaluation position for both.

Exercise 2 — Weight tying: parameter cost vs quality.
Using TinyTrainingDemo as a base, add a tie_weights: bool flag. Build two training runs on the toy corpus (tied vs untied with a separate LM head). Compare (a) final training loss, (b) total parameter count, (c) memory (using embedding_memory_mb). Report a table. Hint: with a tiny corpus and tiny model, you should see near-identical loss but ~2× fewer embedding params when tied. Discuss why this scales to real LLMs.

Exercise 3 — Vocabulary extension.
Starting with a tokenizer trained on "the cat sat on the mat", "extend" the vocab by adding 3 new words (e.g., "dog", "log", "sonnet"). Demonstrate two initialisation strategies for the new embedding rows: (i) random init, (ii) mean of existing rows. Compute the OOV rate on a held-out sentence before and after extension. Hint: use tokenizer.vocab[word] = len(tokenizer.vocab) for each new word, then np.concatenate a new random (or mean) block onto embedding.weight. If you can, run 50 SGD steps on the new words only and show the loss drop.



"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from embedding_learning_module import (
    EmbeddingConfig,
    EmbeddingMetrics,
    InputEmbedding,
    LearnedPE,
    SimpleWordTokenizer,
    SinusoidalPE,
    TinyTrainingDemo,
    TokenEmbedding,
    TiedLMHead,
    load_corpus,
    validate_token_ids,
    SEED,
)

# Per-exercise output directory
OUT_DIR = Path("exercise_figures")
OUT_DIR.mkdir(exist_ok=True)


# ======================================================================
# EXERCISE 1 — SINUSOIDAL vs LEARNED PE UNDER POSITION SHIFT
# ======================================================================

@dataclass
class Exercise1Solution:
    """
    Train a tiny order-classifier on 2-token sequences at positions 0–4,
    then evaluate at positions 5–15 (unseen during training).

    Intuition
    ---------
    - Sinusoidal PE: pattern is defined at all positions, so classification
      should transfer (with some accuracy loss).
    - Learned PE: rows 5–15 were never trained; the model is forced to use
      random uninitialised rows → accuracy collapses to chance.

    This makes the card's "con" about fixed max positions concrete.
    """

    n_train_positions: int = 5
    n_test_positions: int = 15
    n_steps: int = 400
    lr: float = 0.3
    d_model: int = 16

    def _make_sequence(self, pos: int, order: int, vocab_size: int) -> Tuple[np.ndarray, int]:
        """Build a padded sequence of length (pos+2) so the discriminating
        tokens appear at positions (pos, pos+1). Label is `order`."""
        L = pos + 2
        ids = np.full((L,), 0, dtype=np.int64)              # <pad> filler
        a, b = 5, 6                                          # two fixed "word" ids
        ids[pos]     = a if order == 0 else b
        ids[pos + 1] = b if order == 0 else a
        return ids, order

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def _train_one(self, pe_type: str) -> Tuple[InputEmbedding, TiedLMHead, float]:
        cfg = EmbeddingConfig(
            vocab_size=10, d_model=self.d_model, padding_idx=0,
            max_position_embeddings=self.n_test_positions + 2,
            positional_encoding_type=pe_type,
            init_std=0.1, scale_by_sqrt_d=False, seed=SEED,
        )
        model = InputEmbedding(cfg)
        tied = TiedLMHead(model.token_embedding)
        rng = np.random.default_rng(SEED)
        data = []
        for _ in range(256):
            pos = int(rng.integers(0, self.n_train_positions))
            order = int(rng.integers(0, 2))
            data.append(self._make_sequence(pos, order, cfg.vocab_size))

        for step in range(1, self.n_steps + 1):
            ids, y = data[(step - 1) % len(data)]
            emb = model(ids[None, :])[0]                    # (L, D)
            hidden = emb.mean(axis=0)                       # (D,)
            logits = tied(hidden[None, :])[0]               # (V,)
            probs = self._softmax(logits)
            # Focus the loss on the class logits we care about (0 vs 1)
            # by only computing CE over {0, 1} for simplicity.
            p = probs[[0, 1]] / probs[[0, 1]].sum()
            loss = -math.log(max(p[y], 1e-12))

            # Backward (tied, mean-pooled hidden)
            dlogits_full = np.zeros_like(probs)
            dlogits_full[0] = p[0] - (1 if y == 0 else 0)
            dlogits_full[1] = p[1] - (1 if y == 1 else 0)
            dW_out = np.outer(dlogits_full, hidden)
            d_hidden = dlogits_full @ tied.weight
            dW_in = np.zeros_like(model.token_embedding.weight)
            for tid in ids:
                if tid == cfg.padding_idx:
                    continue
                dW_in[tid] += d_hidden / len(ids)
            grad = dW_out + dW_in
            grad[cfg.padding_idx] = 0.0
            model.token_embedding.weight -= self.lr * grad

        # Final training accuracy
        correct = 0
        for ids, y in data:
            emb = model(ids[None, :])[0]
            hidden = emb.mean(axis=0)
            logits = tied(hidden[None, :])[0, [0, 1]]
            pred = int(np.argmax(logits))
            correct += (pred == y)
        return model, tied, correct / len(data)

    def _eval_at_positions(
        self, model: InputEmbedding, tied: TiedLMHead, positions: Sequence[int],
    ) -> np.ndarray:
        """Accuracy at each test position (0 = chance, 1 = perfect)."""
        accs = []
        for pos in positions:
            correct = 0
            for order in (0, 1):
                ids, y = self._make_sequence(pos, order, model.config.vocab_size)
                try:
                    emb = model(ids[None, :])[0]
                except ValueError:
                    # Position beyond table: count as a wrong prediction.
                    accs.append(0.0)
                    correct = 0
                    break
                hidden = emb.mean(axis=0)
                logits = tied(hidden[None, :])[0, [0, 1]]
                correct += int(np.argmax(logits) == y)
            else:
                accs.append(correct / 2)
        return np.asarray(accs)

    def run(self) -> pd.DataFrame:
        print("=" * 70)
        print(" Exercise 1 — Sinusoidal vs Learned PE under position shift")
        print("=" * 70)
        sin_model, sin_head, sin_train_acc = self._train_one("sinusoidal")
        lrn_model, lrn_head, lrn_train_acc = self._train_one("learned")
        print(f"Train accuracy: sinusoidal={sin_train_acc:.2f}, learned={lrn_train_acc:.2f}")

        train_pos = list(range(self.n_train_positions))
        test_pos  = list(range(self.n_test_positions))
        sin_train = self._eval_at_positions(sin_model, sin_head, train_pos)
        sin_test  = self._eval_at_positions(sin_model, sin_head, test_pos)
        lrn_train = self._eval_at_positions(lrn_model, lrn_head, train_pos)
        lrn_test  = self._eval_at_positions(lrn_model, lrn_head, test_pos)

        df = pd.DataFrame({
            "position": test_pos,
            "sinusoidal": sin_test,
            "learned": lrn_test,
        })
        print(df.to_string(index=False))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(test_pos, sin_test, marker="o", label="Sinusoidal (generalises)")
        ax.plot(test_pos, lrn_test, marker="s", label="Learned (capped)")
        ax.axvline(self.n_train_positions - 0.5, color="gray", ls="--",
                   label="Training boundary")
        ax.set_xlabel("Sequence position (0 = first token)")
        ax.set_ylabel("Classification accuracy")
        ax.set_title("Exercise 1 — PE generalisation to unseen positions")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = OUT_DIR / "ex1_pe_generalisation.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Ex1] Figure saved: {path.resolve()}")
        return df


# ======================================================================
# EXERCISE 2 — WEIGHT TYING: PARAMETER COST vs QUALITY
# ======================================================================

@dataclass
class Exercise2Solution:
    """
    Train `TinyTrainingDemo` twice on the same toy corpus:
    once with tied input/output weights and once with an untied LM head.

    Report final loss, parameter count, and memory for each variant.
    Expected: nearly identical loss, ~2× fewer embedding params tied.
    """

    corpus: str = (
        "the cat sat on the mat\n"
        "the dog sat on the log\n"
        "a cat and a dog\n"
        "the mat is on the floor\n"
        "the dog is on the mat\n"
    ).strip()
    d_model: int = 8
    steps: int = 300
    lr: float = 0.5

    def _train(self, tie: bool) -> Tuple[float, int, float]:
        """Return (final_loss, n_params, memory_mb)."""
        tok = SimpleWordTokenizer().build(self.corpus)
        cfg = EmbeddingConfig(
            vocab_size=tok.vocab_size, d_model=self.d_model,
            padding_idx=tok.pad_id, max_position_embeddings=16,
            positional_encoding_type="none", init_std=0.1,
            scale_by_sqrt_d=False, seed=SEED + 2,
        )
        if tie:
            demo = TinyTrainingDemo(corpus=self.corpus, config=cfg,
                                    steps=self.steps, lr=self.lr)
            history = demo.run(verbose_every=None)
            n_params = cfg.vocab_size * cfg.d_model
        else:
            # Untied variant: copy the demo but with a separate output matrix.
            history, n_params = self._run_untied(cfg, tok)
        memory_mb = n_params * 2 / (1024 ** 2)       # fp16
        return float(history["loss"].iloc[-1]), int(n_params), memory_mb

    def _run_untied(self, cfg: EmbeddingConfig, tok: SimpleWordTokenizer) -> Tuple[pd.DataFrame, int]:
        """Same 4-token->next-token task but with a *separate* output matrix."""
        pairs = []
        ids = tok.encode(self.corpus, add_special=False)
        ctx_len = 4
        for i in range(len(ids) - ctx_len):
            pairs.append((np.asarray(ids[i:i + ctx_len], dtype=np.int64), ids[i + ctx_len]))
        emb = TokenEmbedding(cfg)
        rng = np.random.default_rng(cfg.seed + 99)
        out_w = rng.normal(0, cfg.init_std, size=(cfg.vocab_size, cfg.d_model))
        pad = cfg.padding_idx
        history = []
        for step in range(1, self.steps + 1):
            ctx, y = pairs[(step - 1) % len(pairs)]
            tok_emb = emb(ctx)
            keep = (ctx != pad).astype(np.float64)[:, None]
            hidden = (tok_emb * keep).sum(axis=0) / max(keep.sum(), 1.0)
            logits = hidden @ out_w.T
            z = logits - logits.max()
            p = np.exp(z) / np.exp(z).sum()
            loss = -math.log(max(p[y], 1e-12))
            # Gradients (untied: out_w and emb receive separate updates)
            dlogits = p.copy()
            dlogits[y] -= 1.0
            dW_out = np.outer(dlogits, hidden)
            d_hidden = dlogits @ out_w
            dW_in = np.zeros_like(emb.weight)
            share = 1.0 / max(keep.sum(), 1.0)
            for tid in ctx:
                if tid == pad:
                    continue
                dW_in[tid] += d_hidden * share
            out_w -= self.lr * dW_out
            emb.weight -= self.lr * dW_in
            emb.weight[pad] = 0.0
            history.append(dict(step=step, loss=loss))
        n_params = cfg.vocab_size * cfg.d_model * 2
        return pd.DataFrame(history), n_params

    def run(self) -> pd.DataFrame:
        print("=" * 70)
        print(" Exercise 2 — Weight tying: parameter cost vs quality")
        print("=" * 70)
        tied_loss, tied_params, tied_mem = self._train(tie=True)
        unt_loss, unt_params, unt_mem = self._train(tie=False)
        df = pd.DataFrame([
            dict(variant="untied", final_loss=unt_loss, params=unt_params, memory_mb=unt_mem),
            dict(variant="tied",   final_loss=tied_loss, params=tied_params, memory_mb=tied_mem),
        ])
        print(df.to_string(index=False))
        ratio = unt_params / tied_params
        print(f"\nParameter reduction: {ratio:.2f}× fewer params with tying")
        print(f"Loss delta: {abs(tied_loss - unt_loss):.4f} (should be small)")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.bar(df["variant"], df["params"], color=["indianred", "seagreen"])
        ax1.set_ylabel("Parameter count")
        ax1.set_title("Exercise 2a — Embedding parameters")
        for i, v in enumerate(df["params"]):
            ax1.text(i, v + max(df["params"]) * 0.01, f"{v:,}", ha="center", fontsize=9)

        ax2.bar(df["variant"], df["final_loss"], color=["indianred", "seagreen"])
        ax2.set_ylabel("Final training loss")
        ax2.set_title("Exercise 2b — Final loss (lower is better)")
        for i, v in enumerate(df["final_loss"]):
            ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
        ax2.set_ylim(0, max(df["final_loss"]) * 1.2)

        fig.tight_layout()
        path = OUT_DIR / "ex2_weight_tying.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Ex2] Figure saved: {path.resolve()}")
        return df


# ======================================================================
# EXERCISE 3 — VOCABULARY EXTENSION
# ======================================================================

@dataclass
class Exercise3Solution:
    """
    Extend a small vocab with 3 new words and compare:
      (i)  random init of new rows, (ii) mean-of-existing init.
    Measure OOV rate on a held-out sentence before/after, and final
    training loss on the toy LM task.

    This mirrors the real recipe used to adapt pretrained LLMs to new
    languages: keep the base rows, add new ones (mean-init), freeze
    old rows, train new ones.
    """

    base_corpus: str = "the cat sat on the mat the dog sat on the log"
    new_words: Tuple[str, ...] = ("sonnet", "bard", "thou")
    held_out: str = "thou wrote a sonnet on the bard"
    d_model: int = 8
    steps: int = 200
    lr: float = 0.5

    @staticmethod
    def _oov_rate(tokenizer: SimpleWordTokenizer, text: str) -> float:
        ids = tokenizer.encode(text, add_special=False)
        if not ids:
            return 0.0
        return sum(1 for i in ids if i == tokenizer.unk_id) / len(ids)

    def _train_extended(self, extended_weight: np.ndarray, tok: SimpleWordTokenizer) -> float:
        """Train on the base corpus with the extended weight frozen-ish."""
        cfg = EmbeddingConfig(
            vocab_size=extended_weight.shape[0], d_model=self.d_model,
            padding_idx=tok.pad_id, max_position_embeddings=32,
            positional_encoding_type="none", init_std=0.1,
            scale_by_sqrt_d=False, seed=SEED,
        )
        emb = TokenEmbedding(cfg)
        emb.weight = extended_weight.copy()
        tied = TiedLMHead(emb)
        ids = tok.encode(self.base_corpus, add_special=False)
        ctx_len = 4
        pairs = [(np.asarray(ids[i:i + ctx_len], dtype=np.int64), ids[i + ctx_len])
                 for i in range(len(ids) - ctx_len)]
        pad = cfg.padding_idx
        for step in range(1, self.steps + 1):
            ctx, y = pairs[(step - 1) % len(pairs)]
            tok_emb = emb(ctx)
            keep = (ctx != pad).astype(np.float64)[:, None]
            hidden = (tok_emb * keep).sum(axis=0) / max(keep.sum(), 1.0)
            logits = tied(hidden[None, :])[0]
            z = logits - logits.max()
            p = np.exp(z) / np.exp(z).sum()
            dlogits = p.copy(); dlogits[y] -= 1.0
            dW_out = np.outer(dlogits, hidden)
            d_hidden = dlogits @ tied.weight
            dW_in = np.zeros_like(emb.weight)
            share = 1.0 / max(keep.sum(), 1.0)
            for tid in ctx:
                if tid == pad:
                    continue
                dW_in[tid] += d_hidden * share
            grad = dW_out + dW_in
            grad[pad] = 0.0
            # Freeze the *original* rows: only new rows get gradient.
            old_v = self.old_vocab_size
            grad[:old_v] = 0.0
            emb.weight -= self.lr * grad
            loss = -math.log(max(p[y], 1e-12))
        return loss

    def run(self) -> pd.DataFrame:
        print("=" * 70)
        print(" Exercise 3 — Vocabulary extension")
        print("=" * 70)
        tok = SimpleWordTokenizer().build(self.base_corpus)
        self.old_vocab_size = tok.vocab_size
        old_oov = self._oov_rate(tok, self.held_out)
        base_cfg = EmbeddingConfig(
            vocab_size=tok.vocab_size, d_model=self.d_model,
            padding_idx=tok.pad_id, max_position_embeddings=32,
            positional_encoding_type="none", init_std=0.1,
            scale_by_sqrt_d=False, seed=SEED,
        )
        base_emb = TokenEmbedding(base_cfg)

        # Extend vocab
        new_ids = []
        for w in self.new_words:
            if w not in tok.vocab:
                tok.vocab[w] = len(tok.vocab)
            new_ids.append(tok.vocab[w])
        new_oov = self._oov_rate(tok, self.held_out)

        rng = np.random.default_rng(SEED + 7)
        # (i) random init
        rand_new = rng.normal(0, base_cfg.init_std, size=(len(self.new_words), self.d_model))
        rand_ext = np.vstack([base_emb.weight, rand_new])
        # (ii) mean-of-existing init
        mean_new = np.tile(base_emb.weight.mean(axis=0, keepdims=True),
                           (len(self.new_words), 1))
        mean_ext = np.vstack([base_emb.weight, mean_new])

        rand_loss = self._train_extended(rand_ext, tok)
        mean_loss = self._train_extended(mean_ext, tok)

        df = pd.DataFrame([
            dict(strategy="random init", new_row_norm=float(np.linalg.norm(rand_new, axis=1).mean()),
                 final_loss=rand_loss, oov_before=old_oov, oov_after=new_oov),
            dict(strategy="mean init",   new_row_norm=float(np.linalg.norm(mean_new, axis=1).mean()),
                 final_loss=mean_loss, oov_before=old_oov, oov_after=new_oov),
        ])
        print(f"held-out sentence: '{self.held_out}'")
        print(f"OOV rate: before={old_oov:.2%}  after={new_oov:.2%}")
        print(df.to_string(index=False))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.bar(df["strategy"], df["final_loss"], color=["indianred", "seagreen"])
        ax1.set_ylabel("Final training loss")
        ax1.set_title("Exercise 3a — New-row init strategy")
        for i, v in enumerate(df["final_loss"]):
            ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

        oovs = [old_oov * 100, new_oov * 100]
        ax2.bar(["before extension", "after extension"], oovs,
                color=["indianred", "seagreen"])
        ax2.set_ylabel("OOV rate on held-out (%)")
        ax2.set_title("Exercise 3b — OOV drops after extension")
        for i, v in enumerate(oovs):
            ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
        fig.tight_layout()
        path = OUT_DIR / "ex3_vocab_extension.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Ex3] Figure saved: {path.resolve()}")
        return df


# ======================================================================
# DRIVER
# ======================================================================

def main() -> None:
    pd.set_option("display.width", 180)
    Exercise1Solution().run()
    print()
    Exercise2Solution().run()
    print()
    Exercise3Solution().run()
    print("\nAll exercise figures saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()



# # Required for the main module
# pip3 install numpy pandas matplotlib

# # Optional (only for the reference cross-check)
# pip3 install torch

# # Required for the exercises (imports from the main module,
# # so run the main module's directory must be on PYTHONPATH)
# python3 embedding_exercises.py

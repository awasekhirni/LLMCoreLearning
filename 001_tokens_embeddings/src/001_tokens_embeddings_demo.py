"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

tokens_and_embeddings.py
=======================
Tokens and Embeddings: a self-contained learning module.

Pipeline:
    raw text  --[Tokenizer]-->  token IDs  --[EmbeddingLayer]-->  dense vectors
                                                        |
                                       [Word2Vec SGNS]  <-- trained with
                                                          contrastive objective

Implements three classes mirroring the three modules of the technical card:
    1. SimpleBPETokenizer  (encode / decode, subword + fallback to bytes)
    2. EmbeddingLayer      (numpy lookup matrix, supports padding_idx)
    3. Word2VecSGNS        (Skip-gram with Negative Sampling, pure numpy)

All RNG is seeded (SEED=42) so outputs are reproducible.
Setting up the environment
python3 -m venv .
#installing the required packages
pip3 install numpy scikit-learn matplotlib
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Constants / Special Tokens
# ---------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


# ===========================================================================
# MODULE 1 — TOKENIZER
# ===========================================================================
@dataclass
class SimpleBPETokenizer:
    """
    A minimal Byte-Pair-Encoding (BPE) tokenizer for educational use.

    BPE algorithm (training):
        1. Start with each character (or byte) as its own token.
        2. Count adjacent token-pair frequencies in the corpus.
        3. Merge the most frequent pair into a new single token.
        4. Repeat steps 2-3 until `vocab_size` is reached or no merges remain.

    Encode: greedy left-to-right application of learned merge rules.
    Decode:  concatenate token strings in order.

    Attributes
    ----------
    vocab_size : int
        Target vocabulary size (including special tokens).
    merges : List[Tuple[str, str]]
        Ordered list of merges; the order encodes priority.
    vocab : Dict[str, int]
        Token-string -> integer ID.
    id_to_token : Dict[int, str]
        Integer ID -> token-string (reverse of vocab).
    """

    vocab_size: int = 500
    # Special tokens get the first IDs (0..3) so they are easy to reference.
    merges: List[Tuple[str, str]] = field(default_factory=list)
    vocab: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)
    pad_id: int = 0
    unk_id: int = 1

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Pre-register the special tokens so they occupy IDs 0..len(SPECIAL)-1.
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.vocab[tok] = i
            self.id_to_token[i] = tok

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Split raw text into word-level pre-tokens. Each pre-token is then
        represented as a list of characters with an end-of-word marker '_'
        so word boundaries survive merging.
        """
        # Normalize unicode (NFKC) and lower-case to shrink the alphabet.
        text = unicodedata.normalize("NFKC", text).lower()
        # Split on whitespace; keep punctuation as separate pre-tokens.
        words = re.findall(r"\w+|[^\w\s]", text)
        return words

    def _word_to_symbols(self, word: str) -> List[str]:
        """Convert a word to a list of single-character symbols + end marker."""
        return list(word) + ["</w>"]

    def train(self, corpus: Sequence[str]) -> "SimpleBPETokenizer":
        """
        Learn BPE merge rules from `corpus`.

        Parameters
        ----------
        corpus : Sequence[str]
            List of raw text strings.

        Returns
        -------
        self (for chaining).
        """
        if not corpus:
            raise ValueError("Cannot train BPE on an empty corpus.")

        # Step 1: pre-tokenize and count word frequencies.
        word_freqs: Counter = Counter()
        for doc in corpus:
            for w in self._pre_tokenize(doc):
                word_freqs[w] += 1

        # Step 2: represent each unique word as a list of symbols.
        #   splits[word] = ['c', 'a', 't', '</w>']
        splits: Dict[str, List[str]] = {
            w: self._word_to_symbols(w) for w in word_freqs
        }

        # Step 3: build the initial vocabulary from the alphabet.
        alphabet: set = set()
        for syms in splits.values():
            alphabet.update(syms)
        for sym in sorted(alphabet):
            if sym not in self.vocab:
                idx = len(self.vocab)
                self.vocab[sym] = idx
                self.id_to_token[idx] = sym

        # Step 4: iteratively merge the most frequent adjacent pair.
        target_merges = self.vocab_size - len(self.vocab)
        for _ in range(target_merges):
            pair_counts: Counter = Counter()
            for word, freq in word_freqs.items():
                syms = splits[word]
                for i in range(len(syms) - 1):
                    pair_counts[(syms[i], syms[i + 1])] += freq
            if not pair_counts:
                break  # nothing left to merge
            best_pair, _ = pair_counts.most_common(1)[0]
            # Apply the merge to every word's symbol list.
            new_token = best_pair[0] + best_pair[1]
            for word in word_freqs:
                syms = splits[word]
                merged: List[str] = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best_pair:
                        merged.append(new_token)
                        i += 2
                    else:
                        merged.append(syms[i])
                        i += 1
                splits[word] = merged
            self.merges.append(best_pair)
            idx = len(self.vocab)
            self.vocab[new_token] = idx
            self.id_to_token[idx] = new_token
        return self

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------
    def _apply_merges(self, symbols: List[str]) -> List[str]:
        """Apply learned merge rules greedily to a list of symbols."""
        # Build a lookup: (a, b) -> merged_token, preserving rule order.
        merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        while True:
            # Find the highest-priority applicable pair.
            best_i, best_rank = -1, float("inf")
            for i in range(len(symbols) - 1):
                r = merge_rank.get((symbols[i], symbols[i + 1]))
                if r is not None and r < best_rank:
                    best_rank, best_i = r, i
            if best_i == -1:
                break
            merged = symbols[best_i] + symbols[best_i + 1]
            symbols = symbols[:best_i] + [merged] + symbols[best_i + 2:]
        return symbols

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode raw text -> list of integer token IDs.

        Parameters
        ----------
        text : str
            Raw input string.
        add_bos, add_eos : bool
            Prepend / append sequence-boundary special tokens.
        max_length : Optional[int]
            If set, truncate to this many IDs (after special tokens).

        Returns
        -------
        List[int] of token IDs.

        Raises
        ------
        TypeError if `text` is not a str.
        ValueError if `text` is empty (after BOS/EOS would still be empty).
        """
        if not isinstance(text, str):
            raise TypeError(f"encode() expects str, got {type(text).__name__}")
        if text == "":
            raise ValueError("encode() received an empty string.")

        ids: List[int] = []
        if add_bos:
            ids.append(self.vocab[BOS_TOKEN])
        for word in self._pre_tokenize(text):
            syms = self._apply_merges(self._word_to_symbols(word))
            for s in syms:
                ids.append(self.vocab.get(s, self.unk_id))
        if add_eos:
            ids.append(self.vocab[EOS_TOKEN])
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs -> raw text.

        Special tokens (PAD, BOS, EOS, UNK) are dropped from the output.
        Unknown IDs raise IndexError.
        """
        if not isinstance(ids, (list, tuple, np.ndarray)):
            raise TypeError(f"decode() expects a sequence, got {type(ids).__name__}")
        parts: List[str] = []
        for i in ids:
            i = int(i)
            if i not in self.id_to_token:
                raise IndexError(f"Token ID {i} is out of vocabulary.")
            tok = self.id_to_token[i]
            if tok in SPECIAL_TOKENS:
                continue
            # Strip the </w> end-of-word marker and re-insert a space.
            if tok.endswith("</w>"):
                parts.append(tok[:-4] + " ")
            else:
                parts.append(tok)
        return "".join(parts).strip()

    def __len__(self) -> int:
        return len(self.vocab)


# ===========================================================================
# MODULE 2 — EMBEDDING LAYER
# ===========================================================================
@dataclass
class EmbeddingLayer:
    """
    A pure-NumPy embedding lookup table.

    Equivalent to `torch.nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)`
    but implemented in NumPy for portability. Weights are initialized with a
    small uniform distribution (Xavier-style) so that early cosine similarities
    are not degenerate.

    Attributes
    ----------
    vocab_size, hidden_dim : int
        Matrix shape.
    pad_id : Optional[int]
        If set, this row is forced to all-zeros and never updated — used so
        that [PAD] tokens contribute nothing to the loss.
    weight : np.ndarray, shape [vocab_size, hidden_dim]
        The learned (or randomly-initialized) embedding matrix.
    """

    vocab_size: int
    hidden_dim: int
    pad_id: Optional[int] = None
    weight: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(SEED)
        # Glorot-uniform init keeps variance balanced across layers.
        limit = np.sqrt(6.0 / (self.vocab_size + self.hidden_dim))
        self.weight = rng.uniform(
            -limit, limit, size=(self.vocab_size, self.hidden_dim)
        ).astype(np.float32)
        if self.pad_id is not None:
            self.weight[self.pad_id] = 0.0

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Look up embeddings for a batch of token IDs.

        Parameters
        ----------
        token_ids : np.ndarray, shape [batch, seq] or [seq]

        Returns
        -------
        np.ndarray, shape [..., hidden_dim]
        """
        if not isinstance(token_ids, np.ndarray):
            token_ids = np.asarray(token_ids)
        if token_ids.size == 0:
            raise ValueError("EmbeddingLayer.forward received empty input.")
        if np.any(token_ids >= self.vocab_size) or np.any(token_ids < 0):
            raise IndexError("Token ID out of range [0, vocab_size).")
        return self.weight[token_ids]

    def mean_pool(self, embeddings: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Mean-pool token embeddings to a single sentence vector.

        Parameters
        ----------
        embeddings : np.ndarray, shape [seq_len, hidden_dim]
        mask : Optional[np.ndarray], shape [seq_len], 1.0 for real tokens.
            If None, all tokens count.
        """
        if mask is None:
            return embeddings.mean(axis=0)
        denom = mask.sum(axis=0, keepdims=True).clip(min=1e-9)
        return (embeddings * mask[:, None]).sum(axis=0) / denom.squeeze(0)


# ===========================================================================
# MODULE 3 — WORD2VEC (SKIP-GRAM + NEGATIVE SAMPLING)
# ===========================================================================
class Word2VecSGNS:
    """
    Skip-gram with Negative Sampling (Mikolov et al. 2013), in pure NumPy.

    Objective per (center, context) pair:
        L = -log σ(v_c · u_o)  -  Σ_k log σ(-v_c · u_k)
    where v_c is the center-word embedding (input matrix W),
          u_o is the context-word embedding (output matrix C),
          u_k are k randomly-sampled negative words,
          σ is the sigmoid.

    Attributes
    ----------
    dim : int
        Embedding dimension.
    neg_samples : int
        Number of negative samples per positive pair.
    W : np.ndarray, shape [vocab, dim]   — input (center) embeddings.
    C : np.ndarray, shape [vocab, dim]   — output (context) embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 32,
        neg_samples: int = 5,
        learning_rate: float = 0.05,
        epochs: int = 20,
        seed: int = SEED,
    ) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        self.neg_samples = neg_samples
        self.lr = learning_rate
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        # Small uniform init — symmetry-breaking.
        self.W = (self.rng.uniform(-0.5 / dim, 0.5 / dim, (vocab_size, dim))).astype(np.float32)
        self.C = np.zeros((vocab_size, dim), dtype=np.float32)
        self.loss_history: List[float] = []
        # Unigram^0.75 sampling distribution for negatives.
        self._neg_dist: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _build_vocab(self, corpus: Sequence[Sequence[str]]) -> Dict[str, int]:
        counts: Counter = Counter()
        for sent in corpus:
            counts.update(sent)
        # Sort by descending frequency so common words get low IDs.
        words = [w for w, _ in counts.most_common()]
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.word_counts = np.array(
            [counts[self.idx2word[i]] for i in range(len(words))], dtype=np.float64
        )
        self.vocab_size = len(words)
        # Resize matrices to actual vocab.
        self.W = (self.rng.uniform(-0.5 / self.dim, 0.5 / self.dim,
                                    (self.vocab_size, self.dim))).astype(np.float32)
        self.C = np.zeros((self.vocab_size, self.dim), dtype=np.float32)
        # Unigram^0.75 distribution: flattens the long tail.
        unigram = self.word_counts ** 0.75
        self._neg_dist = unigram / unigram.sum()
        return self.word2idx

    # ------------------------------------------------------------------
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid.
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                       np.exp(x) / (1.0 + np.exp(x)))

    # ------------------------------------------------------------------
    def _generate_pairs(self, corpus: Sequence[Sequence[str]], window: int) -> List[Tuple[int, int]]:
        """Create (center, context) training pairs from the corpus."""
        pairs: List[Tuple[int, int]] = []
        for sent in corpus:
            ids = [self.word2idx[w] for w in sent if w in self.word2idx]
            for i, center in enumerate(ids):
                # Reduced window (Mikolov trick): sample a random radius 1..window.
                r = self.rng.integers(1, window + 1)
                lo = max(0, i - r)
                hi = min(len(ids), i + r + 1)
                for j in range(lo, hi):
                    if j != i:
                        pairs.append((center, ids[j]))
        return pairs

    # ------------------------------------------------------------------
    def train(self, corpus: Sequence[Sequence[str]], window: int = 2) -> "Word2VecSGNS":
        """
        Train the SGNS model on `corpus`.

        Parameters
        ----------
        corpus : Sequence[Sequence[str]]
            List of tokenized sentences (each a list of word strings).
        window : int
            Maximum context radius.
        """
        self._build_vocab(corpus)
        pairs = self._generate_pairs(corpus, window)
        if not pairs:
            raise ValueError("No training pairs generated — corpus too small.")
        print(f"[Word2Vec] vocab={self.vocab_size}  pairs={len(pairs)}")

        for epoch in range(self.epochs):
            self.rng.shuffle(pairs)
            total_loss = 0.0
            for center, context in pairs:
                # Sample `neg_samples` negative word indices (not the context).
                negs = self.rng.choice(
                    self.vocab_size, size=self.neg_samples, p=self._neg_dist
                )
                v_c = self.W[center]                # [dim]
                u_o = self.C[context]                # [dim]
                u_k = self.C[negs]                   # [neg, dim]

                # Forward
                pos_score = self._sigmoid(u_o @ v_c)         # scalar
                neg_scores = self._sigmoid(u_k @ v_c)        # [neg]

                # Loss (for logging only)
                total_loss += -np.log(pos_score + 1e-9) - np.sum(np.log(1 - neg_scores + 1e-9))

                # Gradients
                grad_pos = (pos_score - 1.0)            # scalar
                grad_neg = neg_scores                   # [neg]
                # grad wrt v_c
                grad_v = grad_pos * u_o + (grad_neg[:, None] * u_k).sum(axis=0)
                # grad wrt u_o
                grad_uo = grad_pos * v_c
                # grad wrt u_k
                grad_uk = grad_neg[:, None] * v_c[None, :]

                # SGD update
                self.W[center] -= self.lr * grad_v
                self.C[context] -= self.lr * grad_uo
                self.C[negs] -= self.lr * grad_uk

            avg_loss = total_loss / len(pairs)
            self.loss_history.append(avg_loss)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  epoch {epoch + 1:2d}/{self.epochs}  avg_loss={avg_loss:.4f}")
        return self

    # ------------------------------------------------------------------
    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return the top-k cosine-similar words to `word`."""
        if word not in self.word2idx:
            raise KeyError(f"'{word}' not in vocabulary.")
        idx = self.word2idx[word]
        vecs = self.W
        target = vecs[idx]
        # Cosine similarity with all words.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        sims = (vecs @ target) / (norms.squeeze() * (np.linalg.norm(target) + 1e-9))
        sims[idx] = -1.0  # exclude self
        top = np.argsort(-sims)[:top_k]
        return [(self.idx2word[i], float(sims[i])) for i in top]


# ===========================================================================
# SCENARIO DRIVERS
# ===========================================================================
def scenario_happy_path() -> None:
    """
    HAPPY PATH:
        Train a small BPE tokenizer on a toy corpus, encode/decode a sentence,
        look up embeddings, and inspect the resulting vectors.
    """
    print("\n" + "=" * 72)
    print("SCENARIO 1 — HAPPY PATH (BPE encode → embedding lookup)")
    print("=" * 72)

    toy_corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are pets",
        "the mat is on the floor",
        "a cat and a dog sat",
    ] * 5  # repeat to make merges more meaningful

    tok = SimpleBPETokenizer(vocab_size=80).train(toy_corpus)
    print(f"[Tokenizer] vocab_size = {len(tok)}")
    print(f"[Tokenizer] first 10 merges: {tok.merges[:10]}")

    sentence = "the cat sat"
    ids = tok.encode(sentence, add_bos=True, add_eos=True)
    print(f"[Encode] '{sentence}' -> {ids}")
    print(f"[Decode] {ids} -> '{tok.decode(ids)}'")

    emb = EmbeddingLayer(vocab_size=len(tok), hidden_dim=8, pad_id=tok.pad_id)
    vectors = emb.forward(np.array(ids))
    print(f"[Embedding] shape={vectors.shape}")
    print(f"[Embedding] PAD row (should be ~0): {emb.weight[tok.pad_id]}")


def scenario_edge_cases() -> None:
    """
    EDGE CASES:
        - Empty string
        - Wrong input type (int instead of str)
        - Out-of-vocabulary characters (uses UNK fallback)
        - Empty embedding lookup
        - Token ID out of range
        - Extreme sequence lengths
    """
    print("\n" + "=" * 72)
    print("SCENARIO 2 — EDGE CASES")
    print("=" * 72)

    tok = SimpleBPETokenizer(vocab_size=40).train(["the cat sat", "a dog ran"])

    # 2a. Empty string
    try:
        tok.encode("")
    except ValueError as e:
        print(f"[Edge-2a] empty string -> caught: {e}")

    # 2b. Wrong type
    try:
        tok.encode(123)  # type: ignore[arg-type]
    except TypeError as e:
        print(f"[Edge-2b] int input -> caught: {e}")

    # 2c. OOV character (e.g., emoji) — should map to UNK, not crash.
    ids = tok.encode("the cat 🐱")
    print(f"[Edge-2c] emoji input -> ids={ids}  (UNK id={tok.unk_id}, count={ids.count(tok.unk_id)})")

    # 2d. Empty embedding lookup
    emb = EmbeddingLayer(vocab_size=len(tok), hidden_dim=4, pad_id=tok.pad_id)
    try:
        emb.forward(np.array([], dtype=np.int64))
    except ValueError as e:
        print(f"[Edge-2d] empty lookup -> caught: {e}")

    # 2e. Out-of-range ID
    try:
        emb.forward(np.array([99999]))
    except IndexError as e:
        print(f"[Edge-2e] OOB id -> caught: {e}")

    # 2f. Long sequence truncation
    long_text = "the " * 500 + "cat"
    ids = tok.encode(long_text, max_length=16)
    print(f"[Edge-2f] truncated to {len(ids)} ids (max_length=16)")

    # 2g. Decode with non-existent ID
    try:
        tok.decode([99999])
    except IndexError as e:
        print(f"[Edge-2g] decode OOB -> caught: {e}")


def scenario_word2vec_training() -> None:
    """
    SCENARIO 3 — Train Word2Vec SGNS on a synthetic corpus and inspect
    learned semantic neighborhoods.
    """
    print("\n" + "=" * 72)
    print("SCENARIO 3 — WORD2VEC SGNS TRAINING")
    print("=" * 72)

    rng = np.random.default_rng(SEED)
    vocab_pool = [
        "cat", "dog", "kitten", "puppy", "pet", "animal",
        "car", "truck", "bus", "vehicle", "engine", "wheel",
        "run", "sat", "jump", "walk", "move", "stop",
        "the", "a", "an", "and", "on", "is",
    ]
    # Generate synthetic sentences by sampling template patterns.
    templates = [
        ["the", "CAT", "VERB", "on", "the", "MAT"],
        ["a", "CAT", "and", "a", "DOG", "are", "pets"],
        ["the", "VEHICLE", "VERB", "fast"],
    ]
    CAT = ["cat", "dog", "kitten", "puppy"]
    VEHICLE = ["car", "truck", "bus"]
    VERB = ["run", "sat", "jump", "walk", "move", "stop"]
    MAT = ["mat", "floor", "log", "road"]

    corpus: List[List[str]] = []
    for _ in range(800):
        tpl = list(rng.choice(templates))
        sent = []
        for tok in tpl:
            if tok == "CAT":
                sent.append(rng.choice(CAT))
            elif tok == "VEHICLE":
                sent.append(rng.choice(VEHICLE))
            elif tok == "VERB":
                sent.append(rng.choice(VERB))
            elif tok == "MAT":
                sent.append(rng.choice(MAT))
            else:
                sent.append(tok)
        corpus.append(sent)

    model = Word2VecSGNS(vocab_size=0, dim=32, neg_samples=5, epochs=20, learning_rate=0.05)
    model.train(corpus, window=2)

    for probe in ["cat", "car", "run"]:
        try:
            sims = model.most_similar(probe, top_k=4)
            print(f"[{probe}]  nearest: {sims}")
        except KeyError as e:
            print(f"[{probe}]  {e}")

    return model, corpus


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    scenario_happy_path()
    scenario_edge_cases()
    scenario_word2vec_training()

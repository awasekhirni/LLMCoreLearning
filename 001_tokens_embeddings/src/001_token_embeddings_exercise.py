"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

exercises.py
============
Solutions for Chapter 2 exercises:
    1. Vocabulary-size sensitivity
    2. Polysemy demonstration (Static vs. Word2Vec static-trained)
    3. Negative-sample ablation in Word2Vec

Setting up the environment
python3 -m venv .
#installing the required packages
pip3 install numpy scikit-learn matplotlib

"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Minimal Class Definitions (from Chapter 2 Module)
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

@dataclass
class SimpleBPETokenizer:
    vocab_size: int = 500
    merges: List[Tuple[str, str]] = field(default_factory=list)
    vocab: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)
    pad_id: int = 0
    unk_id: int = 1

    def __post_init__(self) -> None:
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.vocab[tok] = i
            self.id_to_token[i] = tok

    def _pre_tokenize(self, text: str) -> List[str]:
        text = unicodedata.normalize("NFKC", text).lower()
        return re.findall(r"\w+|[^\w\s]", text)

    def _word_to_symbols(self, word: str) -> List[str]:
        return list(word) + ["</w>"]

    def train(self, corpus: Sequence[str]) -> "SimpleBPETokenizer":
        word_freqs: Counter = Counter()
        for doc in corpus:
            for w in self._pre_tokenize(doc):
                word_freqs[w] += 1

        splits: Dict[str, List[str]] = {
            w: self._word_to_symbols(w) for w in word_freqs
        }

        alphabet: set = set()
        for syms in splits.values():
            alphabet.update(syms)
        for sym in sorted(alphabet):
            if sym not in self.vocab:
                idx = len(self.vocab)
                self.vocab[sym] = idx
                self.id_to_token[idx] = sym

        target_merges = self.vocab_size - len(self.vocab)
        for _ in range(target_merges):
            pair_counts: Counter = Counter()
            for word, freq in word_freqs.items():
                syms = splits[word]
                for i in range(len(syms) - 1):
                    pair_counts[(syms[i], syms[i + 1])] += freq
            if not pair_counts:
                break
            best_pair, _ = pair_counts.most_common(1)[0]
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

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        while True:
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

    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str) or text == "":
            return []
        ids: List[int] = []
        for word in self._pre_tokenize(text):
            syms = self._apply_merges(self._word_to_symbols(word))
            for s in syms:
                ids.append(self.vocab.get(s, self.unk_id))
        return ids

class EmbeddingLayer:
    def __init__(self, vocab_size: int, hidden_dim: int, pad_id: Optional[int] = None):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(SEED)
        limit = np.sqrt(6.0 / (vocab_size + hidden_dim))
        self.weight = rng.uniform(-limit, limit, (vocab_size, hidden_dim)).astype(np.float32)
        if pad_id is not None:
            self.weight[pad_id] = 0.0

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        return self.weight[token_ids]

    def mean_pool(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings.mean(axis=0)

class Word2VecSGNS:
    def __init__(self, dim: int = 16, neg_samples: int = 5, lr: float = 0.05, epochs: int = 15, seed: int = SEED):
        self.dim = dim
        self.neg_samples = neg_samples
        self.lr = lr
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        self.loss_history: List[float] = []

    def _build_vocab(self, corpus: Sequence[Sequence[str]]) -> None:
        counts: Counter = Counter()
        for sent in corpus:
            counts.update(sent)
        words = [w for w, _ in counts.most_common()]
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(words)
        self.W = (self.rng.uniform(-0.5 / self.dim, 0.5 / self.dim, (self.vocab_size, self.dim))).astype(np.float32)
        self.C = np.zeros((self.vocab_size, self.dim), dtype=np.float32)
        unigram = np.array([counts[self.idx2word[i]] for i in range(self.vocab_size)], dtype=np.float64) ** 0.75
        self._neg_dist = unigram / unigram.sum()

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    def _generate_pairs(self, corpus: Sequence[Sequence[str]], window: int) -> List[Tuple[int, int]]:
        pairs = []
        for sent in corpus:
            ids = [self.word2idx[w] for w in sent if w in self.word2idx]
            for i, center in enumerate(ids):
                r = self.rng.integers(1, window + 1)
                lo, hi = max(0, i - r), min(len(ids), i + r + 1)
                for j in range(lo, hi):
                    if j != i: pairs.append((center, ids[j]))
        return pairs

    def train(self, corpus: Sequence[Sequence[str]], window: int = 2) -> "Word2VecSGNS":
        self._build_vocab(corpus)
        pairs = self._generate_pairs(corpus, window)
        for _ in range(self.epochs):
            self.rng.shuffle(pairs)
            total_loss = 0.0
            for center, context in pairs:
                negs = self.rng.choice(self.vocab_size, size=max(0, self.neg_samples), p=self._neg_dist)
                v_c, u_o, u_k = self.W[center], self.C[context], self.C[negs]

                pos_score = self._sigmoid(u_o @ v_c)
                neg_scores = self._sigmoid(u_k @ v_c)

                total_loss += -np.log(pos_score + 1e-9) - np.sum(np.log(1 - neg_scores + 1e-9))

                grad_pos = pos_score - 1.0
                grad_neg = neg_scores
                grad_v = grad_pos * u_o + (grad_neg[:, None] * u_k).sum(axis=0)
                grad_uo = grad_pos * v_c
                grad_uk = grad_neg[:, None] * v_c[None, :]

                self.W[center] -= self.lr * grad_v
                self.C[context] -= self.lr * grad_uo
                self.C[negs] -= self.lr * grad_uk

            self.loss_history.append(total_loss / len(pairs))
        return self

# ---------------------------------------------------------------------------
# Helper: Cosine Similarity
# ---------------------------------------------------------------------------
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D arrays."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)


# ===========================================================================
# EXERCISE 1: Vocabulary-size sensitivity
# ===========================================================================
def exercise_1() -> None:
    print("\n" + "=" * 72)
    print("EXERCISE 1: Vocabulary-size sensitivity")
    print("=" * 72)

    # Load a small subset of 20 Newsgroups for speed
    print("Loading 20 Newsgroups (subset: sci.space)...")
    newsgroups = fetch_20newsgroups(subset='train', categories=['sci.space'], remove=('headers', 'footers', 'quotes'))
    corpus = [doc for doc in newsgroups.data if len(doc) > 50]  # Filter out empty/very short docs
    print(f"Loaded {len(corpus)} documents.")

    vocab_sizes = [20, 50, 100, 200]
    avg_tokens_list = []
    avg_token_lens_list = []

    for v_size in vocab_sizes:
        print(f"  Training BPE with vocab_size={v_size}...")
        tok = SimpleBPETokenizer(vocab_size=v_size).train(corpus)

        doc_token_counts = []
        doc_token_lengths = []

        for doc in corpus[:200]:  # Evaluate on first 200 docs for speed
            ids = tok.encode(doc)
            if not ids: continue
            doc_token_counts.append(len(ids))
            # Calculate string length of each token (excluding the </w> marker)
            for i in ids:
                token_str = tok.id_to_token.get(i, "")
                clean_str = token_str.replace("</w>", "")
                if clean_str:
                    doc_token_lengths.append(len(clean_str))

        avg_tokens_list.append(np.mean(doc_token_counts))
        avg_token_lens_list.append(np.mean(doc_token_lengths))
        print(f"    Avg tokens/doc: {avg_tokens_list[-1]:.1f} | Avg token len: {avg_token_lens_list[-1]:.1f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(vocab_sizes, avg_tokens_list, marker='o', color='steelblue')
    ax1.set_title("(a) Avg Tokens per Document")
    ax1.set_xlabel("Vocabulary Size")
    ax1.set_ylabel("Average Number of Tokens")
    ax1.grid(alpha=0.3)

    ax2.plot(vocab_sizes, avg_token_lens_list, marker='s', color='darkorange')
    ax2.set_title("(b) Avg Token Length (Characters)")
    ax2.set_xlabel("Vocabulary Size")
    ax2.set_ylabel("Average Characters per Token")
    ax2.grid(alpha=0.3)

    plt.suptitle("Exercise 1: BPE Vocabulary-Size Sensitivity", fontsize=14)
    plt.tight_layout()
    plt.savefig("exercise1_vocab_sensitivity.png", dpi=120)
    plt.close()
    print("-> Saved plot: exercise1_vocab_sensitivity.png")


# ===========================================================================
# EXERCISE 2: Polysemy demonstration
# ===========================================================================
def exercise_2() -> None:
    print("\n" + "=" * 72)
    print("EXERCISE 2: Polysemy demonstration")
    print("=" * 72)

    sent1 = "I sat by the river bank"
    sent2 = "I deposited cash at the bank"

    # Train a quick BPE tokenizer that knows these words
    tok = SimpleBPETokenizer(vocab_size=50).train([sent1, sent2] * 10)

    # 1. Random Static Embeddings
    print("\n1. Using Random Static Embeddings (EmbeddingLayer):")
    emb_layer = EmbeddingLayer(vocab_size=len(tok), hidden_dim=16, pad_id=tok.pad_id)

    ids1 = tok.encode(sent1)
    ids2 = tok.encode(sent2)

    vec1 = emb_layer.mean_pool(emb_layer.forward(np.array(ids1)))
    vec2 = emb_layer.mean_pool(emb_layer.forward(np.array(ids2)))
    sim_random = cosine_similarity(vec1, vec2)
    print(f"   Cosine similarity (sentence 1 vs 2): {sim_random:.4f}")

    # 2. Trained Word2Vec Static Embeddings
    print("\n2. Using Trained Word2Vec Embeddings (W matrix):")
    # Create a synthetic corpus to train Word2Vec so it learns some semantics
    poly_corpus = [
        ["river", "bank", "water", "sat"],
        ["deposited", "cash", "bank", "money"],
        ["river", "water", "flow"],
        ["cash", "money", "deposited"]
    ] * 50

    w2v = Word2VecSGNS(dim=16, neg_samples=5, epochs=20)
    w2v.train(poly_corpus)

    # Map sentence words to Word2Vec IDs
    w2v_ids1 = [w2v.word2idx[w] for w in sent1.lower().split() if w in w2v.word2idx]
    w2v_ids2 = [w2v.word2idx[w] for w in sent2.lower().split() if w in w2v.word2idx]

    w2v_vec1 = w2v.W[w2v_ids1].mean(axis=0)
    w2v_vec2 = w2v.W[w2v_ids2].mean(axis=0)
    sim_w2v = cosine_similarity(w2v_vec1, w2v_vec2)
    print(f"   Cosine similarity (sentence 1 vs 2): {sim_w2v:.4f}")

    # Extract just the "bank" vector to prove it's identical
    bank_vec = w2v.W[w2v.word2idx["bank"]]
    print(f"   - Vector for 'bank' in Sent 1: {np.round(bank_vec[:4], 4)}")
    print(f"   - Vector for 'bank' in Sent 2: {np.round(bank_vec[:4], 4)}")

    print("\nComment on why static embeddings fail polysemy:")
    print("Both the random EmbeddingLayer and the trained Word2Vec W matrix are 'static' lookups.")
    print("This means the word 'bank' maps to the EXACT SAME vector row whether it appears next")
    print("to 'river' or 'cash'. Because 'bank' is the only overlapping word, its single vector")
    print("artificially inflates the cosine similarity between the two sentences, even though their")
    print("meanings are entirely different. True polysemy resolution requires a contextual mechanism")
    print("(like Transformer Attention), where the output vector for 'bank' is dynamically updated")
    print("based on the surrounding tokens.")


# ===========================================================================
# EXERCISE 3: Negative-sample ablation
# ===========================================================================
def exercise_3() -> None:
    print("\n" + "=" * 72)
    print("EXERCISE 3: Negative-sample ablation")
    print("=" * 72)

    # Re-use the synthetic Scenario 3 corpus generator
    rng = np.random.default_rng(SEED)
    templates = [
        ["the", "CAT", "VERB", "on", "the", "MAT"],
        ["a", "CAT", "and", "a", "DOG", "are", "pets"],
        ["the", "VEHICLE", "VERB", "fast"],
    ]
    CAT = ["cat", "dog", "kitten", "puppy"]
    VEHICLE = ["car", "truck", "bus"]
    VERB = ["run", "sat", "jump", "walk", "move", "stop"]
    MAT = ["mat", "floor", "log", "road"]

    corpus = []
    for _ in range(400):
        tpl = list(rng.choice(templates))
        sent = []
        for t in tpl:
            if t == "CAT": sent.append(rng.choice(CAT))
            elif t == "VEHICLE": sent.append(rng.choice(VEHICLE))
            elif t == "VERB": sent.append(rng.choice(VERB))
            elif t == "MAT": sent.append(rng.choice(MAT))
            else: sent.append(t)
        corpus.append(sent)

    neg_samples_list = [0, 1, 5, 15]
    final_losses = []
    cat_dog_sims = []

    for ns in neg_samples_list:
        print(f"  Training Word2Vec with neg_samples={ns}...")
        model = Word2VecSGNS(dim=16, neg_samples=ns, epochs=10, lr=0.05)
        model.train(corpus)

        final_losses.append(model.loss_history[-1])

        # Compute similarity between 'cat' and 'dog'
        if "cat" in model.word2idx and "dog" in model.word2idx:
            v_cat = model.W[model.word2idx["cat"]]
            v_dog = model.W[model.word2idx["dog"]]
            sim = cosine_similarity(v_cat, v_dog)
        else:
            sim = 0.0
        cat_dog_sims.append(sim)
        print(f"    Final Loss: {final_losses[-1]:.4f} | Sim(cat, dog): {sim:.4f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(neg_samples_list, final_losses, marker='o', color='crimson')
    ax1.set_title("(a) Final Epoch Loss")
    ax1.set_xlabel("Number of Negative Samples")
    ax1.set_ylabel("Average Loss")
    ax1.set_xticks(neg_samples_list)
    ax1.grid(alpha=0.3)

    ax2.plot(neg_samples_list, cat_dog_sims, marker='s', color='forestgreen')
    ax2.set_title("(b) Cosine Similarity (cat vs. dog)")
    ax2.set_xlabel("Number of Negative Samples")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_xticks(neg_samples_list)
    ax2.grid(alpha=0.3)

    plt.suptitle("Exercise 3: Negative-Sample Ablation", fontsize=14)
    plt.tight_layout()
    plt.savefig("exercise3_neg_sampling_ablation.png", dpi=120)
    plt.close()
    print("-> Saved plot: exercise3_neg_sampling_ablation.png")

    print("\nWhat happens at neg_samples=0?")
    print("With 0 negative samples, the loss function only contains the positive term: -log σ(v_c · u_o).")
    print("To minimize this loss, the model simply pushes ALL input vectors (v_c) and output vectors (u_o)")
    print("to point in the exact same direction (e.g., all vectors become [1, 1, 1...]).")
    print("This causes the loss to drop artificially close to 0, but the embeddings collapse into a")
    print("single point/cluster. Consequently, the cosine similarity between 'cat' and 'dog' approaches 1.0,")
    print("meaning the model has failed to learn any distinct boundaries between concepts.")


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================
if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    print("\nAll exercises complete. Check console output and generated PNG files.")

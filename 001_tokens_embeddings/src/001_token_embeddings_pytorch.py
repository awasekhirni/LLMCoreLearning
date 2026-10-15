"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

exercises_pytorch.py
====================
Solutions for Chapter 2 exercises implemented using PyTorch.
    1. Vocabulary-size sensitivity
    2. Polysemy demonstration (Static nn.Embedding vs. Word2Vec static-trained)
    3. Negative-sample ablation in Word2Vec

Setting up the environment
python3 -m venv .
#installing the required packages
pip3 install torch numpy scikit-learn matplotlib
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

# ---------------------------------------------------------------------------
# Reproducibility (PyTorch & NumPy)
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
# Force CPU for deterministic, accessible execution across all machines
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Minimal Class Definitions
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

@dataclass
class SimpleBPETokenizer:
    """Pure Python BPE Tokenizer (remains unchanged as it is string-based)."""
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


class Word2VecSGNS(nn.Module):
    """
    Skip-gram with Negative Sampling implemented in PyTorch.
    Replaces manual NumPy gradients with torch.autograd.
    """
    def __init__(self, vocab_size: int, dim: int = 16, neg_samples: int = 5, lr: float = 0.05, epochs: int = 15, seed: int = SEED):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.neg_samples = neg_samples
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

        torch.manual_seed(seed)
        # W: Input (center) embeddings
        self.W = nn.Parameter(torch.randn(vocab_size, dim) * (0.5 / dim))
        # C: Output (context) embeddings
        self.C = nn.Parameter(torch.zeros(vocab_size, dim))

        self.loss_history: List[float] = []
        self._neg_dist: Optional[torch.Tensor] = None

    def _build_vocab(self, corpus: Sequence[Sequence[str]]) -> None:
        counts: Counter = Counter()
        for sent in corpus:
            counts.update(sent)
        words = [w for w, _ in counts.most_common()]
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(words)

        # Reinitialize parameters to match actual vocab size
        self.W = nn.Parameter(torch.randn(self.vocab_size, self.dim) * (0.5 / self.dim))
        self.C = nn.Parameter(torch.zeros(self.vocab_size, self.dim))

        unigram = torch.tensor([counts[self.idx2word[i]] for i in range(self.vocab_size)], dtype=torch.float64) ** 0.75
        self._neg_dist = unigram / unigram.sum()

    def _generate_pairs(self, corpus: Sequence[Sequence[str]], window: int) -> List[Tuple[int, int]]:
        pairs = []
        for sent in corpus:
            ids = [self.word2idx[w] for w in sent if w in self.word2idx]
            for i, center in enumerate(ids):
                r = torch.randint(1, window + 1, (1,)).item()
                lo, hi = max(0, i - r), min(len(ids), i + r + 1)
                for j in range(lo, hi):
                    if j != i: pairs.append((center, ids[j]))
        return pairs

    def forward(self, center_ids: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass computing SGNS loss for a batch."""
        v_c = self.W[center_ids]             # [batch, dim]
        u_o = self.C[context_ids]            # [batch, dim]

        pos_score = torch.sigmoid((v_c * u_o).sum(dim=1)) # [batch]
        pos_loss = -torch.log(pos_score + 1e-9)

        if self.neg_samples > 0:
            # Sample negatives using PyTorch multinomial
            neg_ids = torch.multinomial(self._neg_dist, num_samples=self.neg_samples * center_ids.size(0), replacement=True)
            neg_ids = neg_ids.view(center_ids.size(0), self.neg_samples) # [batch, neg]

            u_k = self.C[neg_ids]            # [batch, neg, dim]

            # Compute negative scores: v_c dot u_k
            # v_c: [batch, dim] -> unsqueeze to [batch, 1, dim] for bmm
            neg_scores = torch.sigmoid(torch.bmm(u_k, v_c.unsqueeze(2)).squeeze(2)) # [batch, neg]
            neg_loss = -torch.log(1 - neg_scores + 1e-9).sum(dim=1)

            loss = (pos_loss + neg_loss).mean()
        else:
            # If 0 negatives, loss is only positive
            loss = pos_loss.mean()

        return loss

    def train_model(self, corpus: Sequence[Sequence[str]], window: int = 2, batch_size: int = 512) -> "Word2VecSGNS":
        self._build_vocab(corpus)
        pairs = self._generate_pairs(corpus, window)
        if not pairs:
            raise ValueError("No training pairs generated.")

        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            # Shuffle pairs
            np.random.shuffle(pairs)
            total_loss = 0.0

            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]
                if not batch: continue

                centers = torch.tensor([p[0] for p in batch], dtype=torch.long, device=DEVICE)
                contexts = torch.tensor([p[1] for p in batch], dtype=torch.long, device=DEVICE)

                optimizer.zero_grad()
                loss = self.forward(centers, contexts)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch)

            self.loss_history.append(total_loss / len(pairs))
        return self


# ---------------------------------------------------------------------------
# Helper: Cosine Similarity (PyTorch)
# ---------------------------------------------------------------------------
def cosine_similarity_t(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D PyTorch tensors."""
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


# ===========================================================================
# EXERCISE 1: Vocabulary-size sensitivity
# ===========================================================================
def exercise_1() -> None:
    print("\n" + "=" * 72)
    print("EXERCISE 1: Vocabulary-size sensitivity")
    print("=" * 72)

    print("Loading 20 Newsgroups (subset: sci.space)...")
    newsgroups = fetch_20newsgroups(subset='train', categories=['sci.space'], remove=('headers', 'footers', 'quotes'))
    corpus = [doc for doc in newsgroups.data if len(doc) > 50]
    print(f"Loaded {len(corpus)} documents.")

    vocab_sizes = [20, 50, 100, 200]
    avg_tokens_list = []
    avg_token_lens_list = []

    for v_size in vocab_sizes:
        print(f"  Training BPE with vocab_size={v_size}...")
        tok = SimpleBPETokenizer(vocab_size=v_size).train(corpus)

        doc_token_counts = []
        doc_token_lengths = []

        for doc in corpus[:200]:
            ids = tok.encode(doc)
            if not ids: continue
            doc_token_counts.append(len(ids))
            for i in ids:
                token_str = tok.id_to_token.get(i, "")
                clean_str = token_str.replace("</w>", "")
                if clean_str:
                    doc_token_lengths.append(len(clean_str))

        avg_tokens_list.append(np.mean(doc_token_counts))
        avg_token_lens_list.append(np.mean(doc_token_lengths))
        print(f"    Avg tokens/doc: {avg_tokens_list[-1]:.1f} | Avg token len: {avg_token_lens_list[-1]:.1f}")

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
    print("EXERCISE 2: Polysemy demonstration (PyTorch Embeddings)")
    print("=" * 72)

    sent1 = "I sat by the river bank"
    sent2 = "I deposited cash at the bank"

    tok = SimpleBPETokenizer(vocab_size=50).train([sent1, sent2] * 10)

    # -----------------------------------------------------------
    # 1. Random Static Embeddings using torch.nn.Embedding
    # -----------------------------------------------------------
    print("\n1. Using Random Static PyTorch Embeddings (nn.Embedding):")
    # torch.nn.Embedding maps perfectly to the technical card's description
    emb_layer = nn.Embedding(num_embeddings=len(tok), embedding_dim=16, padding_idx=tok.pad_id)

    ids1 = torch.tensor(tok.encode(sent1), dtype=torch.long)
    ids2 = torch.tensor(tok.encode(sent2), dtype=torch.long)

    # mean_pool: average token embeddings across sequence length
    vec1 = emb_layer(ids1).mean(dim=0)
    vec2 = emb_layer(ids2).mean(dim=0)
    sim_random = cosine_similarity_t(vec1, vec2)
    print(f"   Cosine similarity (sentence 1 vs 2): {sim_random:.4f}")

    # -----------------------------------------------------------
    # 2. Trained Word2Vec Static Embeddings
    # -----------------------------------------------------------
    print("\n2. Using Trained Word2Vec Embeddings (W matrix):")
    poly_corpus = [
        ["river", "bank", "water", "sat"],
        ["deposited", "cash", "bank", "money"],
        ["river", "water", "flow"],
        ["cash", "money", "deposited"]
    ] * 50

    w2v = Word2VecSGNS(dim=16, neg_samples=5, epochs=20)
    w2v.train_model(poly_corpus)

    w2v_ids1 = torch.tensor([w2v.word2idx[w] for w in sent1.lower().split() if w in w2v.word2idx], dtype=torch.long)
    w2v_ids2 = torch.tensor([w2v.word2idx[w] for w in sent2.lower().split() if w in w2v.word2idx], dtype=torch.long)

    w2v_vec1 = w2v.W[w2v_ids1].mean(dim=0).detach()
    w2v_vec2 = w2v.W[w2v_ids2].mean(dim=0).detach()
    sim_w2v = cosine_similarity_t(w2v_vec1, w2v_vec2)
    print(f"   Cosine similarity (sentence 1 vs 2): {sim_w2v:.4f}")

    bank_vec = w2v.W[w2v.word2idx["bank"]].detach()
    print(f"   - Vector for 'bank' in Sent 1: {np.round(bank_vec[:4].numpy(), 4)}")
    print(f"   - Vector for 'bank' in Sent 2: {np.round(bank_vec[:4].numpy(), 4)}")

    print("\nComment on why static embeddings fail polysemy:")
    print("Both the PyTorch nn.Embedding layer and the trained Word2Vec W matrix are 'static' lookups.")
    print("This means the word 'bank' maps to the EXACT SAME vector row whether it appears next")
    print("to 'river' or 'cash'. Because 'bank' is the only overlapping word, its single vector")
    print("artificially inflates the cosine similarity between the two sentences, even though their")
    print("meanings are entirely different. True polysemy resolution requires a contextual mechanism")
    print("(like Transformer Self-Attention), where the output vector for 'bank' is dynamically updated")
    print("based on the surrounding tokens.")


# ===========================================================================
# EXERCISE 3: Negative-sample ablation
# ===========================================================================
def exercise_3() -> None:
    print("\n" + "=" * 72)
    print("EXERCISE 3: Negative-sample ablation (PyTorch)")
    print("=" * 72)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

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
        print(f"  Training PyTorch Word2Vec with neg_samples={ns}...")
        model = Word2VecSGNS(vocab_size=0, dim=16, neg_samples=ns, epochs=10, lr=0.05)
        model.train_model(corpus)

        final_losses.append(model.loss_history[-1])

        if "cat" in model.word2idx and "dog" in model.word2idx:
            v_cat = model.W[model.word2idx["cat"]].detach()
            v_dog = model.W[model.word2idx["dog"]].detach()
            sim = cosine_similarity_t(v_cat, v_dog)
        else:
            sim = 0.0
        cat_dog_sims.append(sim)
        print(f"    Final Loss: {final_losses[-1]:.4f} | Sim(cat, dog): {sim:.4f}")

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

    plt.suptitle("Exercise 3: Negative-Sample Ablation (PyTorch)", fontsize=14)
    plt.tight_layout()
    plt.savefig("exercise3_neg_sampling_ablation.png", dpi=120)
    plt.close()
    print("-> Saved plot: exercise3_neg_sampling_ablation.png")

    print("\nWhat happens at neg_samples=0?")
    print("With 0 negative samples, the PyTorch loss function only contains the positive term: -log σ(v_c · u_o).")
    print("Because there are no negative samples pushing words apart, the autograd engine updates the weights")
    print("such that ALL input vectors (W) and output vectors (C) are pushed to point in the exact same direction.")
    print("This causes the loss to drop artificially close to 0, but the embeddings collapse into a single point.")
    print("Consequently, the cosine similarity between 'cat' and 'dog' approaches 1.0, meaning the model has")
    print("failed to learn any distinct boundaries between concepts.")


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================
if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    print("\nAll PyTorch exercises complete. Check console output and generated PNG files.")

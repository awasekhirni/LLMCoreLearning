"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

advanced_concepts.py
====================
Supplementary relevant topics


Concepts Covered:
1. Batching & Padding (Why padding_idx matters in nn.Embedding)
2. Static vs. Contextualized Embeddings (How Attention solves polysemy)
3. Domain-Specific Tokenization (Code vs. Natural Language BPE)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Minimal Tokenizer Definition (Required for standalone execution)
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

@dataclass
class SimpleBPETokenizer:
    """A minimal, pure-Python BPE tokenizer for educational purposes."""
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

    def train(self, corpus: List[str]) -> "SimpleBPETokenizer":
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


# ===========================================================================
#  Class
# ===========================================================================
@dataclass
class AdvancedConceptDemos:
    """
    Encapsulates supplementary concepts.

    This class adheres to DRY principles by centralizing random seed management
    and tokenization utilities, ensuring that demonstrations are reproducible
    and isolated from production pipeline code.

    Attributes
    ----------
    seed : int
        The random seed used for all PyTorch and NumPy operations to ensure
        reproducible educational outputs.
    device : torch.device
        The device on which to allocate tensors (defaults to CPU for accessibility).
    """
    seed: int = 42
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        """Set the global random seeds upon initialization."""
        torch.manual_seed(self.seed)
        print(f"AdvancedConceptDemos initialized (Seed={self.seed}, Device={self.device})")

    # ------------------------------------------------------------------
    # Demo 1: Batching & Padding
    # ------------------------------------------------------------------
    def demo_batching_and_padding(self) -> None:
        """
        Illustrates why `padding_idx` is critical in `nn.Embedding`.

        Neural networks require fixed-shape tensors to process data in batches.
        Sentences of varying lengths are padded with a special `<PAD>` token (ID 0).
        If we don't specify `padding_idx`, the network will learn weights for this
        meaningless token, polluting the embedding space and gradients.
        """
        print("\n" + "=" * 72)
        print("DEMO 1: Batching, Padding, and the `padding_idx` Argument")
        print("=" * 72)

        # Train a tiny tokenizer
        tok = SimpleBPETokenizer(vocab_size=50).train(["the cat sat", "a dog ran", "birds fly high"])

        # Encode three sentences of different token lengths
        sent1 = tok.encode("the cat sat")      # len 3
        sent2 = tok.encode("a dog ran fast")   # len 4
        sent3 = tok.encode("birds fly")        # len 2

        # Manual Padding: Create a [batch_size=3, seq_len=4] tensor filled with PAD_ID
        max_len = max(len(s) for s in [sent1, sent2, sent3])
        padded_batch = torch.full((3, max_len), tok.pad_id, dtype=torch.long, device=self.device)

        for i, s in enumerate([sent1, sent2, sent3]):
            padded_batch[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)

        print("Padded Token ID Batch [Batch=3, Seq=4]:\n", padded_batch)

        # Create Embedding layer. padding_idx=0 forces the PAD row to remain zeros.
        emb_layer = nn.Embedding(
            num_embeddings=len(tok),
            embedding_dim=4,
            padding_idx=tok.pad_id
        ).to(self.device)

        # Lookup embeddings
        embeddings = emb_layer(padded_batch)
        print("\nEmbedding shape [Batch, Seq, Dim]:", tuple(embeddings.shape))

        # Prove that the PAD token results in a zero vector
        print("\nEmbedding for Sentence 3 (2 real words + 2 PADs):")
        print(embeddings[2])
        print("\nNotice: The last two rows (PAD tokens) are strictly 0.0.")
        print("This ensures PAD tokens contribute 0 to gradients and pooling operations.")


    # ------------------------------------------------------------------
    # Demo 2: Static vs. Contextualized Embeddings
    # ------------------------------------------------------------------
    def demo_contextualization(self) -> None:
        """
        Demonstrates how a Transformer layer takes static embeddings and makes
        them contextual, solving the polysemy problem (e.g., "river bank" vs "money bank").
        """
        print("\n" + "=" * 72)
        print("DEMO 2: Static vs. Contextualized Embeddings (Solving Polysemy)")
        print("=" * 72)

        # Tiny vocab and embeddings
        vocab = {"river": 0, "bank": 1, "money": 2, "sat": 3}
        emb_layer = nn.Embedding(num_embeddings=4, embedding_dim=8).to(self.device)

        # Sentence 1: "river bank"  | Sentence 2: "money bank"
        sent1 = torch.tensor([[0, 1]], device=self.device) # river bank
        sent2 = torch.tensor([[2, 1]], device=self.device) # money bank

        # 1. Static Embeddings (Before Attention)
        static_bank_1 = emb_layer(sent1)[0, 1] # Vector for "bank" in sent 1
        static_bank_2 = emb_layer(sent2)[0, 1] # Vector for "bank" in sent 2

        sim_static = F.cosine_similarity(static_bank_1.unsqueeze(0), static_bank_2.unsqueeze(0))
        print(f"Static similarity of 'bank' in two contexts: {sim_static.item():.4f} (Identical)")

        # 2. Contextualized Embeddings (After Transformer Attention)
        # A single layer of self-attention mixes the vectors based on surrounding words
        transformer = nn.TransformerEncoderLayer(
            d_model=8, nhead=2, batch_first=True
        ).to(self.device)

        # Pass embeddings through the transformer (no_grad for inference)
        with torch.no_grad():
            contextual_sent1 = transformer(emb_layer(sent1))
            contextual_sent2 = transformer(emb_layer(sent2))

        # Extract the NEW vector for "bank" (position 1)
        ctx_bank_1 = contextual_sent1[0, 1]
        ctx_bank_2 = contextual_sent2[0, 1]

        sim_ctx = F.cosine_similarity(ctx_bank_1.unsqueeze(0), ctx_bank_2.unsqueeze(0))
        print(f"Contextual similarity of 'bank' in two contexts: {sim_ctx.item():.4f} (Different!)")
        print("\nConcept: The Attention mechanism mixed 'river' and 'money' into the 'bank' vector,")
        print("creating distinct contextual representations for the same word.")


    # ------------------------------------------------------------------
    # Demo 3: Domain-Specific Tokenization
    # ------------------------------------------------------------------
    def demo_domain_specific_tokenization(self) -> None:
        """
        Shows how BPE trained on English vs. Code produces vastly different tokens.
        This illustrates why code models (like StarCoder) use custom tokenizers
        to preserve semantic whitespace efficiently.
        """
        print("\n" + "=" * 72)
        print("DEMO 3: Domain-Specific Tokenization (Code vs. Natural Language)")
        print("=" * 72)

        code_snippet = "def add(x, y):\n    return x + y"

        # Train one tokenizer on English text
        english_corpus = ["the cat sat on the mat", "a dog ran fast", "birds fly high"] * 10
        tok_eng = SimpleBPETokenizer(vocab_size=100).train(english_corpus)

        # Train another tokenizer on Python code
        code_corpus = [
            "def foo(x):\n    return x",
            "    if x == 1:\n        print(x)",
            "y = a + b\n    return y"
        ] * 10
        tok_code = SimpleBPETokenizer(vocab_size=100).train(code_corpus)

        # Encode the code snippet with both
        eng_ids = tok_eng.encode(code_snippet)
        code_ids = tok_code.encode(code_snippet)

        print(f"Code Snippet: {repr(code_snippet)}")
        print(f"Tokens using English BPE: {len(eng_ids)} tokens")
        print(f"Tokens using Code BPE:    {len(code_ids)} tokens")

        # Show the actual string pieces
        eng_pieces = [tok_eng.id_to_token[i].replace("</w>", " ") for i in eng_ids]
        code_pieces = [tok_code.id_to_token[i].replace("</w>", " ") for i in code_ids]

        print("\nEnglish BPE pieces (splits indentation into individual spaces/chars):")
        print(eng_pieces[:15])

        print("\nCode BPE pieces (merges '    ' and 'def' into single tokens):")
        print(code_pieces[:15])
        print("\nConcept: For code, whitespace is semantic. Code-BPE merges it into single tokens to save context window space.")


# ===========================================================================
# Main Execution
# ===========================================================================
if __name__ == "__main__":
    # Instantiate the class
    demos = AdvancedConceptDemos()

    # Run the three supplementary educational examples
    demos.demo_batching_and_padding()
    demos.demo_contextualization()
    demos.demo_domain_specific_tokenization()

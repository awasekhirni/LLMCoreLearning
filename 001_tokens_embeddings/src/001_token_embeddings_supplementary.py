"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

supplementary_tc02_concepts.py
==============================
Covers the remaining specific items from Technical Card 02:
1. Weight Tying (Input Embeddings <-> LM Head)
2. CLS Pooling vs. Mean Pooling
3. Byte-level fallback for Out-Of-Vocabulary (OOV) characters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use the SimpleBPETokenizer from the previous script for demonstration
# from exercises_pytorch import SimpleBPETokenizer

def demonstrate_weight_tying():
    """
    Dry Implementation Hint:
    "Consider tying the weights of the Embedding Layer (input) with the
    LM Head (output) to save memory and improve performance."
    """
    print("\n" + "=" * 72)
    print("SUPPLEMENTARY 1: Weight Tying (Input Embeddings <-> LM Head)")
    print("=" * 72)

    vocab_size = 1000
    hidden_dim = 64

    # 1. The Input Embedding Layer
    # shape: [vocab_size, hidden_dim]
    embedding_layer = nn.Embedding(vocab_size, hidden_dim)

    # 2. The Output LM Head (projects hidden states back to vocab probabilities)
    # Normally this would be: nn.Linear(hidden_dim, vocab_size, bias=False)
    # which creates a NEW weight matrix of shape [vocab_size, hidden_dim]
    lm_head_unTIED = nn.Linear(hidden_dim, vocab_size, bias=False)

    # 3. Tying the weights: We force the LM Head to use the EXACT same weight matrix
    # as the Embedding layer, just transposed.
    lm_head_tied = nn.Linear(hidden_dim, vocab_size, bias=False)
    lm_head_tied.weight = embedding_layer.weight  # <--- THE MAGIC LINE

    # Verify they share the same memory
    embedding_layer.weight.data[0, 0] = 99.99
    print(f"Embedding weight[0,0] changed to: {embedding_layer.weight.data[0,0]}")
    print(f"LM Head weight[0,0] automatically changed to: {lm_head_tied.weight.data[0,0]}")
    print("Memory saved: We now store one [vocab_size, hidden_dim] matrix instead of two.")


def demonstrate_pooling_strategies():
    """
    Text/Sentence Embeddings:
    "Generation methods: Mean Pooling ... CLS Pooling"
    """
    print("\n" + "=" * 72)
    print("SUPPLEMENTARY 2: CLS Pooling vs. Mean Pooling")
    print("=" * 72)

    batch_size = 1
    seq_len = 5
    hidden_dim = 8

    # Simulate the output of a Transformer block for a sequence
    # Batch shape: [batch_size, seq_len, hidden_dim]
    token_embeddings = torch.randn(batch_size, seq_len, hidden_dim)

    # Strategy A: Mean Pooling (Average all token vectors)
    # Ignores padding if a mask is provided, but here we assume no padding
    mean_pooled = token_embeddings.mean(dim=1) # shape: [batch_size, hidden_dim]
    print(f"Mean Pooled Vector shape: {mean_pooled.shape}")

    # Strategy B: CLS Pooling (Take the vector of the FIRST token)
    # In BERT, the first token is always [CLS], which is trained to represent the whole sentence
    cls_pooled = token_embeddings[:, 0, :] # shape: [batch_size, hidden_dim]
    print(f"CLS Pooled Vector shape: {cls_pooled.shape}")
    print("Concept: CLS pooling relies on the model's attention mechanism to aggregate all sentence meaning into token 0.")


def demonstrate_byte_fallback():
    """
    TC02 Module 1 (Byte Tokenization):
    "Tokenization-free encoding using raw UTF-8 bytes. Guaranteed no OOV tokens.
    Used as fallback in BPE"
    """
    print("\n" + "=" * 72)
    print("SUPPLEMENTARY 3: Byte-Level Fallback for OOV characters")
    print("=" * 72)

    text = "Hello world 🚀" # The rocket emoji is likely OOV in a small BPE vocab

    # Train a tiny BPE that has never seen the emoji
    tok = SimpleBPETokenizer(vocab_size=30).train(["hello world cat dog"])

    # 1. Standard BPE encoding (Falls back to UNK)
    standard_ids = tok.encode(text)
    print(f"Standard BPE IDs: {standard_ids}")
    print(f"Contains UNK ({tok.unk_id})? {tok.unk_id in standard_ids}")

    # 2. Byte-level fallback encoding
    # If a pre-token contains an OOV character, we encode the WHOLE word to UTF-8 bytes
    byte_ids = []
    for word in text.split():
        word_with_marker = word + "</w>"
        # Try standard BPE merges
        syms = tok._apply_merges(list(word) + ["</w>"])

        # Check if any symbol is not in vocab (OOV)
        if any(s not in tok.vocab for s in syms):
            # FALLBACK: Convert the raw word string to UTF-8 bytes
            word_bytes = word.encode('utf-8')
            print(f"  Word '{word}' is OOV. Falling back to bytes: {list(word_bytes)}")
            # Map bytes to IDs (typically offset by a base number, e.g., 256 + byte_val)
            # For this demo, we just append the raw byte values
            byte_ids.extend(list(word_bytes))
        else:
            byte_ids.extend([tok.vocab[s] for s in syms])

    print(f"Byte-Fallback IDs: {byte_ids}")
    print("Concept: Modern tokenizers like GPT-2/Llama use Byte-level BPE, meaning their base vocabulary is the 256 possible bytes, mathematically guaranteeing UNK is never needed.")


if __name__ == "__main__":
    demonstrate_weight_tying()
    demonstrate_pooling_strategies()
    demonstrate_byte_fallback()

"""

# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
eli5_nope.py
============
A 20‑line demonstration of NoPE using plain language print statements.
Run this **after** reading the main module if you want a gentle recap.
"""

import torch
import torch.nn.functional as F

# --- Toy sequence: four tokens -------------------------------------------------
tokens = torch.tensor([[2, 5, 1, 7]])          # shape (1, 4)

# --- Step 1: embed tokens (NO position vector is added) -----------------------
emb = torch.randn(1, 4, 8)                     # pretend embeddings
print("Step 1 – Embeddings shape:", emb.shape)
print("         Notice: there is NO position embedding added here.")

# --- Step 2: causal attention --------------------------------------------------
q = k = v = emb
attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("\nStep 2 – Causal attention output shape:", attn.shape)
print("         Token 0 can only see token 0.")
print("         Token 1 can see tokens 0 and 1.")
print("         The mask (not a position code) creates this order.")

# --- Step 3: why length generalisation works -----------------------------------
print("\nStep 3 – Now try a longer sequence (6 tokens).")
long_tokens = torch.tensor([[2, 5, 1, 7, 3, 9]])
long_emb = torch.randn(1, 6, 8)
long_attn = F.scaled_dot_product_attention(long_emb, long_emb, long_emb, is_causal=True)
print("         Longer attention output shape:", long_attn.shape)
print("         The model did not need any new position table –")
print("         the same causal rule just applies to more tokens!")
print("\nThat is NoPE: no position badges, just the mask.")

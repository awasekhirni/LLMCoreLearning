"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.

eli5_walkthrough.py
===================
A narrator-style walkthrough that prints every step in plain English,
as if explaining to a five-year-old. Uses the same classes as the main module.
"""

import numpy as np
from tokens_and_embeddings import (
    SimpleBPETokenizer, EmbeddingLayer, Word2VecSGNS, SEED
)

print("\n🐱  HELLO!  We are going to turn words into numbers a computer loves.\n")

# --- STEP 1: Build a tiny dictionary of word-pieces (the tokenizer) ---
print("STEP 1: Show the computer some sentences so it can learn word-pieces.")
story = ["the cat sat", "the dog ran", "the cat ran", "the dog sat"]
tokenizer = SimpleBPETokenizer(vocab_size=30).train(story)
print(f"   The computer learned {len(tokenizer)} little pieces.\n")

# --- STEP 2: Cut a new sentence into pieces and give each a sticker-number ---
print("STEP 2: Take a new sentence and cut it into pieces.")
sentence = "the cat sat"
ids = tokenizer.encode(sentence, add_bos=True, add_eos=True)
print(f"   Sentence: '{sentence}'")
print(f"   Sticker numbers: {ids}")
print(f"   Pieces back:     {tokenizer.decode(ids)}\n")

# --- STEP 3: Look up the fingerprint for each sticker-number ---
print("STEP 3: Each sticker-number gets a 'fingerprint' (a tiny list of numbers).")
emb = EmbeddingLayer(vocab_size=len(tokenizer), hidden_dim=4, pad_id=tokenizer.pad_id)
fingerprints = emb.forward(np.array(ids))
print(f"   Fingerprints shape: {fingerprints.shape}")
print(f"   First fingerprint: {fingerprints[0]}\n")

# --- STEP 4: Play the 'guess the neighbor' game (Word2Vec) ---
print("STEP 4: Play the neighbor-guessing game so fingerprints learn MEANING.")
corpus = [["the", "cat", "sat"], ["the", "dog", "sat"],
          ["the", "cat", "ran"], ["the", "dog", "ran"]] * 50
model = Word2VecSGNS(vocab_size=0, dim=8, epochs=10, learning_rate=0.1)
model.train(corpus, window=1)
print("   After the game, 'cat' and 'dog' should have SIMILAR fingerprints!")
for w in ["cat", "dog"]:
    print(f"   {w}: {model.most_similar(w, top_k=2)}")
print("\n🎉  Done! Numbers now understand a little bit of what words MEAN.")

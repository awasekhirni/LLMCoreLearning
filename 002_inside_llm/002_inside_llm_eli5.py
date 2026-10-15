"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.
inside_llm_eli5_walkthrough.py
============================
A *tiny* LLM that runs in one screen, narrated for a five-year-old.

Vocabulary: 5 pretend words, encoded as numbers 0..4.
    0 = "cat", 1 = "sat", 2 = "on", 3 = "mat", 4 = "."
Goal: feed the model "[cat, sat, on]" and watch it guess the next token.
# 1. Install packages
pip install numpy pandas matplotlib scikit-learn
"""
import numpy as np

# ---------- our entire "model" in 10 lines of plain NumPy ----------

def tiny_softmax(x):
    """Turn scores into percentages that add up to 100%."""
    e = np.exp(x - np.max(x))
    return e / e.sum()

def tiny_attention(Q, K, V):
    """Decide how much each word matters and mix the V's accordingly."""
    scores = Q @ K.T                  # how friendly are Q and K with each other?
    weights = tiny_softmax(scores)    # turn friendliness into percentages
    return weights @ V, weights

def main():
    np.random.seed(42)   # so the story is identical every time we run it

    print("\n🌟 HELLO! I am a tiny pretend-LLM. Let's see what happens inside me.\n")

    # ---------- STEP 1: our 5 pretend words ----------
    words = ["cat", "sat", "on", "mat", "."]
    print(f"📚 My dictionary has {len(words)} words: {words}")
    print(f"   Each word is just a number: 0={words[0]}, 1={words[1]}, ...")

    # ---------- STEP 2: turn words into "vectors" (lists of numbers) ----------
    # Each word gets a 3-number "personality" (3 dims so we can print it easily).
    embedding_table = np.random.randn(5, 3)
    print(f"\n🪑 Each word gets a 3-number personality (an 'embedding vector'):")
    for i, w in enumerate(words):
        print(f"   {w:4s} -> {embedding_table[i].round(2)}")

    # ---------- STEP 3: read a 3-word prompt ----------
    prompt_ids = np.array([0, 1, 2])    # "cat sat on"
    print(f"\n📖 You showed me the prompt: {[words[i] for i in prompt_ids]}")
    print(f"   As numbers: {prompt_ids.tolist()}")

    vectors = embedding_table[prompt_ids]   # (3, 3)
    print(f"\n🪄 I turned each word into its personality vector:")
    print(vectors.round(2))

    # ---------- STEP 4: make Query, Key, Value from each vector ----------
    # Use a random 3x3 matrix as our "W" (in a real LLM this is learned).
    W = np.random.randn(3, 3)
    Q = vectors @ W   # what each word is *looking for*
    K = vectors @ W   # what each word *offers*
    V = vectors @ W   # what each word *carries forward*
    print(f"\n🔍 Each word now wears three hats: Q (question), K (key), V (value).")
    print(f"   Q for 'cat': {Q[0].round(2)}")

    # ---------- STEP 5: attention — who cares about whom? ----------
    out, weights = tiny_attention(Q, K, V)
    print(f"\n🤝 Attention weights (who listens to whom):")
    row_labels = [words[i] for i in prompt_ids]
    for i, rl in enumerate(row_labels):
        print(f"   '{rl}' listens -> " +
              ", ".join(f"'{row_labels[j]}'={weights[i, j]:.2f}"
                        for j in range(len(row_labels))))

    # ---------- STEP 6: final score for EACH possible next word ----------
    # In a real LLM this is the "LM head" — a linear projection to vocab size.
    head_W = np.random.randn(3, 5)   # (hidden, vocab)
    last_vector = out[-1]            # we only generate the NEXT token from the last position
    logits = last_vector @ head_W    # raw scores, one per word in the dictionary
    print(f"\n🔢 Raw scores (logits) for the next word:")
    for w, l in zip(words, logits):
        print(f"   {w:4s}: {l:+.2f}")

    # ---------- STEP 7: turn scores into percentages (softmax) ----------
    probs = tiny_softmax(logits)
    print(f"\n📊 Percentages (softmax) for the next word:")
    for w, p in zip(words, probs):
        bar = "█" * int(p * 40)
        print(f"   {w:4s}: {p*100:5.1f}%  {bar}")

    # ---------- STEP 8: greedy pick — the highest percentage wins ----------
    best = int(np.argmax(probs))
    print(f"\n🏆 The winner is: '{words[best]}' (id {best}) — that's my guess for the next word!")
    print(f"   New sentence so far: {' '.join(words[i] for i in prompt_ids)} {words[best]}\n")

if __name__ == "__main__":
    main()

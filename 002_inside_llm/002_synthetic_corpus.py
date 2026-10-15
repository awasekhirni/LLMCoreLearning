# synthetic_corpus.py
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
import numpy as np

def make_toy_corpus(vocab_size: int = 20, length: int = 64, seed: int = 42) -> np.ndarray:
    """
    Generate a deterministic 1-D array of token IDs for exercises.

    Why synthetic:
        - Reproducible across every learner's machine.
        - Tiny enough to print on a single screen.
        - Avoids depending on a 3rd-party download at lesson time.
    """
    rng = np.random.default_rng(seed)
    # Bias toward lower IDs to create "frequent words" vs "rare words"
    probs = np.exp(-np.arange(vocab_size) / 5)
    probs /= probs.sum()
    return rng.choice(vocab_size, size=length, p=probs).astype(np.int32)

if __name__ == "__main__":
    corpus = make_toy_corpus()
    print(corpus)  # e.g. [0 1 0 2 1 0 3 1 ...]

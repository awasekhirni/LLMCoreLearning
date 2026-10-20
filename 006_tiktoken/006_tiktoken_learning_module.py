"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#
tiktoken_learning_module.py
===========================

A self-contained learning module for OpenAI's Tiktoken tokenizer.



Purpose
-------
Teach Tiktoken (OpenAI BPE variant) through:
  1. A dataclass-based adapter that exposes every metric a practitioner
     needs (token_count, fertility, tokens_per_char, compression_ratio).
  2. Happy-path encoding/decoding.
  3. Edge cases: empty input, wrong types, special-token literals,
     non-Latin scripts, emoji, and batch encoding.
  4. An ELI5 walkthrough with toy data.
  5. Visualisations of pre-tokenization and token-count comparisons.

Dependencies
------------
Python 3.10+, tiktoken, matplotlib, numpy, pandas.

    pip3 install tiktoken matplotlib numpy pandas

Run
---
    python3 tiktoken_learning_module.py

Figures saved to `fig_tiktoken_pretokenize.png` and `fig_tiktoken_fertility.png`.
"""

from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GUTENBERG_URL = "https://www.gutenberg.org/files/100/100-0.txt"

EMBEDDED_SHAKESPEARE: str = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die-to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream-ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause-there's the respect
That makes calamity of so long life.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first, the infant,
Mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth.
""".strip()

MULTILINGUAL_SAMPLES: List[Tuple[str, str]] = [
    ("English (seen)", "hello world how are you today"),
    ("Spanish (seen)", "hola mundo como estas hoy"),
    ("Chinese (mixed)", "你好世界宇宙无限"),
    ("Code (seen)", "def foo x return x plus one"),
    ("Emoji (unseen)", "hello 😊🚀 world"),
    ("Hindi (unseen)", "नमस्ते दुनिया"),
]


# ======================================================================
# Shared utilities
# ======================================================================
def load_shakespeare_corpus(
    max_chars: Optional[int] = 20_000,
    prefer_download: bool = True,
    url: str = GUTENBERG_URL,
    timeout_s: float = 10.0,
) -> str:
    """
    Load a Shakespeare corpus, preferring the full Gutenberg text.

    Parameters
    ----------
    max_chars : int or None
        Truncate to this many characters (keeps demos fast).
    prefer_download : bool
        Try the network first; fall back to the embedded excerpt on any error.
    url : str
        Gutenberg URL for *The Complete Works of William Shakespeare*.
    timeout_s : float
        Network timeout – never hang a live demo.

    Returns
    -------
    str
        Non-empty raw text.
    """
    text: Optional[str] = None
    if prefer_download:
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                text = resp.read().decode("utf-8")
        except Exception as exc:
            print(f"[Info] Download failed ({exc}); using embedded excerpt.")
    if text is None:
        text = EMBEDDED_SHAKESPEARE
    return text[:max_chars] if max_chars is not None else text


def section(title: str) -> None:
    """Print a visually distinct section header."""
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


# ======================================================================
# Core adapter – every metric lives here once (DRY)
# ======================================================================
@dataclass
class TiktokenAdapter:
    """
    Uniform façade over a Tiktoken encoding.

    Exposes:
      - encode / decode / encode_batch
      - token_count
      - fertility (tokens / whitespace word)
      - tokens_per_char (tokens / Unicode code point)
      - compression_ratio (UTF-8 bytes / token)
      - vocab_size

    The dataclass makes it trivial to instantiate:
        adapter = TiktokenAdapter(encoding_name="o200k_base")
    """

    encoding_name: str = "o200k_base"
    allowed_special: Union[str, set] = field(default_factory=set)
    disallowed_special: Union[str, set] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Load the tiktoken encoding and expose it as `self.enc`."""
        self.enc = tiktoken.get_encoding(self.encoding_name)

    # ------------------------------------------------------------------
    # Core encoding / decoding
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """Encode `text` to a list of integer token IDs."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        return self.enc.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )

    def decode(self, ids: Sequence[int]) -> str:
        """Inverse of `encode`. Returns a UTF-8 string."""
        if not isinstance(ids, (list, tuple)):
            raise TypeError(f"Expected list/tuple of ints, got {type(ids).__name__}")
        return self.enc.decode(ids)

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        """Encode many strings in one Rust call (faster than a Python loop)."""
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("All items in `texts` must be strings.")
        return self.enc.encode_batch(
            texts,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def token_count(self, text: str) -> int:
        """Number of tokens produced by encoding `text`."""
        return len(self.encode(text))

    def fertility(self, corpus: str) -> float:
        """
        Tokens per whitespace-separated word.

        Words = `corpus.split()`. For space-less scripts (Chinese),
        use `tokens_per_char` instead.
        """
        n_words = len(corpus.split())
        return self.token_count(corpus) / n_words if n_words else 0.0

    def tokens_per_char(self, text: str) -> float:
        """Tokens per Unicode code point – script-agnostic."""
        return self.token_count(text) / len(text) if text else 0.0

    def compression_ratio(self, text: str) -> float:
        """
        UTF-8 bytes per token. Higher = better (fewer tokens for same bytes).

        Uses UTF-8 *bytes* (not `len(text)`) so the metric is comparable
        across scripts – a CJK character is 3 bytes.
        """
        n_tokens = self.token_count(text)
        return len(text.encode("utf-8")) / n_tokens if n_tokens else 0.0

    @property
    def vocab_size(self) -> int:
        """Actual vocabulary size of the loaded encoding."""
        return self.enc.n_vocab


# ======================================================================
# Happy-path scenario
# ======================================================================
def happy_path_demo() -> None:
    """Basic encode → decode round-trip and metric inspection."""
    section("HAPPY PATH – encode, decode, metrics")

    adapter = TiktokenAdapter(encoding_name="o200k_base")
    text = "Hello, world! Tiktoken is fast 🚀"

    ids = adapter.encode(text)
    decoded = adapter.decode(ids)

    print(f"Original : {text!r}")
    print(f"Token IDs: {ids}")
    print(f"Decoded  : {decoded!r}")
    print(f"Round-trip lossless: {decoded == text}")
    print(f"Token count        : {len(ids)}")
    print(f"Vocab size         : {adapter.vocab_size:,}")
    print(f"Fertility          : {adapter.fertility(text):.3f} tokens/word")
    print(f"Tokens per char    : {adapter.tokens_per_char(text):.3f}")
    print(f"Compression ratio  : {adapter.compression_ratio(text):.2f} bytes/token")


# ======================================================================
# Edge cases and failure scenarios
# ======================================================================
def edge_case_empty_string() -> None:
    """Empty string → empty token list (not an error)."""
    section("EDGE CASE – empty string")
    adapter = TiktokenAdapter()
    assert adapter.encode("") == []
    assert adapter.decode([]) == ""
    print("✓ Empty string encodes to [] and decodes back to ''.")


def edge_case_wrong_type() -> None:
    """Passing a non-string raises a clear TypeError."""
    section("EDGE CASE – wrong input type")
    adapter = TiktokenAdapter()
    try:
        adapter.encode(123)          # type: ignore[arg-type]
    except TypeError as exc:
        print(f"✓ TypeError raised as expected: {exc}")


def edge_case_special_token_literal() -> None:
    """
    Tiktoken's `encode` raises on literal special-token strings by default.

    This is a security feature: user text containing ``<|endoftext|>``
    should not silently become a special token.
    """
    section("EDGE CASE – literal special-token string")
    adapter = TiktokenAdapter(encoding_name="cl100k_base")
    text = "This is a test <|endoftext|> with a special token."

    try:
        adapter.encode(text)
    except ValueError as exc:
        print(f"✓ ValueError raised (default disallowed_special): {exc}")

    # Safe path: treat special tokens as ordinary text.
    safe_ids = adapter.enc.encode(text, disallowed_special=())
    print(f"✓ With disallowed_special=(), encoded to {len(safe_ids)} tokens without error.")


def edge_case_non_latin_and_emoji() -> None:
    """Byte-level fallback handles any Unicode character losslessly."""
    section("EDGE CASE – non-Latin scripts and emoji")
    adapter = TiktokenAdapter(encoding_name="o200k_base")
    for text in ["नमस्ते दुनिया", "你好世界", "😊🚀"]:
        ids = adapter.encode(text)
        decoded = adapter.decode(ids)
        print(f"{text!r:20s} → {len(ids):3d} tokens | lossless: {decoded == text}")


def edge_case_extreme_values() -> None:
    """Very long text and very large token IDs."""
    section("EDGE CASE – extreme values")
    adapter = TiktokenAdapter()

    long_text = "word " * 10_000
    ids = adapter.encode(long_text)
    print(f"Encoded {len(long_text):,} characters → {len(ids):,} tokens "
          f"(no overflow, memory-safe).")

    # Decoding an ID outside the vocabulary: tiktoken handles gracefully?
    try:
        adapter.decode([adapter.vocab_size + 10])
    except Exception as exc:
        print(f"✓ Out-of-vocab ID raises a clear exception: {type(exc).__name__}")


def edge_case_batch_encoding() -> None:
    """`encode_batch` must produce identical results to a Python loop."""
    section("EDGE CASE – batch encoding consistency")
    adapter = TiktokenAdapter()
    texts = ["hello", "world", "tiktoken 🚀"]
    batch_ids = adapter.encode_batch(texts)
    loop_ids = [adapter.encode(t) for t in texts]
    assert batch_ids == loop_ids
    print("✓ encode_batch == [encode(t) for t in texts]")


# ======================================================================
# ELI5 walkthrough (toy data, plain-language narration)
# ======================================================================
def eli5_walkthrough() -> None:
    """
    A simplified, commented walkthrough for a five-year-old.

    We use a tiny toy vocabulary and narrate every step.
    """
    section("ELI5 WALKTHROUGH – how Tiktoken cuts text into puzzle pieces")

    # Step 1: pretend we have a very small tokenizer.
    # (In reality, o200k_base has 200k pieces – we use a toy example for clarity.)
    print("Imagine Tiktoken has a box of puzzle pieces it already knows:")
    toy_pieces = [" the", " cat", " sat", " on", " mat", "ing", "s", "🚀"]
    print(f"  {toy_pieces}\n")

    # Step 2: a sentence arrives.
    sentence = "the cat sat on the mat"
    print(f"Your sentence: {sentence!r}")
    print("Tiktoken first uses a fixed rule to split it into words:")
    words = sentence.split()
    print(f"  Words: {words}\n")

    # Step 3: for each word, try the biggest puzzle piece first.
    print("For each word, it looks for the biggest piece it knows:")
    for word in words:
        for piece in toy_pieces:
            if piece.strip() == word:
                print(f"  {word!r:8s} → found piece {piece!r}")
                break
        else:
            print(f"  {word!r:8s} → no exact piece; it would fall back to smaller pieces or raw bytes.")

    # Step 4: the real Tiktoken does this with integer IDs.
    print("\nIn the real Tiktoken, each piece is a number (token ID).")
    print("So the model sees a list of numbers, not letters.")
    print("This is why token counting is different from character counting!")


# ======================================================================
# Visualization
# ======================================================================
def visualize_pretokenization(
    save_to: Union[str, Path] = "fig_tiktoken_pretokenize.png",
) -> None:
    """
    Show the cl100k_base regex pattern visually splitting a sentence.

    Tiktoken first applies a regex (the `pat_str`) to cut text into
    chunks. We approximate that split here to make the concept tangible.
    """
    section("VISUALIZATION – regex pre-tokenization (cl100k_base style)")

    # Simplified version of the cl100k_base pattern for demonstration.
    # The real pattern is more complex; this captures the spirit:
    # contractions | words | 1-3 digit numbers | punctuation/whitespace.
    pattern = (
        r"'(?i:[sdmt]|ll|ve|re)"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+"
        r"|\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    )

    text = "Hello, world! I can't believe it's 2025 🚀"
    # Python's `re` does not support \p{L} without the `regex` module.
    # For teaching, we fall back to a simpler split that mirrors the
    # spirit: contractions, words, 1-3 digit groups, punctuation.
    simple_pattern = r"'(?i:[sdmt]|ll|ve|re)|[^\W\d_]+|\d{1,3}|[^\w\s]|\s+"
    chunks = [m.group() for m in re.finditer(simple_pattern, text, re.UNICODE)]
    chunks = [c for c in chunks if c.strip() or c in (" ",)]  # keep meaningful chunks

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")
    ax.set_title(
        "Tiktoken pre-tokenization: the regex cuts text into chunks before BPE merges",
        fontsize=11,
    )

    x_pos = 0.02
    for i, chunk in enumerate(chunks):
        color = plt.cm.Paired(i % 8)
        ax.text(
            x_pos, 0.5, chunk,
            fontsize=12, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
            transform=ax.transAxes,
        )
        x_pos += max(0.08, len(chunk) * 0.025)

    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"Figure saved: {Path(save_to).resolve()}")


def visualize_fertility_comparison(
    save_to: Union[str, Path] = "fig_tiktoken_fertility.png",
) -> None:
    """
    Bar chart: token count for the same sentence across languages and encodings.

    Shows why o200k_base is fairer to non-Latin scripts than cl100k_base.
    """
    section("VISUALIZATION – fertility across languages and encodings")

    samples = MULTILINGUAL_SAMPLES
    labels = [s[0] for s in samples]
    texts = [s[1] for s in samples]

    cl = TiktokenAdapter(encoding_name="cl100k_base")
    o2 = TiktokenAdapter(encoding_name="o200k_base")

    cl_counts = [cl.token_count(t) for t in texts]
    o2_counts = [o2.token_count(t) for t in texts]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, cl_counts, width, label="cl100k_base (GPT-4)", color="steelblue")
    bars2 = ax.bar(x + width/2, o2_counts, width, label="o200k_base (GPT-4o)", color="seagreen")

    ax.set_ylabel("Number of tokens")
    ax.set_title("Token count per sentence: cl100k_base vs o200k_base")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.show()
    print(f"Figure saved: {Path(save_to).resolve()}")


# ======================================================================
# Expected outcomes (printed for verification)
# ======================================================================
def print_expected_outcomes() -> None:
    """Print the exact outputs the learner should observe."""
    section("EXPECTED OUTCOMES (verify your run)")

    adapter = TiktokenAdapter(encoding_name="o200k_base")
    text = "Hello, world! Tiktoken is fast 🚀"
    ids = adapter.encode(text)

    print("Happy path:")
    print(f"  - Round-trip lossless: True")
    print(f"  - Token count for {text!r}: {len(ids)} (varies by encoding/version)")
    print(f"  - o200k_base vocab size: {adapter.vocab_size:,} (expected ~200,000)")

    print("\nEdge cases:")
    print("  - Empty string → [] and '' (no error)")
    print("  - Non-str input → TypeError")
    print("  - Literal '<|endoftext|>' → ValueError by default")
    print("  - Emoji and Hindi decode back losslessly (byte-level fallback)")

    print("\nBatch consistency:")
    print("  - encode_batch(texts) == [encode(t) for t in texts]")


# ======================================================================
# Driver
# ======================================================================
def main() -> None:
    """Run all demonstrations in order."""
    pd.set_option("display.width", 200)

    happy_path_demo()
    edge_case_empty_string()
    edge_case_wrong_type()
    edge_case_special_token_literal()
    edge_case_non_latin_and_emoji()
    edge_case_extreme_values()
    edge_case_batch_encoding()

    eli5_walkthrough()
    visualize_pretokenization()
    visualize_fertility_comparison()
    print_expected_outcomes()

    section("DONE")
    print("Figures: fig_tiktoken_pretokenize.png, fig_tiktoken_fertility.png")


if __name__ == "__main__":
    main()

"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
Exercise 1 – Fertility across encodings.
Using TiktokenAdapter, compute the fertility of the embedded Shakespeare excerpt for cl100k_base, o200k_base, and p50k_base. Which encoding gives the lowest fertility on English? Plot a bar chart.

Hint: Instantiate three adapters and call .fertility(corpus) on each. The p50k_base encoding is older and has a 50k vocabulary; you may need to handle the possibility that it is not installed by falling back gracefully.

Exercise 2 – Safe encoding of user input.
Write a function safe_encode(text: str) -> List[int] that never raises on special-token literals. It should treat <|endoftext|> as ordinary text. Verify that safe_encode("<|endoftext|>") returns a non-empty list and that decode(safe_encode(s)) recovers the original string.

Hint: Pass disallowed_special=() to enc.encode. Be careful: allowed_special is separate; leaving it empty is fine.

Exercise 3 – Batch vs loop speed.
Create a list of 1,000 short sentences. Measure the wall-clock time to encode them with a Python for loop calling adapter.encode versus one call to adapter.encode_batch. Report the speedup factor. Use time.perf_counter().

Hint: encode_batch releases the GIL and processes in Rust; the speedup should be noticeable. Do a warm-up run first to avoid timing the one-time encoding-load cost.

"""

"""
tiktoken_exercise_solutions.py
==============================

Complete solutions to Exercises 1–3 from the Tiktoken Learning Module.

# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.

Overview
--------
This file provides three dataclass-based solution classes, each solving
one exercise from `tiktoken_learning_module.py`:

  Exercise 1 – Fertility across encodings (cl100k, o200k, p50k)
  Exercise 2 – Safe encoding of user input (no ValueError on special tokens)
  Exercise 3 – Batch vs loop speed (measure the speedup factor)

Design principles
-----------------
- --DRY-- – every metric lives on `TiktokenAdapter`; every solution class
  reuses the same adapter. Nothing is duplicated.
- --Dataclasses-- – every stateful concept is a `@dataclass` so it is
  trivial to instantiate, inspect, and test.
- --Graceful degradation-- – `p50k_base` (or any encoding) may not be
  installed on the learner's machine. We detect and skip it, printing a
  clear message instead of crashing.
- --Reproducibility-- – the embedded Shakespeare excerpt is public domain
  and shipped with the module; no network call is required.
- --Documentation-- – module/class/method docstrings plus inline comments
  explaining *why*, not just *what*.

Dependencies
------------
Python 3.10+, tiktoken, matplotlib, numpy, pandas.

    pip3 install tiktoken matplotlib numpy pandas

Dataset sources
---------------
- --Shakespeare excerpt-- – embedded public-domain text from *The Complete
  Works of William Shakespeare* (Project Gutenberg #100,
  https://www.gutenberg.org/files/100/100-0.txt). License: public domain
  in the United States.
- --Synthetic sentences-- – authored here for the batch-speed exercise.
  Fixed random seed (42) for reproducibility. License: CC0.

Usage
-----
    python3 tiktoken_exercise_solutions.py

Saves figures `fig_ex1_tiktoken_fertility.png` and
`fig_ex3_tiktoken_batch.png` in the current working directory.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken

# Fixed seeds – Tiktoken itself is deterministic, but matplotlib and any
# synthetic-data shuffles should be reproducible across machines.
random.seed(42)
np.random.seed(42)


# ======================================================================
# CONSTANTS – embedded corpora keep the script runnable offline
# ======================================================================

# Public-domain Shakespeare excerpt (~750 words). Used for Exercise 1
# (fertility comparison) and available offline so no network is needed.
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
Even in the cannon's mouth. And then the justice,
In fair round belly with good capon lined,
With eyes severe and beard of formal cut,
Full of wise saws and modern instances;
And so he plays his part. The sixth age shifts
Into the lean and slipper'd pantaloon,
With spectacles on nose and pouch on side,
His youthful hose, well saved, a world too wide
For his shrunk shank; and his big manly voice,
Turning again toward childish treble, pipes
And whistles in his sound. Last scene of all,
That ends this strange eventful history,
Is second childishness and mere oblivion,
Sans teeth, sans eyes, sans taste, sans everything.

Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day
To the last syllable of recorded time,
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage
And then is heard no more. It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date;
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.
""".strip()


# ======================================================================
# SHARED ADAPTER (DRY – a single façade for every metric)
# ======================================================================

@dataclass
class TiktokenAdapter:
    """
    Uniform façade over a Tiktoken encoding.

    Every exercise in this file calls into this class, so metric
    definitions (fertility, compression_ratio, tokens_per_char) are
    defined exactly once and behave identically everywhere.

    Attributes
    ----------
    encoding_name : str
        Name of the Tiktoken encoding (e.g. ``"cl100k_base"``,
        ``"o200k_base"``, ``"p50k_base"``).
    allowed_special : set
        Literal strings the caller *wants* treated as special tokens.
        Defaults to empty (nothing is special).
    disallowed_special : set
        Literal strings that must raise ``ValueError`` if seen in input.
        Defaults to empty as well – see ``safe_encode`` in Exercise 2
        for the rationale. (Tiktoken's own default is ``"all"``, which
        is a security feature, not a bug.)
    enc : tiktoken.Encoding
        The loaded encoding object. Populated in ``__post_init__``.
    """

    encoding_name: str = "o200k_base"
    allowed_special: set = field(default_factory=set)
    disallowed_special: set = field(default_factory=set)

    def __post_init__(self) -> None:
        self.enc = tiktoken.get_encoding(self.encoding_name)

    # ---- core encoding / decoding -------------------------------
    def encode(self, text: str) -> List[int]:
        """Encode ``text`` to a list of integer token IDs."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        return self.enc.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )

    def decode(self, ids: List[int]) -> str:
        """Inverse of ``encode``."""
        return self.enc.decode(ids)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode many strings in one Rust call."""
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("All items must be strings.")
        return self.enc.encode_batch(
            texts,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )

    # ---- metrics ------------------------------------------------
    def token_count(self, text: str) -> int:
        """Number of tokens produced by encoding ``text``."""
        return len(self.encode(text))

    def fertility(self, corpus: str) -> float:
        """Tokens per whitespace-separated word."""
        n_words = len(corpus.split())
        return self.token_count(corpus) / n_words if n_words else 0.0

    def tokens_per_char(self, text: str) -> float:
        """Tokens per Unicode code point (script-agnostic)."""
        return self.token_count(text) / len(text) if text else 0.0

    def compression_ratio(self, text: str) -> float:
        """UTF-8 bytes per token (higher = better)."""
        n_tokens = self.token_count(text)
        return len(text.encode("utf-8")) / n_tokens if n_tokens else 0.0

    @property
    def vocab_size(self) -> int:
        """Actual vocabulary size of the loaded encoding."""
        return self.enc.n_vocab


def section(title: str) -> None:
    """Print a visually distinct section header."""
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


# ======================================================================
# EXERCISE 1 SOLUTION – Fertility across encodings
# ======================================================================

@dataclass
class Exercise1Solution:
    """
    Compute fertility of the same corpus under several Tiktoken encodings.

    The exercise asks: "Which encoding gives the lowest fertility on
    English?"  This class answers it with a table and a bar chart.

    Attributes
    ----------
    corpus : str
        The evaluation text (default: embedded Shakespeare excerpt).
    encoding_names : tuple of str
        Encodings to compare. ``p50k_base`` is included to show how an
        older 50k vocab performs; it is skipped gracefully if not
        installed (some stripped-down tiktoken wheels omit it).
    """

    corpus: str = EMBEDDED_SHAKESPEARE
    encoding_names: Tuple[str, ...] = ("p50k_base", "cl100k_base", "o200k_base")

    def run(self) -> pd.DataFrame:
        """
        Compute one row per available encoding.

        Returns
        -------
        pandas.DataFrame
            Columns: ``encoding``, ``vocab_size``, ``token_count``,
            ``fertility``, ``tokens_per_char``, ``compression_ratio``.
        """
        rows: List[dict] = []
        for name in self.encoding_names:
            try:
                adapter = TiktokenAdapter(encoding_name=name)
            except Exception as exc:
                # Not every wheel ships every encoding; skip gracefully.
                print(f"[Ex1] Skipping {name!r}: {exc}")
                continue

            rows.append(
                dict(
                    encoding=name,
                    vocab_size=adapter.vocab_size,
                    token_count=adapter.token_count(self.corpus),
                    fertility=adapter.fertility(self.corpus),
                    tokens_per_char=adapter.tokens_per_char(self.corpus),
                    compression_ratio=adapter.compression_ratio(self.corpus),
                )
            )

        if not rows:
            raise RuntimeError(
                "No encodings available – ensure `pip install tiktoken` "
                "installed a full release."
            )
        return pd.DataFrame(rows).sort_values("fertility").reset_index(drop=True)

    def plot(
        self,
        df: pd.DataFrame,
        save_to: Union[str, Path] = "fig_ex1_tiktoken_fertility.png",
    ) -> None:
        """Bar chart of fertility by encoding (lower is better on English)."""
        if df.empty:
            return

        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(df)))
        bars = ax.bar(df["encoding"], df["fertility"], color=colors)

        for bar, tpc in zip(bars, df["tokens_per_char"]):
            ax.annotate(
                f"{bar.get_height():.3f}\n({tpc:.3f} tok/char)",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_ylabel("Fertility (tokens / whitespace word)")
        ax.set_title(
            "Exercise 1 – Fertility on the Shakespeare excerpt "
            "(lower is better)"
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex1] Figure saved: {Path(save_to).resolve()}")

    def report(self, df: pd.DataFrame) -> None:
        """Print a human-readable answer to the exercise question."""
        if df.empty:
            return
        best = df.iloc[0]
        print("\n[Ex1] Answer to the exercise question:")
        print(
            f"  Lowest fertility on English: {best['encoding']!r} "
            f"with {best['fertility']:.3f} tokens/word "
            f"(vocab={best['vocab_size']:,})."
        )
        if "p50k_base" in df["encoding"].values:
            p50k_row = df[df["encoding"] == "p50k_base"].iloc[0]
            print(
                f"  Note: the 50k-vocab p50k_base gives "
                f"{p50k_row['fertility']:.3f} tokens/word – higher (worse) "
                f"than the larger vocabs, which is the expected "
                f"vocabulary-size effect."
            )


# ======================================================================
# EXERCISE 2 SOLUTION – Safe encoding of user input
# ======================================================================

@dataclass
class Exercise2Solution:
    """
    A production-minded wrapper around ``TiktokenAdapter.encode`` that
    never raises on literal special-token strings.

    Why this matters
    ----------------
    Tiktoken's default ``disallowed_special="all"`` raises ``ValueError``
    when the input contains a literal like ``<|endoftext|>``. That is a
    security feature – a user who types ``<|endoftext|>`` into a chat
    prompt should not silently inject an end-of-sequence token. But in
    many pipelines (e.g. preprocessing a user-uploaded document), you
    *do* want to treat such strings as ordinary text and continue.

    This class implements the safe-encoding contract:
        safe_encode(text)      → List[int]    (never raises for valid str)
        decode(safe_encode(t)) == t           (round-trip, for any t)
    """

    encoding_name: str = "o200k_base"

    def __post_init__(self) -> None:
        # Load one underlying encoding – reused for both safe and default
        # paths so we don't pay the load cost twice.
        self.enc = tiktoken.get_encoding(self.encoding_name)

    # ---- the safe path ------------------------------------------
    def safe_encode(self, text: str) -> List[int]:
        """
        Encode ``text`` treating *every* special-token literal as normal
        text.  Never raises ``ValueError`` on special strings.

        Parameters
        ----------
        text : str
            Arbitrary user input. ``TypeError`` is still raised for
            non-string inputs (a programmer error, not user data).

        Returns
        -------
        list of int
            Token IDs. ``decode(ids) == text``.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        # `disallowed_special=()` disables the safety check entirely;
        # `allowed_special=set()` keeps the special-token set empty so
        # nothing is treated as special. Both are needed.
        return self.enc.encode(
            text,
            allowed_special=set(),
            disallowed_special=(),
        )

    # ---- the strict path (for contrast) -------------------------
    def strict_encode(self, text: str) -> List[int]:
        """
        Encode with Tiktoken's default ``disallowed_special="all"``.

        Raises
        ------
        ValueError
            If ``text`` contains a literal special-token string.
        """
        return self.enc.encode(text)  # default disallowed_special="all"

    # ---- verification -------------------------------------------
    def verify(self) -> None:
        """
        Assert the two contracts of ``safe_encode``:

        1. ``safe_encode("<|endoftext|>")`` returns a non-empty list.
        2. ``decode(safe_encode(t)) == t`` for several special literals
           and for ordinary text containing them mid-sentence.
        """
        probes = [
            "<|endoftext|>",
            "Hello <|endoftext|> world",
            "Multiple <|endoftext|><|endoftext|> tokens",
            "Ordinary text with no specials",
            "Unicode: 你好 😊 नमस्ते",
        ]
        for text in probes:
            ids = self.safe_encode(text)
            assert ids, f"safe_encode returned [] for {text!r}"
            assert self.enc.decode(ids) == text, (
                f"round-trip failed for {text!r}"
            )
        print(f"[Ex2] ✓ safe_encode round-trips {len(probes)} probes losslessly.")

        # Demonstrate that the strict path *does* raise – proving the
        # safety feature is not silently disabled.
        try:
            self.strict_encode("Hello <|endoftext|> world")
        except ValueError as exc:
            print(f"[Ex2] ✓ strict_encode raises ValueError as expected: {exc}")
        else:
            raise AssertionError(
                "strict_encode should have raised on <|endoftext|>"
            )


# ======================================================================
# EXERCISE 3 SOLUTION – Batch vs loop speed
# ======================================================================

@dataclass
class Exercise3Solution:
    """
    Measure the wall-clock speedup of ``encode_batch`` over a Python loop.

    ``encode_batch`` releases the GIL and processes in Rust, so for
    hundreds of short strings it is typically several times faster than
    ``[enc.encode(t) for t in texts]``. The exact speedup depends on the
    machine and the string lengths; the number is *measured*, not
    hard-coded, so the report is honest.

    Attributes
    ----------
    encoding_name : str
        Encoding to benchmark.
    n_sentences : int
        Number of synthetic sentences to generate.
    words_per_sentence : tuple of (min, max)
        Sampling range for sentence length.
    seed : int
        RNG seed for the synthetic corpus (reproducibility).
    """

    encoding_name: str = "o200k_base"
    n_sentences: int = 1_000
    words_per_sentence: Tuple[int, int] = (5, 20)
    seed: int = 42

    # ---- synthetic corpus ---------------------------------------
    def make_corpus(self) -> List[str]:
        """
        Generate a reproducible list of synthetic English sentences.

        The vocabulary is small and deliberately repetitive; that is
        realistic for batch preprocessing where many strings share
        common words (and therefore shared token sub-sequences).
        """
        vocab = (
            "the quick brown fox jumps over lazy dog "
            "data model token learn train test eval "
            "python rust code batch encode decode "
            "user input text corpus sample file"
        ).split()
        rng = random.Random(self.seed)
        lo, hi = self.words_per_sentence
        return [
            " ".join(rng.choices(vocab, k=rng.randint(lo, hi)))
            for _ in range(self.n_sentences)
        ]

    # ---- benchmark ----------------------------------------------
    def _time(self, fn, *args, repeat: int = 3) -> float:
        """
        Return the best wall-clock time over ``repeat`` runs (seconds).

        Using the *best* rather than the mean is standard for micro-
        benchmarks: it removes one-off scheduling noise without hiding
        the real cost.
        """
        best = float("inf")
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn(*args)
            best = min(best, time.perf_counter() - t0)
        return best

    def run(self) -> pd.DataFrame:
        """
        Benchmark ``encode`` (loop) and ``encode_batch`` on the synthetic
        corpus.  Returns one row per method plus a speedup ratio.
        """
        adapter = TiktokenAdapter(encoding_name=self.encoding_name)
        texts = self.make_corpus()

        # Warm-up: first call loads lazy caches inside tiktoken. Without
        # a warm-up, the batch path pays the load cost and looks slow.
        _ = adapter.encode_batch(texts[:5])

        loop_time = self._time(
            lambda: [adapter.encode(t) for t in texts]
        )
        batch_time = self._time(lambda: adapter.encode_batch(texts))

        # Correctness check: both paths must produce identical IDs.
        loop_ids = [adapter.encode(t) for t in texts]
        batch_ids = adapter.encode_batch(texts)
        assert loop_ids == batch_ids, "batch and loop disagree!"

        total_bytes = sum(len(t.encode("utf-8")) for t in texts)
        df = pd.DataFrame(
            [
                dict(method="loop",    total_seconds=loop_time,
                     throughput_mb_s=total_bytes / loop_time / 1e6),
                dict(method="batch",   total_seconds=batch_time,
                     throughput_mb_s=total_bytes / batch_time / 1e6),
            ]
        )
        speedup = loop_time / batch_time if batch_time else float("nan")
        df.attrs["speedup"] = speedup
        df.attrs["n_sentences"] = len(texts)
        df.attrs["total_bytes"] = total_bytes
        return df

    # ---- reporting ----------------------------------------------
    def plot(
        self,
        df: pd.DataFrame,
        save_to: Union[str, Path] = "fig_ex3_tiktoken_batch.png",
    ) -> None:
        """Bar chart of throughput (MB/s) for loop vs batch."""
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["steelblue", "seagreen"]
        bars = ax.bar(df["method"], df["throughput_mb_s"], color=colors)
        for bar, secs in zip(bars, df["total_seconds"]):
            ax.annotate(
                f"{bar.get_height():.2f} MB/s\n({secs*1000:.1f} ms)",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylabel("Throughput (MB/s)")
        ax.set_title(
            f"Exercise 3 – encode_batch vs Python loop "
            f"({df.attrs['n_sentences']} sentences)"
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex3] Figure saved: {Path(save_to).resolve()}")

    def report(self, df: pd.DataFrame) -> None:
        """Print the measured speedup and throughput."""
        if df.empty:
            return
        speedup = df.attrs.get("speedup", float("nan"))
        print("\n[Ex3] Benchmark results:")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(
            f"  Speedup (loop / batch) = {speedup:.2f}× "
            f"on {df.attrs['n_sentences']} sentences "
            f"({df.attrs['total_bytes']/1e6:.2f} MB total)."
        )
        if speedup < 1.0:
            print(
                "  Note: batch was not faster on this machine. This can "
                "happen with very short strings or under CPU contention; "
                "the loop overhead dominates only when strings are longer."
            )


# ======================================================================
# DRIVER
# ======================================================================

def main() -> None:
    """Run all three exercise solutions end-to-end."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)

    # ------------------------------------------------------------ 1
    section("EXERCISE 1 – Fertility across encodings")
    ex1 = Exercise1Solution()
    print(f"Corpus: embedded Shakespeare excerpt "
          f"({len(ex1.corpus):,} chars, "
          f"{len(ex1.corpus.split()):,} whitespace words)")
    df1 = ex1.run()
    print("\n[Ex1] Fertility table:")
    print(df1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ex1.report(df1)
    ex1.plot(df1)

    # ------------------------------------------------------------ 2
    section("EXERCISE 2 – Safe encoding of user input")
    ex2 = Exercise2Solution()
    ex2.verify()
    # Show a couple of concrete encodings for the learner to eyeball.
    for probe in ["<|endoftext|>", "Hello <|endoftext|> world"]:
        ids = ex2.safe_encode(probe)
        print(f"  safe_encode({probe!r}) → {len(ids)} tokens, "
              f"ids[:5]={ids[:5]}…")

    # ------------------------------------------------------------ 3
    section("EXERCISE 3 – Batch vs loop speed")
    ex3 = Exercise3Solution()
    df3 = ex3.run()
    ex3.report(df3)
    ex3.plot(df3)

    section("DONE")
    print("Figures: fig_ex1_tiktoken_fertility.png, "
          "fig_ex3_tiktoken_batch.png")


if __name__ == "__main__":
    main()

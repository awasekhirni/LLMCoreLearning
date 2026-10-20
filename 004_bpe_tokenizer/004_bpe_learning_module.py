"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
#
# (ELI5)

Imagine you have a huge box of letter tiles. Some letter pairs, like "t" and "h", keep showing up next to each other. You decide to glue those two tiles together into one new tile called "th". You keep gluing the most frequent pairs together until you have a nice set of bigger and bigger tiles. Now, when you see a new word, you can build it using these pre‑glued tiles instead of every single letter. That’s Byte‑Pair Encoding (BPE): it starts with individual characters (or bytes) and repeatedly merges the most frequent adjacent pair into a new token. The result is a vocabulary of subword pieces that can represent any text-even words the tokenizer has never seen-by falling back to smaller pieces or individual bytes. This makes it perfect for feeding text into language models: the model sees a sequence of integer token IDs instead of raw characters, and the sequence is shorter and more meaningful than a pure character‑level stream.

#
# Purpose:
    Educational implementation of Byte-Pair Encoding (BPE) tokenization.
    Includes a from-scratch character-level BPE tokenizer, a production-grade
    HuggingFace `tokenizers` example, edge-case handling, and visualisation
    helpers.

    Installation Environment
pip3 install tokenizers matplotlib numpy pandas



Complete solutions to Exercises 1–3 from the BPE Learning Module:

  Exercise 1 - Tune vocabulary size and measure fertility on Shakespeare.
  Exercise 2 - Compare character-level vs byte-level BPE on multilingual text.
  Exercise 3 - Compute compression ratio (UTF-8 bytes / token).

Design principles
-----------------
-DRY - a single `TokenizerAdapter` base class supplies every metric
  (`token_count`, `fertility`, `tokens_per_char`, `compression_ratio`);
  a single `BPETokenizer._train_internal` powers both `train` and
  `train_with_snapshots`; corpora and plotting helpers are shared.
--Dataclasses - every stateful concept is a `@dataclass` so it is
  trivially instantiable, inspectable, and comparable.
--One training pass, many measurements - `train_with_snapshots` uses
  the fact that BPE is deterministic and incremental to train once and
  snapshot at multiple vocabulary sizes.
---Full documentation-- - module, class, and method docstrings;
  inline comments explain 'why', not just 'what'.

Usage
-----
    python bpe_exercises.py

Saves figures `fig_ex1.png`, `fig_ex2.png`, `fig_ex3.png` in the current
working directory. Works offline: if the Gutenberg download fails, an
embedded Shakespeare excerpt is used instead.

Dependencies
------------
Python 3.10+, tokenizers (HuggingFace), matplotlib, numpy, pandas.

    pip install tokenizers matplotlib numpy pandas

Optional: if `tokenizers` is not installed, Exercises 1 and 3 still run;
Exercise 2 gracefully degrades to character-level-only mode.

Author: Senior AI Engineer / Educator
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import copy
import random
import re
import tempfile
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

# ----------------------------------------------------------------------
# Third-party
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# HuggingFace `tokenizers` is optional so this file still runs when the
# library is absent (Exercises 1 and 3 work; Exercise 2 downgrades).
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE as HFBPE       # renamed to avoid name clash
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    HF_AVAILABLE = True
except ImportError:                                   # pragma: no cover
    HF_AVAILABLE = False
    print("[Warning] HuggingFace `tokenizers` not installed. "
          "Install with: pip install tokenizers")

# Fixed seeds for symmetry with the module; BPE itself is deterministic.
random.seed(42)
np.random.seed(42)


# ======================================================================
# CONSTANTS - embedded corpora keep the script runnable offline
# ======================================================================

GUTENBERG_URL = "https://www.gutenberg.org/files/100/100-0.txt"

# Public-domain Shakespeare excerpt (~750 words across 6 famous passages).
# Used as a fallback if the Gutenberg download fails.
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

Is this a dagger which I see before me,
The handle toward my hand? Come, let me clutch thee.
I have thee not, and yet I see thee still.
Art thou not, fatal vision, sensible
To feeling as to sight? or art thou but
A dagger of the mind, a false creation,
Proceeding from the heat-oppressed brain?
I see thee yet, in form as palpable
As this which now I draw.

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

Let me not to the marriage of true minds
Admit impediments. Love is not love
Which alters when it alteration finds,
Or bends with the remover to remove.
O no! it is an ever-fixed mark
That looks on tempests and is never shaken;
It is the star to every wandering bark,
Whose worth's unknown, although his height be taken.
""".strip()


# Small multilingual training corpus (English / Spanish / a little Chinese
# and code).  Deliberately under-represents Chinese so the byte-level
# fallback has something to prove in Exercise 2.
MULTILINGUAL_TRAIN: str = """
the quick brown fox jumps over the lazy dog
the lazy dog sleeps all day
hello world how are you today
i love programming in python
data science is fun and challenging
machine learning models need lots of data
el rapido zorro marron salta sobre el perro perezoso
el perro perezoso duerme todo el dia
hola mundo como estas hoy
me encanta programar en python
la ciencia de datos es divertida y desafiante
你好世界今天你好吗
我喜欢用python编程
机器学习模型需要大量数据
def foo x return x plus one
print hello
""".strip()


# Held-out test sentences.  Each tuple is (label, text).  All text is
# already lowercase/caseless so that the from-scratch (case-insensitive)
# tokenizer and byte-level HF tokenizer (case-sensitive) can be compared
# fairly on *subword segmentation* alone.
MULTILINGUAL_TEST: List[Tuple[str, str]] = [
    ("english (seen)",  "hello world how are you today"),
    ("spanish (seen)",  "hola mundo como estas hoy"),
    ("chinese (mixed)", "你好世界宇宙无限"),  # 宇宙无限 are unseen
    ("code (seen)",     "def foo x return x plus one"),
    ("emoji (unseen)",  "hello 😊🚀 world"),
    ("hindi (unseen)",  "नमस्ते दुनिया"),
]


# ======================================================================
# SHARED UTILITIES
# ======================================================================

def load_shakespeare_corpus(
    max_chars: Optional[int] = 50_000,
    prefer_download: bool = True,
    url: str = GUTENBERG_URL,
    timeout_s: float = 10.0,
) -> str:
    """
    Load a Shakespeare corpus, preferring the full Gutenberg text.

    Parameters
    ----------
    max_chars : int or None
        Truncate the corpus to this many characters (keeps runtimes
        classroom-friendly).  ``None`` disables truncation.
    prefer_download : bool
        Try the network first, fall back to the embedded excerpt on any
        error.  An offline run therefore always succeeds.
    url : str
        Gutenberg URL for *The Complete Works of William Shakespeare*.
    timeout_s : float
        Network timeout - never hang a live demo.

    Returns
    -------
    str
        Raw text, guaranteed non-empty.
    """
    text: Optional[str] = None
    if prefer_download:
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                text = resp.read().decode("utf-8")
        except Exception as exc:                     # noqa: BLE001 - fallback is intended
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
# CORE - from-scratch character-level BPE tokenizer
# ======================================================================

@dataclass
class BPETokenizer:
    """
    A minimal character-level Byte-Pair Encoding tokenizer.

    The implementation is intentionally small and readable.  It is
    suitable for teaching; for production, use HuggingFace ``tokenizers``
    (see :class:`HFByteLevelAdapter`).

    Attributes
    ----------
    vocab_size : int
        Target vocabulary size *including* special tokens and the base
        characters.  Training stops early if pairs drop below frequency 2.
    special_tokens : list of str
        Reserved tokens, always placed first in the vocabulary
        (``<pad>``, ``<eos>``, ``<bos>``, ``<unk>`` by default).
    merges : dict[(str, str), str]
        Learned merge rules, in insertion order.  ``merges[(a, b)] == a + b``.
    vocab : dict[str, int]
        Token string → integer ID.
    inverse_vocab : dict[int, str]
        Integer ID → token string.
    _word_cache : dict[str, list[str]]
        Per-word memoisation of the tokenizer output.  Populated lazily
        during encoding; cleared on every retrain.  This is a pure
        performance optimisation - it does not affect correctness.
    """

    vocab_size: int = 1000
    special_tokens: List[str] = field(
        default_factory=lambda: ["<pad>", "<eos>", "<bos>", "<unk>"]
    )
    merges: Dict[Tuple[str, str], str] = field(default_factory=dict)
    vocab: Dict[str, int] = field(default_factory=dict)
    inverse_vocab: Dict[int, str] = field(default_factory=dict)
    _word_cache: Dict[str, List[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Seed the vocabulary with the special tokens."""
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.inverse_vocab = {i: tok for tok, i in self.vocab.items()}

    # ------------------------------------------------------------------
    # Training primitives
    # ------------------------------------------------------------------
    def _get_stats(self, sequences: List[List[str]]) -> Counter:
        """Count adjacent-pair frequencies across all sequences."""
        stats: Counter = Counter()
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                stats[(a, b)] += 1
        return stats

    def _merge_pair(
        self,
        pair: Tuple[str, str],
        sequences: List[List[str]],
    ) -> List[List[str]]:
        """
        Replace non-overlapping occurrences of ``pair`` left-to-right with
        its concatenation.  Example: ``("a","b")`` on ``["a","b","a","b"]``
        → ``["ab","ab"]``.  Left-to-right scanning matters because BPE
        defines *non-overlapping* merges.
        """
        left, right = pair
        merged = left + right
        out: List[List[str]] = []
        for seq in sequences:
            new_seq: List[str] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == left and seq[i + 1] == right:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            out.append(new_seq)
        return out

    def _train_internal(
        self,
        corpus: str,
        verbose: bool,
        on_progress: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Shared training loop used by :meth:`train` and
        :meth:`train_with_snapshots`.

        The optional ``on_progress`` callback is invoked once *after*
        the initial vocabulary is seeded and then *after every successful
        merge*.  This is the DRY seam that lets snapshots be captured
        without duplicating training logic.
        """
        # -- 0. Reset state so this tokenizer can be reused ---------------
        self.merges.clear()
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.inverse_vocab = {i: tok for tok, i in self.vocab.items()}
        self._word_cache.clear()

        # -- 1. Pre-tokenise into words (word chars or single punctuation)
        # The "</w>" marker teaches the tokenizer where words end, which
        # is essential for learning suffixes like "-ing" or "-est".
        words = re.findall(r"\w+|[^\w\s]", corpus.lower())
        sequences: List[List[str]] = [[ch for ch in w] + ["</w>"] for w in words]

        # -- 2. Seed with base characters --------------------------------
        for seq in sequences:
            for tok in seq:
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
                    self.inverse_vocab[len(self.vocab) - 1] = tok

        # -- 3. Notify listener of initial state -------------------------
        if on_progress is not None:
            on_progress()

        # -- 4. Greedy merge loop ----------------------------------------
        num_merges = max(0, self.vocab_size - len(self.vocab))
        for merge_idx in range(num_merges):
            stats = self._get_stats(sequences)
            if not stats:
                break
            most_frequent = max(stats, key=stats.get)
            if stats[most_frequent] < 2:
                break                              # no pair repeats → stop
            new_token = most_frequent[0] + most_frequent[1]
            self.merges[most_frequent] = new_token
            # The merged token is always new; if it somehow exists, skip
            # adding it again to avoid ID drift.
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.inverse_vocab[len(self.vocab) - 1] = new_token
            sequences = self._merge_pair(most_frequent, sequences)
            if verbose and (merge_idx + 1) % 100 == 0:
                print(f"  merge {merge_idx + 1}: {most_frequent} -> {new_token}")
            if on_progress is not None:
                on_progress()

    # ------------------------------------------------------------------
    # Public training APIs
    # ------------------------------------------------------------------
    def train(self, corpus: str, verbose: bool = False) -> None:
        """Train from scratch up to ``self.vocab_size``."""
        self._train_internal(corpus, verbose, on_progress=None)

    def train_with_snapshots(
        self,
        corpus: str,
        snapshot_sizes: Sequence[int],
        verbose: bool = False,
    ) -> Dict[int, "BPETokenizer"]:
        """
        Train once and return deep-copied snapshots at each requested size.

        Because BPE is deterministic and incremental, the state captured
        when ``len(vocab) == K`` is byte-for-byte identical to what a
        fresh ``BPETokenizer(vocab_size=K)`` would produce - but this
        method trains **once** instead of N times.  See
        :meth:`Exercise1Solution.sanity_check` for the verification.

        Parameters
        ----------
        corpus : str
        snapshot_sizes : sequence of int
            Requested vocabulary sizes.  Sorted internally; duplicates
            collapsed.
        verbose : bool

        Returns
        -------
        dict[int, BPETokenizer]
            Only sizes actually reached are present (training may stop
            early if no pair repeats).
        """
        pending = sorted(set(snapshot_sizes))
        snapshots: Dict[int, "BPETokenizer"] = {}

        def capture() -> None:
            # `while` (not `if`) because seeding the base chars can jump
            # past several small targets at once.
            while pending and len(self.vocab) >= pending[0]:
                size = pending.pop(0)
                snapshots[size] = copy.deepcopy(self)

        self._train_internal(corpus, verbose, on_progress=capture)
        return snapshots

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Apply learned merges to a single word; unknown chars become
        ``<unk>``.  Cached because a small corpus contains many repeated
        words - this is the hot path in :meth:`encode`.
        """
        if word in self._word_cache:
            return list(self._word_cache[word])       # copy so callers can't mutate

        unk = self.special_tokens[3]
        seq = [ch if ch in self.vocab else unk for ch in word.lower()]
        seq.append("</w>")
        # Merges are applied in the order they were learned - later merges
        # may depend on earlier ones, so iteration order is not optional.
        for pair in self.merges:
            seq = self._merge_pair(pair, [seq])[0]

        self._word_cache[word] = list(seq)
        return seq

    def encode(self, text: str) -> List[int]:
        """Encode ``text`` into a list of integer token IDs."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        if not text.strip():
            return []
        ids: List[int] = []
        unk_id = self.vocab[self.special_tokens[3]]
        for word in re.findall(r"\w+|[^\w\s]", text.lower()):
            for tok in self._tokenize_word(word):
                ids.append(self.vocab.get(tok, unk_id))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Inverse of :meth:`encode`.  Case and unknown chars are not recovered."""
        if not ids:
            return ""
        unk = self.special_tokens[3]
        toks = [self.inverse_vocab.get(i, unk) for i in ids]
        return "".join(toks).replace("</w>", " ").strip()

    @property
    def effective_vocab_size(self) -> int:
        """Actual vocabulary size (may be smaller than ``vocab_size``)."""
        return len(self.vocab)


# ======================================================================
# ADAPTER BASE CLASS - every metric lives here, once (DRY)
# ======================================================================

@dataclass
class TokenizerAdapter:
    """
    Uniform façade over any tokenizer, exposing every metric used across
    the three exercises.  Subclasses only implement :meth:`encode`,
    :meth:`decode`, and :attr:`vocab_size`.
    """

    name: str

    # --- to be implemented by subclasses ------------------------------
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    # --- shared metric implementations --------------------------------
    def token_count(self, text: str) -> int:
        """Number of tokens produced by encoding ``text``."""
        return len(self.encode(text))

    def fertility(self, corpus: str) -> float:
        """
        Tokens per whitespace-separated word.

        Words = ``corpus.split()`` - a coarse but standard convention.
        For space-less scripts (Chinese), use :meth:`tokens_per_char`.
        """
        n_words = len(corpus.split())
        return self.token_count(corpus) / n_words if n_words else 0.0

    def tokens_per_char(self, text: str) -> float:
        """Tokens per Unicode code point - script-agnostic alternative."""
        return self.token_count(text) / len(text) if text else 0.0

    def compression_ratio(self, text: str) -> float:
        """
        UTF-8 bytes per token.  Higher = better.

        Uses UTF-8 *bytes* (not ``len(text)``) so the metric is
        comparable across scripts - a CJK character is 3 bytes, and that
        is what actually consumes memory in the model.
        """
        n_tokens = self.token_count(text)
        return len(text.encode("utf-8")) / n_tokens if n_tokens else 0.0


# ======================================================================
# CONCRETE ADAPTERS
# ======================================================================

@dataclass
class FromScratchAdapter(TokenizerAdapter):
    """Adapter for :class:`BPETokenizer`."""

    tokenizer: BPETokenizer = field(default_factory=BPETokenizer)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.effective_vocab_size


@dataclass
class HFByteLevelAdapter(TokenizerAdapter):
    """Adapter for a trained HuggingFace ``tokenizers.Tokenizer``."""

    tokenizer: Tokenizer                       # required - no default, no silent None

    def encode(self, text: str) -> List[int]:
        # `add_special_tokens=False`: our model has no post-processor, but
        # being explicit future-proofs the code against pipeline changes.
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


def train_hf_bytelevel(corpus: str, vocab_size: int = 500) -> HFByteLevelAdapter:
    """
    Train a byte-level BPE tokenizer with HuggingFace ``tokenizers``.

    Uses a temporary on-disk file because ``tokenizers.train`` requires
    a file path.  The file is deleted automatically on exit.

    Parameters
    ----------
    corpus : str
    vocab_size : int
        Target vocabulary size including special tokens.

    Returns
    -------
    HFByteLevelAdapter
    """
    if not HF_AVAILABLE:
        raise ImportError("Install HuggingFace tokenizers: pip install tokenizers")

    # NamedTemporaryFile with delete=False so `tokenizer.train()` can open
    # it after this block exits; we delete explicitly in `finally`.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False
    )
    try:
        tmp.write(corpus)
        tmp.close()

        tok = Tokenizer(HFBPE(unk_token="<unk>"))
        # ByteLevel = the byte-level fallback mentioned in the technical card.
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>"],
            min_frequency=2,
            show_progress=False,
        )
        tok.train(files=[tmp.name], trainer=trainer)
        return HFByteLevelAdapter(name=f"hf-bytelevel-{vocab_size}", tokenizer=tok)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


# ======================================================================
# EXERCISE 1 - vocabulary-size tuning
# ======================================================================

@dataclass
class Exercise1Solution:
    """
    Train once at ``max_vocab``, snapshot at several target sizes, and
    report fertility + compression for each.

    The snapshot trick uses BPE's determinism: state at vocabulary ``K``
    is identical to training a fresh tokenizer with ``vocab_size=K``.
    """

    corpus: str
    vocab_targets: Tuple[int, ...] = (100, 200, 400)
    max_vocab: int = 400

    # --- public API ---------------------------------------------------
    def run(self) -> pd.DataFrame:
        """
        Execute the exercise and return one row per snapshot.

        Returns
        -------
        pandas.DataFrame
            Columns: ``requested_vocab``, ``actual_vocab``, ``fertility``,
            ``tokens_per_char``, ``compression_ratio``, ``train_seconds``.
            Rows are NaN when a target was not reached (training stopped
            because no pair repeated).
        """
        targets = sorted({t for t in self.vocab_targets if t <= self.max_vocab})
        tokenizer = BPETokenizer(vocab_size=self.max_vocab)

        t0 = time.perf_counter()
        snapshots = tokenizer.train_with_snapshots(self.corpus, targets)
        elapsed = time.perf_counter() - t0

        rows: List[dict] = []
        for requested in targets:
            snap = snapshots.get(requested)
            if snap is None:
                # Training ended before reaching this size - record NaN so
                # the learner sees the shortfall instead of a silent gap.
                rows.append(dict(
                    requested_vocab=requested, actual_vocab=np.nan,
                    fertility=np.nan, tokens_per_char=np.nan,
                    compression_ratio=np.nan, train_seconds=elapsed,
                ))
                continue
            adapter = FromScratchAdapter(name=f"scratch-{requested}", tokenizer=snap)
            rows.append(dict(
                requested_vocab=requested,
                actual_vocab=snap.effective_vocab_size,
                fertility=adapter.fertility(self.corpus),
                tokens_per_char=adapter.tokens_per_char(self.corpus),
                compression_ratio=adapter.compression_ratio(self.corpus),
                train_seconds=elapsed,
            ))
        return pd.DataFrame(rows)

    def sanity_check(self) -> None:
        """
        Verify snapshot == fresh-train at the smallest target.

        Uses a corpus slice for speed; the assertion compares *both*
        vocabulary and merges, which is the strongest possible guarantee.
        """
        targets = sorted({t for t in self.vocab_targets if t <= self.max_vocab})
        if not targets:
            return
        probe = targets[0]                        # smallest = fastest
        slice_corpus = self.corpus[:20_000]

        snap_tok = BPETokenizer(vocab_size=probe)
        snapshots = snap_tok.train_with_snapshots(slice_corpus, [probe])
        if probe not in snapshots:
            print(f"[Sanity] target {probe} not reached on slice - skipping.")
            return

        fresh = BPETokenizer(vocab_size=probe)
        fresh.train(slice_corpus)

        assert snapshots[probe].vocab == fresh.vocab, "vocabulary mismatch!"
        assert snapshots[probe].merges == fresh.merges, "merges mismatch!"
        print(f"[Sanity] ✓ snapshot@{probe} == fresh@{probe} "
              f"(vocab size = {len(fresh.vocab)})")

    def plot(
        self, df: pd.DataFrame, save_to: Union[str, Path] = "fig_ex1.png"
    ) -> None:
        """Plot fertility and tokens-per-char vs actual vocabulary size."""
        ok = df.dropna(subset=["fertility"])
        if ok.empty:
            print("[Ex1] No snapshots reached - nothing to plot.")
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(ok["actual_vocab"], ok["fertility"],
                 marker="o", color="steelblue")
        ax1.set_title("Exercise 1 - Fertility vs Vocabulary Size")
        ax1.set_xlabel("Actual vocabulary size")
        ax1.set_ylabel("Fertility (tokens / whitespace word)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(ok["actual_vocab"], ok["tokens_per_char"],
                 marker="s", color="darkorange")
        ax2.set_title("Exercise 1 - Tokens per Character vs Vocabulary Size")
        ax2.set_xlabel("Actual vocabulary size")
        ax2.set_ylabel("Tokens per character")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex1] Figure saved: {Path(save_to).resolve()}")


# ======================================================================
# EXERCISE 2 - character-level vs byte-level BPE on multilingual text
# ======================================================================

@dataclass
class Exercise2Solution:
    """
    Compare a from-scratch character-level BPE with a HuggingFace
    byte-level BPE on multilingual held-out sentences.

    The critical lesson: character-level BPE produces ``<unk>`` for any
    unseen Unicode character, while byte-level BPE is always lossless
    because every code point is a sequence of UTF-8 bytes.
    """

    train_corpus: str = MULTILINGUAL_TRAIN
    test_set: List[Tuple[str, str]] = field(
        default_factory=lambda: list(MULTILINGUAL_TEST)
    )
    vocab_size: int = 500

    def run(self) -> pd.DataFrame:
        """
        Encode every test case with both tokenizers and report token
        counts, round-trip losslessness, and whether ``<unk>`` appeared.
        """
        # --- From-scratch tokenizer ---
        scratch_tok = BPETokenizer(vocab_size=self.vocab_size)
        scratch_tok.train(self.train_corpus)
        scratch = FromScratchAdapter("scratch", scratch_tok)
        unk_id = scratch_tok.vocab[scratch_tok.special_tokens[3]]

        # --- HF byte-level tokenizer (optional) ---
        hf = (
            train_hf_bytelevel(self.train_corpus, self.vocab_size)
            if HF_AVAILABLE else None
        )

        rows: List[dict] = []
        for label, text in self.test_set:
            s_ids = scratch.encode(text)
            s_decoded = scratch.decode(s_ids)
            # Compare against `text.lower()` because scratch is case-insensitive
            # by design; isolating the <unk> lossiness from the case-folding.
            s_lossless = (s_decoded == text.lower())
            s_has_unk = unk_id in s_ids

            if hf is not None:
                h_ids = hf.encode(text)
                h_decoded = hf.decode(h_ids)
                h_lossless = (h_decoded == text)
            else:
                h_ids, h_decoded, h_lossless = [], "", False

            rows.append(dict(
                case=label, text=text,
                scratch_tokens=len(s_ids),
                scratch_decoded=s_decoded,
                scratch_has_unk=s_has_unk,
                scratch_lossless=s_lossless,
                hf_tokens=len(h_ids),
                hf_decoded=h_decoded,
                hf_lossless=h_lossless,
            ))
        return pd.DataFrame(rows)

    def plot(
        self, df: pd.DataFrame, save_to: Union[str, Path] = "fig_ex2.png"
    ) -> None:
        """Grouped bar chart of token counts per test case."""
        if df.empty:
            return
        x = np.arange(len(df))
        w = 0.4
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.bar(x - w / 2, df["scratch_tokens"], w,
               label="Character-level (scratch)", color="steelblue")
        ax.bar(x + w / 2, df["hf_tokens"], w,
               label="Byte-level (HF)", color="seagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(df["case"], rotation=25, ha="right")
        ax.set_ylabel("Number of tokens")
        ax.set_title("Exercise 2 - Token counts: character-level vs byte-level BPE")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex2] Figure saved: {Path(save_to).resolve()}")


# ======================================================================
# EXERCISE 3 - compression ratio (UTF-8 bytes / token)
# ======================================================================

@dataclass
class Exercise3Solution:
    """
    Compute ``compression_ratio = utf8_bytes(text) / len(encode(text))``
    for both tokenizers on the same corpus.  Higher is better: each token
    carries more information, so sequences are shorter and inference is
    cheaper.

    UTF-8 bytes (not ``len(text)``) is the correct unit - a CJK character
    costs 3 bytes in the model's memory, and that is what actually
    determines efficiency.
    """

    train_corpus: str
    test_corpus: Optional[str] = None
    vocab_size: int = 500

    def __post_init__(self) -> None:
        # Default to evaluating on the training corpus, matching the
        # exercise wording ("on the Shakespeare excerpt").
        if self.test_corpus is None:
            self.test_corpus = self.train_corpus

    def run(self) -> pd.DataFrame:
        """Return one row per tokenizer with its compression ratio."""
        scratch_tok = BPETokenizer(vocab_size=self.vocab_size)
        scratch_tok.train(self.train_corpus)
        scratch = FromScratchAdapter("scratch", scratch_tok)

        rows: List[dict] = [dict(
            tokenizer=scratch.name,
            vocab_size=scratch.vocab_size,
            n_tokens=scratch.token_count(self.test_corpus),
            n_bytes=len(self.test_corpus.encode("utf-8")),
            compression_ratio=scratch.compression_ratio(self.test_corpus),
        )]

        if HF_AVAILABLE:
            hf = train_hf_bytelevel(self.train_corpus, self.vocab_size)
            rows.append(dict(
                tokenizer=hf.name,
                vocab_size=hf.vocab_size,
                n_tokens=hf.token_count(self.test_corpus),
                n_bytes=len(self.test_corpus.encode("utf-8")),
                compression_ratio=hf.compression_ratio(self.test_corpus),
            ))
        return pd.DataFrame(rows)

    def plot(
        self, df: pd.DataFrame, save_to: Union[str, Path] = "fig_ex3.png"
    ) -> None:
        """Bar chart comparing compression ratios."""
        if df.empty:
            return
        colors = ["steelblue", "seagreen"][: len(df)]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(df["tokenizer"], df["compression_ratio"], color=colors)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center", fontsize=10,
            )
        ax.set_ylabel("Compression ratio (UTF-8 bytes / token)")
        ax.set_title("Exercise 3 - Compression Ratio (higher is better)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Ex3] Figure saved: {Path(save_to).resolve()}")


# ======================================================================
# DRIVER
# ======================================================================

def main() -> None:
    """Run all three exercises with clear headers and timing."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 55)

    # ---------------------------------------------------------------- 1
    section("EXERCISE 1 - Vocabulary-size tuning (from-scratch BPE)")
    shakespeare = load_shakespeare_corpus(max_chars=50_000, prefer_download=True)
    print(f"Corpus length: {len(shakespeare):,} chars "
          f"({len(shakespeare.split()):,} whitespace words)")

    ex1 = Exercise1Solution(corpus=shakespeare, max_vocab=400)
    ex1.sanity_check()

    t0 = time.perf_counter()
    df1 = ex1.run()
    print(f"\nTraining + evaluation took {time.perf_counter() - t0:.2f}s")
    print(df1.to_string(index=False))
    ex1.plot(df1, save_to="fig_ex1.png")

    # ---------------------------------------------------------------- 2
    section("EXERCISE 2 - Character-level vs byte-level BPE (multilingual)")
    ex2 = Exercise2Solution(vocab_size=500)
    df2 = ex2.run()
    print(df2.to_string(index=False))
    ex2.plot(df2, save_to="fig_ex2.png")

    # ---------------------------------------------------------------- 3
    section("EXERCISE 3 - Compression ratio (Shakespeare excerpt)")
    ex3 = Exercise3Solution(
        train_corpus=shakespeare[:20_000],
        test_corpus=shakespeare[:20_000],
        vocab_size=500,
    )
    df3 = ex3.run()
    print(df3.to_string(index=False))
    ex3.plot(df3, save_to="fig_ex3.png")

    section("DONE")
    print("Figures: fig_ex1.png, fig_ex2.png, fig_ex3.png")


if __name__ == "__main__":
    main()



#Instructions to execute ( i am assuming that you are using linux/mac flavour. I do not recommend windows os)
# python3 -m venv .
# source bin/activate
# pip3 install tokenizers matplotlib numpy pandas
# python3 004_bpe_learning_module.py

#tawaajjo dein
# we have captured all possible metrics, token_count, fertiliity, tokens_per_char, compresssion_ration
# we have used sanity_check that is helpful in snapshotting which equals to fresh training

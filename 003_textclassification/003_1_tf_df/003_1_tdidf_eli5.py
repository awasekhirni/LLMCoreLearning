"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.


tfidf_eli5.py
=============
A narrated, plain-language walkthrough of TF-IDF using toy data.
Read the printed output top-to-bottom; each block is a step a child
could follow.
"""

from dataclasses import dataclass
import math

@dataclass
class Book:
    name: str
    text: str

def show(title, obj):
    print(f"\n--- {title} ---")
    print(obj)

# 1) Three tiny "books" - our library.
books = [
    Book("Animal Book", "cat cat dog bird"),
    Book("Sport Book",  "goal goal kick run"),
    Book("Food Book",   "pizza pizza pasta bread"),
]

print("HELLO! We have a tiny library with 3 books.")
for b in books:
    print(f"  - {b.name}: '{b.text}'")

# 2) Split each book into words.
print("\nSTEP 1: Split each book into words.")
all_tokens = {b.name: b.text.split() for b in books}
for name, toks in all_tokens.items():
    print(f"  {name} -> {toks}")

# 3) Count how many books each word appears in (DF).
print("\nSTEP 2: Count in HOW MANY books each word appears.")
df = {}
for toks in all_tokens.values():
    for w in set(toks):
        df[w] = df.get(w, 0) + 1
show("Document Frequency (DF)", df)

# 4) Compute IDF: rarer word -> bigger IDF.
print("\nSTEP 3: Compute IDF. A word in only 1 book is rare; its IDF is BIG.")
N = len(books)
idf = {w: math.log(N / df[w]) + 1 for w in df}
show("IDF (bigger = rarer = more special)", idf)

# 5) For each book, count word occurrences (TF).
print("\nSTEP 4: Count word occurrences inside each book (TF).")
tf = {}
for b in books:
    counts = {}
    for w in all_tokens[b.name]:
        counts[w] = counts.get(w, 0) + 1
    tf[b.name] = counts
for name, counts in tf.items():
    print(f"  {name}: {counts}")

# 6) TF-IDF = TF * IDF. The bigger the score, the more 'special' the word is
#    for THAT book.
print("\nSTEP 5: Multiply TF x IDF. The biggest score wins - that's the book's "
      "special word!")
for b in books:
    print(f"\n  {b.name}:")
    for w, c in tf[b.name].items():
        score = c * idf[w]
        print(f"    word '{w}': tf={c}, idf={idf[w]:.3f} -> TF-IDF={score:.3f}")

print("\nDONE. The word with the biggest TF-IDF is the 'fingerprint' of each book.")

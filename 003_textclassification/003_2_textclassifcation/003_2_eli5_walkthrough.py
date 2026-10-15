"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.

eli5_walkthrough.py
===================
A separate, simplified script that explains text classification to a
five-year-old using toy data and plain-language print statements.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("ELI5: TEXT CLASSIFICATION")
print("=" * 60)

# Step 0 -- setup the toy world
print("\nImagine you have four toy boxes labelled:")
print("   'sports', 'space', 'politics', 'computers'")
print("Each toy comes with a name tag (some words).")
print("Your job: read the name tag, drop the toy in the right box.\n")

toys_text = [
    "home run in the ninth inning",     # sports
    "rocket launched to the moon",      # space
    "election results were announced",  # politics
    "new graphics card released today", # computers
]
toys_label = ["sports", "space", "politics", "computers"]

# Step 1 -- turn each name tag into a list of numbers (an "embedding")
print("STEP 1: Turn each name tag's words into a list of numbers.")
print("        (This is called an 'embedding'.)")
vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(toys_text)
print(f"        Each toy became a list of {X.shape[1]} numbers.\n")

# Step 2 -- show the lists + correct answers to a 'voting box'
print("STEP 2: Show the numbers AND the right answers to a voting box.")
print("        (This is called 'training the classifier'.)")
clf = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
clf.fit(X, toys_label)
print("        The voting box has learned which numbers go with which box!\n")

# Step 3 -- a brand-new toy arrives
new_toy_text = "the spacecraft flew past mars"
print(f"STEP 3: A new toy arrives! Name tag: '{new_toy_text}'")
new_vec = vec.transform([new_toy_text])
prediction = clf.predict(new_vec)[0]
probs = clf.predict_proba(new_vec)[0]
classes = list(clf.classes_)
print(f"        Voting box says: put it in the '{prediction}' box!")
print(f"        Confidence scores:")
for c, p in zip(classes, probs):
    print(f"           {c:>10}: {p:.3f}")

print("\nThat's it! Text classification = turn words into numbers,")
print("then let a trained box pick the right label.")

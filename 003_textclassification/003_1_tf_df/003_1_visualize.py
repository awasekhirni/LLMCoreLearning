"""
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.


tfidf_visualize.py
==================
Step-by-step visualizations of TF-IDF on a tiny corpus.
Saves: fig_input_tokens.png, fig_df_idf.png, fig_tfidf_heatmap.png,
       fig_top_keywords_per_doc.png
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfidf_module import Document, TfidfVectorizer, TfidfConfig

@dataclass
class Cfg:
    seed: int = 42

def main() -> None:
    # Toy corpus deliberately chosen so visualizations are readable.
    docs = [
        Document("animals", "cat cat cat dog bird"),
        Document("sports",  "goal goal kick run team"),
        Document("food",    "pizza pizza pasta bread cheese"),
        Document("travel",  "paris rome tokyo paris rome"),
    ]
    vec = TfidfVectorizer(TfidfConfig(min_df=1))
    mat = vec.fit_transform(docs).toarray()
    feats = vec.get_feature_names()
    df_index = [d.doc_id for d in docs]
    idf = vec.idf_

    # ---- (a) Input tokens ----------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, d in enumerate(docs):
        toks = d.text.split()
        ax.scatter(range(len(toks)), [i] * len(toks),
                   s=200, c=f"C{i}", label=d.doc_id, edgecolor="black")
        for j, t in enumerate(toks):
            ax.text(j, i, t, ha="center", va="center", fontsize=9, color="white")
    ax.set_yticks(range(len(docs)))
    ax.set_yticklabels(df_index)
    ax.set_xlabel("Token position in document")
    ax.set_title("(a) Input documents tokenized")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("fig_input_tokens.png", dpi=120)
    plt.close()

    # ---- (b) DF vs IDF -------------------------------------------------
    n_docs = len(docs)
    df_vals = np.array([
        sum(1 for d in docs if t in d.text.split()) for t in feats
    ])
    fig, ax1 = plt.subplots(figsize=(11, 4))
    x = np.arange(len(feats))
    ax1.bar(x - 0.2, df_vals, width=0.4, label="DF (documents containing term)",
            color="steelblue")
    ax1.set_ylabel("Document Frequency (DF)", color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(feats, rotation=45, ha="right")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, idf, width=0.4, label="IDF (rarity score)",
            color="darkorange")
    ax2.set_ylabel("IDF", color="darkorange")
    plt.title("(b) DF vs IDF per term — IDF shrinks as DF grows")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.95))
    plt.tight_layout()
    plt.savefig("fig_df_idf.png", dpi=120)
    plt.close()

    # ---- (c) TF-IDF heatmap -------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=45, ha="right")
    ax.set_yticks(range(len(docs)))
    ax.set_yticklabels(df_index)
    ax.set_xlabel("Vocabulary term")
    ax.set_ylabel("Document")
    ax.set_title("(c) TF-IDF matrix (darker = higher score)")
    plt.colorbar(im, ax=ax, label="TF-IDF")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black",
                        fontsize=8)
    plt.tight_layout()
    plt.savefig("fig_tfidf_heatmap.png", dpi=120)
    plt.close()

    # ---- (d) Top-3 keywords per document ------------------------------
    fig, axes = plt.subplots(1, len(docs), figsize=(14, 4), sharey=True)
    for i, d in enumerate(docs):
        row = mat[i]
        top_idx = np.argsort(row)[::-1][:3]
        top_vals = row[top_idx]
        top_terms = [feats[j] for j in top_idx]
        axes[i].barh(top_terms, top_vals, color=f"C{i}")
        axes[i].invert_yaxis()
        axes[i].set_title(d.doc_id)
        axes[i].set_xlabel("TF-IDF")
    plt.suptitle("(d) Top-3 keywords per document (its 'fingerprint')")
    plt.tight_layout()
    plt.savefig("fig_top_keywords_per_doc.png", dpi=120)
    plt.close()

    print("Saved figures:")
    for f in ["fig_input_tokens.png", "fig_df_idf.png",
              "fig_tfidf_heatmap.png", "fig_top_keywords_per_doc.png"]:
        print("  -", f)

if __name__ == "__main__":
    main()

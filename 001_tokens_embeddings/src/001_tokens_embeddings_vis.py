"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.
visualizations.py
=================
Generates 4 figures that walk through the pipeline:
    fig1_tokenization.png  — input text → token IDs
    fig2_embeddings.png    — token IDs → embedding matrix (heatmap)
    fig3_word2vec_pca.png  — learned embeddings projected to 2D
    fig4_loss_curve.png    — Word2Vec training loss over epochs
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tokens_and_embeddings import (
    SimpleBPETokenizer, EmbeddingLayer, Word2VecSGNS, SEED
)
from scenario_drivers import scenario_word2vec_training  # noqa: F401 (illustrative)

np.random.seed(SEED)


def fig1_tokenization() -> None:
    tok = SimpleBPETokenizer(vocab_size=60).train([
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are pets",
    ] * 5)
    text = "the cat sat"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    tokens = [tok.id_to_token[i] for i in ids]

    fig, ax = plt.subplots(figsize=(10, 3))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ids)))
    for i, (t, tid) in enumerate(zip(tokens, ids)):
        ax.bar(i, 1, color=colors[i], edgecolor="black")
        ax.text(i, 0.5, f"{t}\n[{tid}]", ha="center", va="center",
                fontsize=9, color="white", weight="bold")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([f"pos {i}" for i in range(len(ids))])
    ax.set_yticks([])
    ax.set_title("Figure 1 — Tokenization: raw text 'the cat sat' → token IDs")
    ax.set_xlabel("Position in sequence")
    plt.tight_layout()
    plt.savefig("fig1_tokenization.png", dpi=120)
    plt.close()


def fig2_embeddings() -> None:
    tok = SimpleBPETokenizer(vocab_size=40).train(["the cat sat", "a dog ran"] * 5)
    emb = EmbeddingLayer(vocab_size=len(tok), hidden_dim=12, pad_id=tok.pad_id)
    text = "the cat sat"
    ids = tok.encode(text)
    vecs = emb.forward(np.array(ids))

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(vecs.T, aspect="auto", cmap="RdBu_r",
                   vmin=-abs(vecs).max(), vmax=abs(vecs).max())
    ax.set_yticks(range(emb.hidden_dim))
    ax.set_yticklabels([f"d{i}" for i in range(emb.hidden_dim)])
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([tok.id_to_token[i][:8] for i in ids], rotation=45, ha="right")
    ax.set_title("Figure 2 — Embedding Matrix: token IDs → dense vectors (12-d)")
    ax.set_xlabel("Token (position)")
    ax.set_ylabel("Embedding dimension")
    plt.colorbar(im, ax=ax, label="weight value")
    plt.tight_layout()
    plt.savefig("fig2_embeddings.png", dpi=120)
    plt.close()


def fig3_word2vec_pca() -> None:
    model, _ = scenario_word2vec_training()
    vecs = model.W
    words = [model.idx2word[i] for i in range(model.vocab_size)]
    pca = PCA(n_components=2, random_state=SEED)
    proj = pca.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj[:, 0], proj[:, 1], s=40, c="steelblue", edgecolor="black")
    for i, w in enumerate(words):
        ax.annotate(w, (proj[i, 0], proj[i, 1]), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_title("Figure 3 — PCA(2D) of Word2Vec Embeddings\n"
                 "Semantically related words should cluster together")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig3_word2vec_pca.png", dpi=120)
    plt.close()

    # Also save the loss curve.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(model.loss_history) + 1), model.loss_history,
            marker="o", color="darkorange")
    ax.set_title("Figure 4 — Word2Vec SGNS Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average negative-log-likelihood")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig4_loss_curve.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    fig1_tokenization()
    fig2_embeddings()
    fig3_word2vec_pca()
    print("Saved: fig1_tokenization.png, fig2_embeddings.png,")
    print("       fig3_word2vec_pca.png, fig4_loss_curve.png")

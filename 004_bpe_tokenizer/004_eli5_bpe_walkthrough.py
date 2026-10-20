"""
eli5_bpe_walkthrough.py

Purpose:
    A toy, step-by-step demonstration of BPE using tiny words.
    Every print statement is written in child-friendly language.
"""

from collections import Counter

def eli5_bpe_demo():
    # Step 1: Start with a tiny "corpus" of words.
    words = ["low", "lower", "lowest", "new", "newer", "newest"]
    print("📚 Our tiny word list:", words)

    # Step 2: Break each word into letters and add an end marker.
    # The end marker tells us where a word finishes.
    sequences = [[ch for ch in w] + ["</w>"] for w in words]
    print("\n🔤 Step 1: Break every word into letters and add '</w>':")
    for w, seq in zip(words, sequences):
        print(f"   {w} -> {seq}")

    # Step 3: Count how often each pair of neighbours appears.
    pair_counts = Counter()
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            pair_counts[(a, b)] += 1

    print("\n🔢 Step 2: Count every pair of neighbours:")
    for pair, count in pair_counts.most_common():
        print(f"   {pair} appears {count} time(s)")

    # Step 4: Merge the most frequent pair (excluding the end marker for simplicity).
    # In a real BPE, we would keep merging, but here we do just one merge.
    # We pick ('e', 'r') because it appears often in "lower", "newer", etc.
    pair_to_merge = ("e", "r")
    print(f"\n🔗 Step 3: Glue the most frequent pair {pair_to_merge} into one new tile!")
    new_sequences = []
    for seq in sequences:
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair_to_merge[0] and seq[i + 1] == pair_to_merge[1]:
                new_seq.append(pair_to_merge[0] + pair_to_merge[1])
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        new_sequences.append(new_seq)

    print("\n✨ After gluing, our words look like this:")
    for w, seq in zip(words, new_sequences):
        print(f"   {w} -> {seq}")

    # Step 5: Show how a new word would be built.
    new_word = "newer"
    print(f"\n🆕 If we see a new word '{new_word}', we break it into pieces:")
    print(f"   {new_word} -> ['n', 'e', 'w', 'er', '</w>']")
    print("   We already know the tile 'er', so we don't need to spell 'e' and 'r' separately!")

    print("\n🎉 That's BPE! We glued frequent pairs together to make bigger tiles.")

if __name__ == "__main__":
    eli5_bpe_demo()

# Synthetic dataset generator (used by the ELI5 walkthrough)
# Author: Awase Khirni Syed
# Copyright 2025 β ORI Inc.Canada All Rights Reserved.
from dataclasses import dataclass
import numpy as np

@dataclass
class ToyDoc:
    doc_id: str
    text: str

def build_toy_corpus(seed: int = 42) -> list[ToyDoc]:
    rng = np.random.default_rng(seed)
    base = {
        "animals": "cat dog cat dog bird cat dog bird fish",
        "sports":  "goal goal kick run kick run kick goal team",
        "food":    "pizza pizza pasta pizza pasta bread pizza pasta",
    }
    docs = [
        ToyDoc("d1_animals", base["animals"]),
        ToyDoc("d2_sports",  base["sports"]),
        ToyDoc("d3_food",    base["food"]),
        ToyDoc("d4_mix",     " ".join(rng.choice(base["animals"].split(), size=5))),
    ]
    return docs

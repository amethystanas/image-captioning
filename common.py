
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path("data")
CLIP_NAME = "openai/clip-vit-base-patch32"
PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIALS = [PAD, START, END, UNK]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str):
    """Minuscules, ponctuation retirée, découpage sur les espaces."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return text.split()


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.start_id = self.stoi[START]
        self.end_id = self.stoi[END]
        self.unk_id = self.stoi[UNK]

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len=None):
        """tokens -> [<start>, w1, ..., wn, <end>] (ids)."""
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if max_len is not None:
            ids = ids[: max_len - 2]
        return [self.start_id] + ids + [self.end_id]

    def decode(self, ids):
        words = []
        for i in ids:
            i = int(i)
            if i == self.end_id:
                break
            if i in (self.pad_id, self.start_id):
                continue
            words.append(self.itos[i])
        return " ".join(words)

    def save(self, path):
        Path(path).write_text(json.dumps({"itos": self.itos}))

    @classmethod
    def load(cls, path=DATA_DIR / "vocab.json"):
        return cls(json.loads(Path(path).read_text())["itos"])


def load_splits(path=DATA_DIR / "splits.json"):
    """{'train': [image_id, ...], 'val': [...], 'test': [...]}"""
    return json.loads(Path(path).read_text())


def load_captions(path=DATA_DIR / "captions.json"):
    """{image_id: [[token, ...], ...]} : 5 captions tokenisées par image."""
    return json.loads(Path(path).read_text())
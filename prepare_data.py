
import argparse
import json
from collections import Counter
from pathlib import Path

from common import DATA_DIR, SPECIALS, Vocab, tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--karpathy_json", required=True, help="dataset_flickr8k.json (splits Karpathy)")
    ap.add_argument("--min_freq", type=int, default=5)
    args = ap.parse_args()

    data = json.loads(Path(args.karpathy_json).read_text())
    splits = {"train": [], "val": [], "test": []}
    captions = {}

    for img in data["images"]:
        split = "train" if img["split"] == "restval" else img["split"]
        image_id = img["filename"]
        caps = [tokenize(s["raw"]) for s in img["sentences"]]
        caps = [c for c in caps if c]  # on retire les captions vides
        captions[image_id] = caps
        splits[split].append(image_id)

    # Vocabulaire construit sur le train uniquement (pas de fuite val/test)
    counter = Counter(tok for i in splits["train"] for cap in captions[i] for tok in cap)
    words = sorted(w for w, c in counter.items() if c >= args.min_freq)
    vocab = Vocab(SPECIALS + words)

    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "splits.json").write_text(json.dumps(splits))
    (DATA_DIR / "captions.json").write_text(json.dumps(captions))
    vocab.save(DATA_DIR / "vocab.json")

    n_caps = sum(len(captions[i]) for i in splits["train"])
    print({k: len(v) for k, v in splits.items()}, "images")
    print(f"{n_caps} captions train, vocab = {len(vocab)} mots (min_freq={args.min_freq})")
    lens = [len(c) for i in splits["train"] for c in captions[i]]
    print(f"longueur des captions : moy {sum(lens)/len(lens):.1f}, max {max(lens)}")


if __name__ == "__main__":
    main()
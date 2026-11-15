
# Calcule les métriques d'évaluation des captions générées.
import argparse
import json
import shutil

import torch

from common import CLIP_NAME, DATA_DIR, load_captions, tokenize


def _prep(preds, refs_all):
    missing = [i for i in preds if i not in refs_all]
    assert not missing, f"ids inconnus (ex. {missing[:3]})"
    res = {i: [" ".join(tokenize(c))] for i, c in preds.items()}
    gts = {i: [" ".join(c) for c in refs_all[i]] for i in preds}
    return gts, res


@torch.no_grad()
def clipscore(preds, device, feats_path=DATA_DIR / "features" / "clip_vit_b32.pt"):
    """CLIPScore (Hessel et al. 2021) = 2.5 * max(cos(img, "A photo depicts " + caption), 0)."""
    from transformers import CLIPModel, CLIPTokenizer

    feats = torch.load(feats_path)
    row = {i: k for k, i in enumerate(feats["ids"])}
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_NAME)

    ids = list(preds)
    img = feats["embeds"][[row[i] for i in ids]].float().to(device)
    img = img / img.norm(dim=-1, keepdim=True)
    texts = ["A photo depicts " + preds[i] for i in ids]
    scores = []
    for s in range(0, len(ids), 256):
        t = tok(texts[s : s + 256], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        pooled = model.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output
        te = model.text_projection(pooled)
        te = te / te.norm(dim=-1, keepdim=True)
        scores.append(2.5 * (img[s : s + 256] * te).sum(-1).clamp(min=0))
    return torch.cat(scores).mean().item()


def evaluate(preds, use_meteor=True, use_clipscore=True):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    gts, res = _prep(preds, load_captions())
    out = {}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    out.update({f"BLEU-{n}": b for n, b in enumerate(bleu, 1)})
    out["ROUGE-L"], _ = Rouge().compute_score(gts, res)
    out["CIDEr"], _ = Cider().compute_score(gts, res)

    if use_meteor and shutil.which("java") is None:
        print("[warn] Java introuvable : METEOR ignoré (utilise --no_meteor pour masquer ce message)")
        use_meteor, out["METEOR"] = False, None
    if use_meteor:
        try:
            from pycocoevalcap.meteor.meteor import Meteor

            out["METEOR"], _ = Meteor().compute_score(gts, res)
        except Exception as e:  # Java absent, etc.
            print(f"[warn] METEOR indisponible ({type(e).__name__}: {e})")
            out["METEOR"] = None
    if use_clipscore:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        out["CLIPScore"] = clipscore(preds, device)
    out["n_images"] = len(preds)
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_meteor", action="store_true")
    ap.add_argument("--no_clipscore", action="store_true")
    a = ap.parse_args()
    scores = evaluate(json.load(open(a.preds)), not a.no_meteor, not a.no_clipscore)
    print(json.dumps(scores, indent=2))
    if a.out:
        json.dump(scores, open(a.out, "w"), indent=2)
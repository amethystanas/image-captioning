"""Reranking CLIP des candidats de la beam search (semaine 2, partie A).

Idée : la beam search donne k phrases candidates par image. On score chaque candidat
avec CLIP (similarité image-texte, comme CLIPScore) et on garde le meilleur,
au lieu du premier candidat du modèle.

Entrée (--cands) : JSON {image_id: [candidat, ...]} où chaque candidat est soit
    "a dog runs on the grass"
soit {"caption": "a dog runs on the grass", "logprob": -7.3}
Le 1er candidat de chaque liste = le choix normal du modèle (baseline).

Usage :
    python clip_rerank.py --cands cands_test.json --out preds_reranked.json --eval --no_meteor
    # option : mélanger avec le score du modèle (nécessite "logprob") :
    python clip_rerank.py --cands cands_test.json --out preds_reranked.json --alpha 0.1

Sortie : JSON {image_id: caption}, directement utilisable par evaluate.py.
"""
import argparse
import json

import torch

from common import CLIP_NAME, DATA_DIR, tokenize


def load_cands(path):
    raw = json.load(open(path))
    return {
        i: [(c, None) if isinstance(c, str) else (c["caption"], c.get("logprob")) for c in lst]
        for i, lst in raw.items()
    }


@torch.no_grad()
def clip_scores(cands, device, feats_path=DATA_DIR / "features" / "clip_vit_b32.pt", bs=256):
    """Renvoie {image_id: [score CLIP de chaque candidat]} (2.5 * max(cos, 0))."""
    from transformers import CLIPModel, CLIPTokenizer

    feats = torch.load(feats_path)
    row = {i: k for k, i in enumerate(feats["ids"])}
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_NAME)

    flat = [(i, k, "A photo depicts " + c) for i, lst in cands.items() for k, (c, _) in enumerate(lst)]
    out = {i: [0.0] * len(lst) for i, lst in cands.items()}
    for s in range(0, len(flat), bs):
        chunk = flat[s : s + bs]
        t = tok([x[2] for x in chunk], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        pooled = model.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output
        te = model.text_projection(pooled)
        te = te / te.norm(dim=-1, keepdim=True)
        img = feats["embeds"][[row[x[0]] for x in chunk]].float().to(device)
        img = img / img.norm(dim=-1, keepdim=True)
        sc = 2.5 * (img * te).sum(-1).clamp(min=0)
        for (i, k, _), v in zip(chunk, sc.tolist()):
            out[i][k] = v
    return out


def rerank(cands, scores, alpha=0.0):
    """Choisit pour chaque image le candidat au meilleur score.
    score = CLIP + alpha * (logprob moyen par mot)   (alpha = 0 : CLIP seul)."""
    best = {}
    for i, lst in cands.items():
        def total(k):
            cap, lp = lst[k]
            s = scores[i][k]
            if alpha > 0 and lp is not None:
                s += alpha * lp / max(1, len(tokenize(cap)))
            return s

        best[i] = lst[max(range(len(lst)), key=total)][0]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--eval", action="store_true", help="évalue baseline (1er candidat) et reranké")
    ap.add_argument("--no_meteor", action="store_true")
    a = ap.parse_args()

    cands = load_cands(a.cands)
    if a.alpha > 0 and all(lp is None for lst in cands.values() for _, lp in lst):
        print("[warn] alpha > 0 mais aucun 'logprob' dans le fichier : reranking CLIP seul")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranked = rerank(cands, clip_scores(cands, device), a.alpha)
    baseline = {i: lst[0][0] for i, lst in cands.items()}

    changed = sum(reranked[i] != baseline[i] for i in cands)
    print(f"{changed}/{len(cands)} captions changées par le reranking")
    json.dump(reranked, open(a.out, "w"), indent=1)

    if a.eval:
        from evaluate import evaluate

        print("baseline (1er candidat) :", evaluate(baseline, use_meteor=not a.no_meteor))
        print("reranké CLIP            :", evaluate(reranked, use_meteor=not a.no_meteor))
        print("Attention : le CLIPScore monte mécaniquement (on optimise cette métrique).")
        print("Regarde surtout CIDEr / BLEU / ROUGE-L pour juger la qualité.")


if __name__ == "__main__":
    main()
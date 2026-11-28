# Partie D : montre les captions générées à côté des images, pour les montrer facilement à quelqu'un.
# Pour chaque image : la photo, une caption humaine, et la caption de chaque modèle donné.
#
# Usage :
#   python show_captions.py --preds results/preds_D_attn_test.json
#   python show_captions.py --preds results/preds_D_attn_test.json results/preds_D_noattn_test.json --n 12
#   python show_captions.py --preds results/preds_D_attn_test.json --images 1000268201_693b08cb0e.jpg 101669240_b2d3e7f17b.jpg
#   python show_captions.py --preds results/preds_D_attn_test.json --random      (images au hasard)
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre, on enregistre directement en PNG
import matplotlib.pyplot as plt
from PIL import Image

from common import load_captions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="un ou plusieurs fichiers preds_*.json")
    ap.add_argument("--img_dir", default="Images")
    ap.add_argument("--n", type=int, default=8, help="nombre d'images")
    ap.add_argument("--images", nargs="+", default=None, help="des image_id précis")
    ap.add_argument("--random", action="store_true", help="choisir les images au hasard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/captions.png")
    args = ap.parse_args()

    # Charger les captions de chaque modèle. Le nom affiché = nom du fichier sans "preds_" ni "_test".
    models = []
    for path in args.preds:
        name = Path(path).stem.replace("preds_", "").replace("_test", "")
        models.append((name, json.load(open(path))))
    human = load_captions()

    # Choisir les images
    if args.images is not None:
        ids = args.images
    else:
        ids = list(models[0][1].keys())
        if args.random:
            random.seed(args.seed)
            ids = random.sample(ids, args.n)
        else:
            ids = ids[: args.n]

    # Une ligne par image : la photo à gauche, le texte à droite
    fig, axes = plt.subplots(len(ids), 2, figsize=(12, 3 * len(ids)),
                             gridspec_kw={"width_ratios": [1, 2]}, squeeze=False)
    for k, img_id in enumerate(ids):
        img = Image.open(Path(args.img_dir) / img_id).convert("RGB")
        axes[k, 0].imshow(img)
        axes[k, 0].axis("off")
        axes[k, 0].set_title(img_id, fontsize=8)

        lines = ["humain : " + " ".join(human[img_id][0])]
        for name, preds in models:
            lines.append(name + " : " + preds.get(img_id, "(pas de caption)"))
        axes[k, 1].text(0, 0.5, "\n\n".join(lines), fontsize=11, va="center", wrap=True)
        axes[k, 1].axis("off")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=80)
    print("sauvegardé :", args.out)


if __name__ == "__main__":
    main()

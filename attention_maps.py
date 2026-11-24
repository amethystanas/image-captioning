# Partie D : affiche les cartes d'attention mot par mot (Show, Attend and Tell, fig. 3).
# Pour chaque mot généré, on agrandit les poids alpha (7x7) à la taille de l'image et on les superpose.
#
# Usage :
#   python attention_maps.py --ckpt checkpoints/D_attn.pt --img_dir Images --n 5
#   python attention_maps.py --ckpt checkpoints/D_attn.pt --img_dir Images --image 1000268201_693b08cb0e.jpg
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre, on enregistre directement en PNG
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from attention_lstm import AttentionCaptioner, get_device, load_grid
from common import Vocab, load_splits


def load_model(ckpt_path, vocab, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = AttentionCaptioner(len(vocab), emb_dim=cfg["emb_dim"], hidden=cfg["hidden"], att_dim=cfg["att_dim"],
                               dropout=cfg["dropout"], use_attention=not cfg["no_attention"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), cfg


def plot_one(model, img_id, grid, row, vocab, img_dir, out_path, device, max_len):
    feats = grid[[row[img_id]]].float().to(device)
    words, alphas = model.greedy(feats, vocab.start_id, vocab.end_id, max_len)
    words = words[0].cpu().tolist()
    alphas = alphas[0].cpu()  # [max_len, 49]

    # On garde les mots jusqu'à <end> (inclus, c'est intéressant de voir où il regarde)
    tokens = []
    for w in words:
        tokens.append(vocab.itos[w])
        if w == vocab.end_id:
            break

    # Même recadrage que pour les features (ResNet50_Weights.IMAGENET1K_V2 : resize 232, crop 224),
    # sinon la grille 7x7 ne correspond pas à la bonne zone de la photo.
    crop = transforms.Compose([transforms.Resize(232), transforms.CenterCrop(224)])
    img = crop(Image.open(Path(img_dir) / img_id).convert("RGB"))

    n = len(tokens) + 1
    cols = 5
    lines = (n + cols - 1) // cols
    fig, axes = plt.subplots(lines, cols, figsize=(3 * cols, 3 * lines))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(img)
    axes[0].set_title("image")
    for k, tok in enumerate(tokens):
        a = alphas[k].reshape(7, 7).numpy()
        # agrandir 7x7 -> 224x224 (interpolation bilinéaire via PIL)
        a_img = Image.fromarray((a / a.max() * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        axes[k + 1].imshow(img)
        axes[k + 1].imshow(np.array(a_img), cmap="jet", alpha=0.5)
        axes[k + 1].set_title(tok)

    fig.suptitle(img_id + " : " + vocab.decode(words))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("sauvegardé :", out_path, "->", vocab.decode(words))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/D_attn.pt")
    ap.add_argument("--img_dir", default="Images")
    ap.add_argument("--image", default=None, help="un image_id précis (sinon les n premières du test)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out_dir", default="results/attention_maps")
    args = ap.parse_args()

    device = get_device()
    vocab = Vocab.load()
    model, cfg = load_model(args.ckpt, vocab, device)
    if cfg["no_attention"]:
        print("[warn] ce modèle n'utilise pas l'attention : les cartes seront uniformes")
    grid, row = load_grid()

    if args.image is not None:
        ids = [args.image]
    else:
        ids = load_splits()["test"][: args.n]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_id in ids:
        out_path = out_dir / (Path(img_id).stem + ".png")
        plot_one(model, img_id, grid, row, vocab, args.img_dir, out_path, device, cfg["max_len"])


if __name__ == "__main__":
    main()

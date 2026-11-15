"""Étape 2 : cache des features gelées (une seule extraction pour toute l'équipe).

Usage :
    python extract_features.py --img_dir Flickr8k/Images

Produit dans data/features/ :
    resnet50.pt        {"ids": [...], "pooled": [N, 2048], "grid": [N, 49, 2048]}  (float16)
    clip_vit_b32.pt    {"ids": [...], "embeds": [N, 512]}  (float16, NON normalisé)
Les lignes suivent l'ordre de "ids" (identique dans les deux fichiers).
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

from common import CLIP_NAME, DATA_DIR, load_splits


class Images(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids, self.img_dir, self.transform = ids, Path(img_dir), transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.img_dir / self.ids[i]).convert("RGB")
        return self.transform(img)


class ClipTransform:
    """Classe (et non lambda) pour être picklable par les workers du DataLoader."""

    def __init__(self, proc):
        self.proc = proc

    def __call__(self, img):
        return self.proc(images=img, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def extract_resnet(ids, img_dir, device, bs, workers):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=weights).to(device).eval()
    body = torch.nn.Sequential(*list(net.children())[:-2])  # -> [B, 2048, 7, 7]
    loader = DataLoader(Images(ids, img_dir, weights.transforms()), batch_size=bs, num_workers=workers)
    pooled, grid = [], []
    for x in tqdm(loader, desc="ResNet-50"):
        f = body(x.to(device))  # [B, 2048, 7, 7]
        pooled.append(f.mean((2, 3)).half().cpu())
        grid.append(f.flatten(2).transpose(1, 2).half().cpu())  # [B, 49, 2048]
    return torch.cat(pooled), torch.cat(grid)


@torch.no_grad()
def extract_clip(ids, img_dir, device, bs, workers):
    from transformers import CLIPImageProcessor, CLIPModel

    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(CLIP_NAME)
    loader = DataLoader(Images(ids, img_dir, ClipTransform(proc)), batch_size=bs, num_workers=workers)
    out = []
    for x in tqdm(loader, desc="CLIP"):
        # Calcul explicite (les versions récentes de transformers renvoient un objet, pas un tenseur)
        pooled = model.vision_model(pixel_values=x.to(device)).pooler_output
        out.append(model.visual_projection(pooled).half().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_splits()
    ids = splits["train"] + splits["val"] + splits["test"]
    out_dir = DATA_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled, grid = extract_resnet(ids, args.img_dir, device, args.batch_size, args.workers)
    torch.save({"ids": ids, "pooled": pooled, "grid": grid}, out_dir / "resnet50.pt")

    embeds = extract_clip(ids, args.img_dir, device, args.batch_size, args.workers)
    torch.save({"ids": ids, "embeds": embeds}, out_dir / "clip_vit_b32.pt")
    print("OK :", pooled.shape, grid.shape, embeds.shape)


if __name__ == "__main__":
    main()
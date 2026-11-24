# Entraînement / évaluation du modèle CNN + LSTM sur Flickr8k.
#
# Exemples (depuis la racine du repo, après prepare_data.py) :
#   # CNN entraîné de zéro (images brutes, cache 128x128 construit au 1er lancement)
#   python train_cnn_lstm.py --encoder scratch --img_dir data/Flicker8k_Dataset --name A_cnn_scratch_lstm
#   # Baseline : features ResNet-50 pré-calculées (extract_features.py)
#   python train_cnn_lstm.py --encoder resnet --img_dir data/Flicker8k_Dataset --name B_lstm_resnet
#
# Sorties dans results/cnn_lstm/<name>/ :
#   history.csv / history.json   métriques par epoch (loss, perplexité, BLEU, CIDEr, lr, ...)
#   curves.png                   courbes d'entraînement (mises à jour à chaque epoch)
#   best.pt                      meilleur checkpoint (CIDEr val)
#   test_metrics.json            scores test (greedy + beam) avec METEOR / CLIPScore si dispo
#   preds_test_greedy.json, preds_test_beam.json, cands_test.json (pour clip_rerank.py)
#   examples.png                 exemples qualitatifs (image, prédiction, référence)
#   + un enregistrement dans results/runs/ (runlog.py)
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cnn_lstm import CaptionModel
from common import DATA_DIR, Vocab, load_captions, load_splits, set_seed
from runlog import RunLogger, count_params

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- données
def build_image_cache(ids, img_dir, size):
    """Charge toutes les images redimensionnées en size x size (uint8) dans un seul tenseur."""
    path = DATA_DIR / f"images_{size}.pt"
    if path.exists():
        cache = torch.load(path)
        if cache["ids"] == ids:
            return cache["images"]
        print("[cache] ids différents, reconstruction")
    if img_dir is None:
        raise SystemExit(f"{path} absent : passe --img_dir pour le construire")
    from PIL import Image

    imgs = torch.empty(len(ids), 3, size, size, dtype=torch.uint8)
    for k, i in enumerate(tqdm(ids, desc=f"cache images {size}px")):
        im = Image.open(Path(img_dir) / i).convert("RGB").resize((size, size), Image.BICUBIC)
        imgs[k] = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
    torch.save({"ids": ids, "images": imgs}, path)
    print(f"[cache] {path} ({imgs.numel() / 2**20:.0f} Mo)")
    return imgs


def encode_captions(image_ids, row_of, captions, vocab, max_len):
    """-> (rows [N], tokens [N, max_len] paddés). Une ligne = une (image, caption)."""
    rows, toks = [], []
    for i in image_ids:
        for cap in captions[i]:
            ids = vocab.encode(cap, max_len)
            rows.append(row_of[i])
            toks.append(ids + [vocab.pad_id] * (max_len - len(ids)))
    return torch.tensor(rows), torch.tensor(toks)


class ImageSource:
    """Fournit le batch d'entrée de l'encodeur pour des indices de lignes donnés."""

    def __init__(self, encoder, images=None, feats=None, crop=None, device="cpu"):
        self.encoder, self.crop, self.device = encoder, crop, device
        if encoder == "scratch":
            self.data = images  # uint8 [N, 3, S, S] (CPU, pinned si possible)
            self.mean, self.std = MEAN.to(device), STD.to(device)
        else:
            self.data = feats.to(device)  # [N, 2048] fp16

    def __call__(self, rows, train):
        if self.encoder != "scratch":
            return self.data[rows.to(self.device)]
        x = self.data[rows].to(self.device, non_blocking=True).float().div_(255)
        S, c = x.size(-1), self.crop
        if train:  # augmentation : crop aléatoire + flip horizontal (par image)
            out = torch.empty(x.size(0), 3, c, c, device=self.device)
            oy = torch.randint(0, S - c + 1, (x.size(0),)).tolist()
            ox = torch.randint(0, S - c + 1, (x.size(0),)).tolist()
            for k in range(x.size(0)):
                out[k] = x[k, :, oy[k] : oy[k] + c, ox[k] : ox[k] + c]
            flip = torch.rand(x.size(0), device=self.device) < 0.5
            out[flip] = out[flip].flip(-1)
            x = out
        else:  # crop central
            o = (S - c) // 2
            x = x[:, :, o : o + c, o : o + c]
        return (x - self.mean) / self.std


# --------------------------------------------------------------------------- boucles
def run_epoch(model, src, rows, toks, opt, args, train, scaler=None):
    """Une passe sur (rows, toks). Renvoie loss moyenne par token, accuracy token, norme du gradient moyenne."""
    model.train(train)
    perm = torch.randperm(len(rows)) if train else torch.arange(len(rows))
    tot_loss, tot_tok, tot_ok, gnorms = 0.0, 0, 0, []
    for s in tqdm(range(0, len(rows), args.batch_size), desc="train" if train else "val", leave=False):
        idx = perm[s : s + args.batch_size]
        t = toks[idx].to(args.device)
        L = int((t != 0).sum(1).max())
        t = t[:, :L]
        inp, tgt = t[:, :-1], t[:, 1:]
        with torch.set_grad_enabled(train), torch.autocast("cuda", enabled=args.amp):
            logits = model(src(rows[idx], train), inp)
            mask = tgt != 0
            loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1),
                                       ignore_index=0, reduction="sum")
        n = int(mask.sum())
        if train:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1),
                                   ignore_index=0, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorms.append(float(nn.utils.clip_grad_norm_(model.parameters(), args.clip)))
            scaler.step(opt)
            scaler.update()
        tot_loss += float(loss_sum)
        tot_tok += n
        tot_ok += int(((logits.argmax(-1) == tgt) & mask).sum())
    return tot_loss / tot_tok, tot_ok / tot_tok, (float(np.mean(gnorms)) if gnorms else None)


@torch.no_grad()
def generate(model, src, rows, image_ids, vocab, args, beam=1):
    """Greedy (beam=1, par batch) ou beam search (image par image). -> preds {id: caption}, cands."""
    model.eval()
    preds, cands = {}, {}
    if beam == 1:
        for s in range(0, len(rows), 256):
            out = model.greedy(src(rows[s : s + 256], False), vocab.start_id, vocab.end_id, args.max_len)
            for i, seq in zip(image_ids[s : s + 256], out.tolist()):
                preds[i] = vocab.decode(seq)
        return preds, None
    for k, i in enumerate(tqdm(image_ids, desc=f"beam {beam}", leave=False)):
        res = model.beam_search(src(rows[k : k + 1], False), vocab.start_id, vocab.end_id, beam, args.max_len)
        cands[i] = [{"caption": vocab.decode(seq), "logprob": round(lp, 4)} for seq, lp in res]
        preds[i] = cands[i][0]["caption"]
    return preds, cands


def quick_scores(preds):
    """BLEU / ROUGE-L / CIDEr seulement (rapide, sans Java ni CLIP) : utilisé à chaque epoch."""
    from evaluate import evaluate

    s = evaluate(preds, use_meteor=False, use_clipscore=False)
    s.pop("METEOR", None)
    s.pop("n_images", None)
    return s


def diversity(preds):
    """Nb de phrases distinctes / nb d'images et taille du vocabulaire utilisé (détecte le mode collapse)."""
    caps = list(preds.values())
    return len(set(caps)) / len(caps), len({w for c in caps for w in c.split()})


# --------------------------------------------------------------------------- figures
COLORS = {"train": "#2a78d6", "val": "#eb6834", "c3": "#1baf7a", "c4": "#6250d6"}


def _style(ax, title, xlabel="epoch"):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(True, color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#a3a29d")


def plot_history(hist, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in hist]
    best = max(hist, key=lambda h: h["val_CIDEr"])["epoch"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    kw = dict(linewidth=2, marker="o", markersize=4)

    def line(ax, key, label, color):
        ax.plot(ep, [h[key] for h in hist], label=label, color=color, **kw)

    ax = axs[0, 0]
    line(ax, "train_loss", "train", COLORS["train"])
    line(ax, "val_loss", "val", COLORS["val"])
    _style(ax, "Cross-entropy par token")
    ax.legend(frameon=False)

    ax = axs[0, 1]
    line(ax, "train_ppl", "train", COLORS["train"])
    line(ax, "val_ppl", "val", COLORS["val"])
    _style(ax, "Perplexité")
    ax.set_yscale("log")
    ax.legend(frameon=False)

    ax = axs[0, 2]
    line(ax, "train_acc", "train", COLORS["train"])
    line(ax, "val_acc", "val", COLORS["val"])
    _style(ax, "Accuracy du mot suivant (teacher forcing)")
    ax.legend(frameon=False)

    ax = axs[1, 0]
    line(ax, "val_CIDEr", "CIDEr (val, greedy)", COLORS["val"])
    _style(ax, "CIDEr val (critère du meilleur modèle)")
    ax.axvline(best, color="#a3a29d", linestyle="--", linewidth=1)
    ax.annotate(f"best = {max(h['val_CIDEr'] for h in hist):.3f} (ep {best})", (best, max(h["val_CIDEr"] for h in hist)),
                textcoords="offset points", xytext=(6, -12), fontsize=9, color="#52514e")

    ax = axs[1, 1]
    for k, c in zip(("val_BLEU-1", "val_BLEU-4", "val_ROUGE-L"), (COLORS["train"], COLORS["val"], COLORS["c3"])):
        line(ax, k, k.replace("val_", ""), c)
    _style(ax, "BLEU / ROUGE-L val (greedy)")
    ax.legend(frameon=False)

    ax = axs[1, 2]
    line(ax, "lr", "learning rate", COLORS["c4"])
    _style(ax, "Learning rate")
    ax.set_yscale("log")

    fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_examples(images, rows, ids, preds, captions, path, n=12, seed=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    cols = 4
    fig, axs = plt.subplots(math.ceil(len(pick) / cols), cols, figsize=(4.5 * cols, 4.6 * math.ceil(len(pick) / cols)))
    for ax, k in zip(axs.flat, pick):
        ax.imshow(images[rows[k]].permute(1, 2, 0).numpy())
        ax.axis("off")
        ref = " ".join(captions[ids[k]][0])
        ax.set_title(f"PRED: {preds[ids[k]]}\nREF:  {ref}", fontsize=8, loc="left", wrap=True)
    for ax in list(axs.flat)[len(pick):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["scratch", "resnet"], default="scratch")
    ap.add_argument("--name", default=None, help="nom du run (défaut : cnn_lstm_<encoder>)")
    ap.add_argument("--img_dir", default=None, help="dossier des images (scratch : construit le cache)")
    ap.add_argument("--img_size", type=int, default=144, help="taille du cache (scratch)")
    ap.add_argument("--crop", type=int, default=128, help="taille du crop vu par le CNN (scratch)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_embed", type=int, default=512)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=5.0, help="clipping de la norme du gradient")
    ap.add_argument("--max_len", type=int, default=22, help="longueur max (avec <start>/<end>)")
    ap.add_argument("--patience", type=int, default=6, help="early stopping sur CIDEr val")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_meteor", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="debug : nb max d'images par split")
    args = ap.parse_args()

    set_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.amp = args.device == "cuda" and not args.no_amp
    name = args.name or f"cnn_lstm_{args.encoder}"
    out = Path("results/cnn_lstm") / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device}  amp={args.amp}  -> {out}")

    vocab, captions, splits = Vocab.load(), load_captions(), load_splits()
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}
    all_ids = splits["train"] + splits["val"] + splits["test"]

    # --- source des images
    images = None
    if args.encoder == "scratch":
        full = load_splits()
        images = build_image_cache(full["train"] + full["val"] + full["test"], args.img_dir, args.img_size)
        cache_ids = full["train"] + full["val"] + full["test"]
        row_of = {i: k for k, i in enumerate(cache_ids)}
        if args.device == "cuda":
            images = images.pin_memory()
        src = ImageSource("scratch", images=images, crop=args.crop, device=args.device)
    else:
        feats = torch.load(DATA_DIR / "features" / "resnet50.pt")
        row_of = {i: k for k, i in enumerate(feats["ids"])}
        src = ImageSource("resnet", feats=feats["pooled"], device=args.device)
        if args.img_dir or (DATA_DIR / f"images_{args.img_size}.pt").exists():
            full = load_splits()
            try:
                images = build_image_cache(full["train"] + full["val"] + full["test"], args.img_dir, args.img_size)
            except SystemExit:
                images = None
            img_row_of = {i: k for k, i in enumerate(full["train"] + full["val"] + full["test"])}

    tr_rows, tr_toks = encode_captions(splits["train"], row_of, captions, vocab, args.max_len)
    va_rows, va_toks = encode_captions(splits["val"], row_of, captions, vocab, args.max_len)
    va_img_rows = torch.tensor([row_of[i] for i in splits["val"]])
    te_img_rows = torch.tensor([row_of[i] for i in splits["test"]])
    print(f"train : {len(splits['train'])} images / {len(tr_rows)} captions | val : {len(splits['val'])} | "
          f"test : {len(splits['test'])} | vocab {len(vocab)}")

    model = CaptionModel(len(vocab), args.encoder, args.img_embed, args.embed_dim, args.hidden_dim,
                         args.num_layers, args.dropout).to(args.device)
    tot, trn = count_params(model)
    enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"paramètres : {tot / 1e6:.2f} M (encodeur {enc / 1e6:.2f} M, décodeur {(tot - enc) / 1e6:.2f} M)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    config = {k: v for k, v in vars(args).items() if k not in ("device",)}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    with RunLogger(name, model=model, **config) as run:
        hist, best, bad = [], -1.0, 0
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, gnorm = run_epoch(model, src, tr_rows, tr_toks, opt, args, True, scaler)
            va_loss, va_acc, _ = run_epoch(model, src, va_rows, va_toks, None, args, False)
            preds_val, _ = generate(model, src, va_img_rows, splits["val"], vocab, args, beam=1)
            sc = quick_scores(preds_val)
            div, used = diversity(preds_val)
            h = {
                "epoch": ep, "lr": opt.param_groups[0]["lr"],
                "train_loss": tr_loss, "val_loss": va_loss,
                "train_ppl": math.exp(tr_loss), "val_ppl": math.exp(va_loss),
                "train_acc": tr_acc, "val_acc": va_acc, "grad_norm": gnorm,
                **{f"val_{k}": v for k, v in sc.items()},
                "val_distinct": div, "val_vocab_used": used,
                "time_s": round(time.time() - t0, 1),
            }
            hist.append(h)
            sched.step(sc["CIDEr"])

            improved = sc["CIDEr"] > best
            if improved:
                best, bad = sc["CIDEr"], 0
                torch.save({"model": model.state_dict(), "config": config, "epoch": ep, "itos": vocab.itos},
                           out / "best.pt")
            else:
                bad += 1
            print(f"ep {ep:3d} | loss {tr_loss:.3f}/{va_loss:.3f} | ppl {h['train_ppl']:.1f}/{h['val_ppl']:.1f} | "
                  f"acc {tr_acc:.3f}/{va_acc:.3f} | B4 {sc['BLEU-4']:.4f} | CIDEr {sc['CIDEr']:.4f}"
                  f"{' *' if improved else ''} | distinct {div:.2f} | lr {h['lr']:.1e} | {h['time_s']}s")
            print(f"        ex : {next(iter(preds_val.values()))}")

            (out / "history.json").write_text(json.dumps(hist, indent=1))
            with open(out / "history.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(h))
                w.writeheader()
                w.writerows(hist)
            plot_history(hist, out / "curves.png", f"{name} — encodeur {args.encoder}")
            if bad >= args.patience:
                print(f"early stopping (pas d'amélioration du CIDEr val depuis {args.patience} epochs)")
                break

        # --- test avec le meilleur modèle
        ck = torch.load(out / "best.pt", map_location=args.device)
        model.load_state_dict(ck["model"])
        print(f"\nmeilleur modèle : epoch {ck['epoch']} (CIDEr val {best:.4f}) -> évaluation test")
        from evaluate import evaluate

        clip_ok = (DATA_DIR / "features" / "clip_vit_b32.pt").exists()
        results = {"best_epoch": ck["epoch"], "best_val_CIDEr": best, "epochs_run": len(hist)}
        p_greedy, _ = generate(model, src, te_img_rows, splits["test"], vocab, args, beam=1)
        results["test_greedy"] = evaluate(p_greedy, not args.no_meteor, clip_ok)
        p_beam, cands = generate(model, src, te_img_rows, splits["test"], vocab, args, beam=args.beam)
        results["test_beam"] = evaluate(p_beam, not args.no_meteor, clip_ok)
        for tag, p in (("greedy", p_greedy), ("beam", p_beam)):
            d, u = diversity(p)
            results[f"test_{tag}"].update({"distinct": round(d, 4), "vocab_used": u,
                                           "avg_len": round(float(np.mean([len(c.split()) for c in p.values()])), 2)})

        json.dump(p_greedy, open(out / "preds_test_greedy.json", "w"), indent=1)
        json.dump(p_beam, open(out / "preds_test_beam.json", "w"), indent=1)
        json.dump(cands, open(out / "cands_test.json", "w"), indent=1)
        (out / "test_metrics.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        run.add(best_epoch=ck["epoch"], **{k: v for k, v in results["test_beam"].items()},
                greedy_CIDEr=results["test_greedy"]["CIDEr"])

        if images is not None:
            rows_img = te_img_rows if args.encoder == "scratch" else torch.tensor([img_row_of[i] for i in splits["test"]])
            plot_examples(images, rows_img.tolist(), splits["test"], p_beam, captions, out / "examples.png")
        print(f"\nterminé. Résultats dans {out}/")


if __name__ == "__main__":
    main()

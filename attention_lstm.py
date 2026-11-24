# LSTM avec attention soft (Show, Attend and Tell, Xu et al. 2015).
# On utilise la grille 7x7 de ResNet-50 (49 vecteurs de 2048) déjà extraite par extract_features.py.
# A chaque mot, le modèle calcule des poids sur les 49 cases et fait une moyenne pondérée.
#
# Avec --no_attention : même décodeur mais le contexte est toujours la moyenne des 49 cases
# (= le vecteur ResNet poolé), pour comparer avec / sans attention.
#
# Usage :
#   python attention_lstm.py --epochs 10
#   python attention_lstm.py --epochs 10 --no_attention
#   python attention_lstm.py --epochs 1 --n_train 500      (petit test rapide)
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from common import DATA_DIR, Vocab, load_captions, load_splits, set_seed
from evaluate import evaluate
from runlog import RunLogger


class Attention(nn.Module):
    """Attention additive : score = w * tanh(W_f * feats + W_h * h)."""

    def __init__(self, feat_dim, hidden, att_dim):
        super().__init__()
        self.feat_proj = nn.Linear(in_features=feat_dim, out_features=att_dim)
        self.h_proj = nn.Linear(in_features=hidden, out_features=att_dim)
        self.score = nn.Linear(in_features=att_dim, out_features=1)

    def forward(self, feats, h):
        # feats : [B, 49, 2048], h : [B, hidden]
        a = self.feat_proj(feats)                     # [B, 49, att_dim]
        b = self.h_proj(h).unsqueeze(1)               # [B, 1, att_dim]
        e = self.score(torch.tanh(a + b)).squeeze(2)  # [B, 49]
        alpha = torch.softmax(e, dim=1)               # [B, 49], somme = 1
        context = (feats * alpha.unsqueeze(2)).sum(dim=1)  # [B, 2048]
        return context, alpha


class AttentionCaptioner(nn.Module):
    def __init__(self, vocab_size, feat_dim=2048, emb_dim=256, hidden=512, att_dim=256,
                 dropout=0.5, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.attention = Attention(feat_dim, hidden, att_dim)
        self.init_h = nn.Linear(in_features=feat_dim, out_features=hidden)
        self.init_c = nn.Linear(in_features=feat_dim, out_features=hidden)
        # entrée du LSTM = embedding du mot + contexte visuel
        self.lstm = nn.LSTMCell(input_size=emb_dim + feat_dim, hidden_size=hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=hidden, out_features=vocab_size)

    def init_state(self, feats):
        mean = feats.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)

    def step(self, feats, word, h, c):
        """Un pas de temps : mot courant -> scores du mot suivant."""
        if self.use_attention:
            context, alpha = self.attention(feats, h)
        else:
            context = feats.mean(dim=1)
            alpha = torch.full((feats.size(0), feats.size(1)), 1.0 / feats.size(1), device=feats.device)
        x = torch.cat([self.embed(word), context], dim=1)
        h, c = self.lstm(x, (h, c))
        logits = self.fc(self.dropout(h))
        return logits, h, c, alpha

    def forward(self, feats, caps):
        """Teacher forcing : on donne le vrai mot t pour prédire le mot t+1."""
        h, c = self.init_state(feats)
        all_logits = []
        all_alphas = []
        for t in range(caps.size(1) - 1):
            logits, h, c, alpha = self.step(feats, caps[:, t], h, c)
            all_logits.append(logits)
            all_alphas.append(alpha)
        return torch.stack(all_logits, dim=1), torch.stack(all_alphas, dim=1)  # [B, T-1, V], [B, T-1, 49]

    @torch.no_grad()
    def greedy(self, feats, start_id, end_id, max_len=20):
        """Décodage glouton : à chaque pas on prend le mot le plus probable."""
        h, c = self.init_state(feats)
        word = torch.full((feats.size(0),), start_id, dtype=torch.long, device=feats.device)
        words = []
        alphas = []
        for t in range(max_len):
            logits, h, c, alpha = self.step(feats, word, h, c)
            word = logits.argmax(dim=1)
            words.append(word)
            alphas.append(alpha)
        return torch.stack(words, dim=1), torch.stack(alphas, dim=1)  # [B, max_len], [B, max_len, 49]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_grid():
    feats = torch.load(DATA_DIR / "features" / "resnet50.pt")
    row = {img_id: k for k, img_id in enumerate(feats["ids"])}
    return feats["grid"], row  # grid : [N, 49, 2048] en float16


def make_pairs(image_ids, captions, vocab, max_len):
    """Une paire (image, caption) par caption -> 5 paires par image."""
    pairs = []
    for img_id in image_ids:
        for cap in captions[img_id]:
            pairs.append((img_id, vocab.encode(cap, max_len=max_len)))
    return pairs


def make_batch(pairs, grid, row, pad_id, device):
    longest = max(len(ids) for _, ids in pairs)
    caps = torch.full((len(pairs), longest), pad_id, dtype=torch.long)
    for k, (_, ids) in enumerate(pairs):
        caps[k, : len(ids)] = torch.tensor(ids)
    rows = [row[img_id] for img_id, _ in pairs]
    feats = grid[rows].float()  # on ne convertit en float32 que le batch (sinon ~3 Go)
    return feats.to(device), caps.to(device)


def generate(model, image_ids, grid, row, vocab, device, batch_size=100, max_len=20):
    """Renvoie {image_id: caption} au format attendu par evaluate.py."""
    model.eval()
    preds = {}
    for s in range(0, len(image_ids), batch_size):
        ids = image_ids[s : s + batch_size]
        feats = grid[[row[i] for i in ids]].float().to(device)
        words, _ = model.greedy(feats, vocab.start_id, vocab.end_id, max_len)
        for img_id, w in zip(ids, words.cpu()):
            preds[img_id] = vocab.decode(w)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--att_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--att_reg", type=float, default=1.0, help="lambda de la régularisation 'doubly stochastic' (0 = off)")
    ap.add_argument("--max_len", type=int, default=22, help="longueur max des captions (avec <start> et <end>)")
    ap.add_argument("--no_attention", action="store_true", help="contexte = moyenne des 49 cases (LSTM simple)")
    ap.add_argument("--n_train", type=int, default=None, help="n'utiliser que les n premières images train (test rapide)")
    ap.add_argument("--no_meteor", action="store_true", help="ne pas calculer METEOR (bloque si Java n'est pas vraiment installé)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="nom du run (défaut : D_attn ou D_noattn)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("device :", device)

    vocab = Vocab.load()
    splits = load_splits()
    captions = load_captions()
    grid, row = load_grid()

    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]
    if args.n_train is not None:
        train_ids = train_ids[: args.n_train]
        val_ids = val_ids[:200]
        test_ids = test_ids[:200]
    train_pairs = make_pairs(train_ids, captions, vocab, args.max_len)
    print(len(train_pairs), "paires (image, caption) pour l'entraînement")

    model = AttentionCaptioner(len(vocab), emb_dim=args.emb_dim, hidden=args.hidden, att_dim=args.att_dim,
                               dropout=args.dropout, use_attention=not args.no_attention).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    name = args.name
    if name is None:
        name = "D_noattn" if args.no_attention else "D_attn"
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}.pt"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    config = dict(vars(args))
    del config["name"]  # 'name' est déjà le 1er argument de RunLogger
    with RunLogger(name, model=model, **config) as run:
        best_cider = -1.0
        best_epoch = -1
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(len(train_pairs)).tolist()
            total_loss = 0.0
            n_batches = 0
            for s in range(0, len(perm), args.batch_size):
                batch = [train_pairs[k] for k in perm[s : s + args.batch_size]]
                feats, caps = make_batch(batch, grid, row, vocab.pad_id, device)

                logits, alphas = model(feats, caps)
                targets = caps[:, 1:]  # le mot à prédire = le mot suivant
                loss = loss_fn(logits.reshape(-1, logits.size(2)), targets.reshape(-1))

                if args.att_reg > 0 and not args.no_attention:
                    # Chaque case doit recevoir à peu près autant d'attention au total sur la phrase
                    # (eq. 14 du papier). On ne compte que les pas qui ne sont pas du padding.
                    mask = (targets != vocab.pad_id).float().unsqueeze(2)  # [B, T-1, 1]
                    att_sum = (alphas * mask).sum(dim=1)                  # [B, 49]
                    loss = loss + args.att_reg * ((1.0 - att_sum) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                if n_batches % 100 == 0:
                    print(f"  epoch {epoch} batch {n_batches} loss {total_loss / n_batches:.3f}")

            # Validation : on choisit la meilleure epoch avec le CIDEr sur val (jamais sur test)
            val_preds = generate(model, val_ids, grid, row, vocab, device, max_len=args.max_len)
            val_scores = evaluate(val_preds, use_meteor=False, use_clipscore=False)
            train_loss = total_loss / n_batches
            print(f"epoch {epoch} : train loss {train_loss:.3f}, val BLEU-4 {val_scores['BLEU-4']}, val CIDEr {val_scores['CIDEr']}")
            print("  exemple :", val_ids[0], "->", val_preds[val_ids[0]])
            history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                            "val_BLEU-4": val_scores["BLEU-4"], "val_CIDEr": val_scores["CIDEr"]})

            if val_scores["CIDEr"] > best_cider:
                best_cider = val_scores["CIDEr"]
                best_epoch = epoch
                torch.save({"state_dict": model.state_dict(), "config": vars(args)}, ckpt_path)
                print("  -> meilleur modèle sauvegardé :", ckpt_path)

        # Test final avec le meilleur checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        test_preds = generate(model, test_ids, grid, row, vocab, device, max_len=args.max_len)
        preds_path = results_dir / f"preds_{name}_test.json"
        json.dump(test_preds, open(preds_path, "w"), indent=1)
        test_scores = evaluate(test_preds, use_meteor=not args.no_meteor, use_clipscore=True)
        print("TEST :", json.dumps(test_scores, indent=2))

        run.add(**test_scores)
        run.add(best_epoch=best_epoch, best_val_CIDEr=best_cider, history=history)


if __name__ == "__main__":
    main()

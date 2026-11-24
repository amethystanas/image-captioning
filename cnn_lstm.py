# Modèle CNN + LSTM (Show and Tell, Vinyals et al. 2015), écrit de zéro.
#
# Deux encodeurs possibles :
#   - "scratch" : un CNN défini ici, initialisé aléatoirement et entraîné de bout en bout
#                 sur les images Flickr8k (aucun poids pré-entraîné).
#   - "resnet"  : les features ResNet-50 pré-calculées (data/features/resnet50.pt, vecteur 2048)
#                 projetées par une couche linéaire (baseline "transfer learning").
# Le décodeur LSTM est le même dans les deux cas : l'image initialise (h0, c0),
# puis le LSTM génère la phrase mot par mot.
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- encodeurs
class ConvBlock(nn.Module):
    """2 x (conv 3x3 -> BN -> ReLU) puis max-pool 2x2 (style VGG)."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class ScratchCNN(nn.Module):
    """CNN entraîné de zéro. Entrée [B, 3, H, W] (H = W = 128 par défaut) -> [B, embed_dim].

    128 -> 64 -> 32 -> 16 -> 8 -> 4 puis global average pooling.
    """

    def __init__(self, embed_dim=512, widths=(32, 64, 128, 256, 512), dropout=0.3):
        super().__init__()
        blocks, c = [], 3
        for w in widths:
            blocks.append(ConvBlock(c, w))
            c = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c, embed_dim))
        self.out_dim = embed_dim
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        f = self.features(x)  # [B, C, 4, 4]
        return self.head(f.mean((2, 3)))


class FeatEncoder(nn.Module):
    """Projette un vecteur de features pré-calculé (ResNet-50 : 2048) -> embed_dim."""

    def __init__(self, in_dim=2048, embed_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, embed_dim))
        self.out_dim = embed_dim

    def forward(self, x):
        return self.net(x.float())


# --------------------------------------------------------------------------- décodeur
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, img_dim, embed_dim=256, hidden_dim=512, num_layers=1, dropout=0.3):
        super().__init__()
        self.num_layers, self.hidden_dim = num_layers, hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.init_h = nn.Linear(img_dim, num_layers * hidden_dim)
        self.init_c = nn.Linear(img_dim, num_layers * hidden_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, img):
        B = img.size(0)
        h = torch.tanh(self.init_h(img)).view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c = torch.tanh(self.init_c(img)).view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        return h, c

    def forward(self, img, tokens):
        """Teacher forcing. tokens = [<start>, w1, ..., wn] -> logits [B, T, V]."""
        out, _ = self.lstm(self.drop(self.embed(tokens)), self.init_state(img))
        return self.fc(self.drop(out))

    def step(self, tok, state):
        """Un pas de décodage : tok [B] -> log-probs [B, V]."""
        out, state = self.lstm(self.embed(tok).unsqueeze(1), state)
        return F.log_softmax(self.fc(out.squeeze(1)), -1), state


# --------------------------------------------------------------------------- modèle complet
class CaptionModel(nn.Module):
    def __init__(self, vocab_size, encoder="scratch", img_embed=512, embed_dim=256, hidden_dim=512,
                 num_layers=1, dropout=0.3, feat_dim=2048):
        super().__init__()
        if encoder == "scratch":
            self.encoder = ScratchCNN(img_embed, dropout=dropout)
        elif encoder == "resnet":
            self.encoder = FeatEncoder(feat_dim, img_embed, dropout)
        else:
            raise ValueError(encoder)
        self.decoder = LSTMDecoder(vocab_size, img_embed, embed_dim, hidden_dim, num_layers, dropout)

    def forward(self, images, tokens):
        return self.decoder(self.encoder(images), tokens)

    @torch.no_grad()
    def greedy(self, images, start_id, end_id, max_len=20):
        img = self.encoder(images)
        state = self.decoder.init_state(img)
        tok = torch.full((img.size(0),), start_id, dtype=torch.long, device=img.device)
        out, done = [], torch.zeros_like(tok, dtype=torch.bool)
        for _ in range(max_len):
            logp, state = self.decoder.step(tok, state)
            tok = logp.argmax(-1)
            out.append(tok)
            done |= tok == end_id
            if done.all():
                break
        return torch.stack(out, 1)  # [B, L]

    @torch.no_grad()
    def beam_search(self, image, start_id, end_id, beam=3, max_len=20, len_norm=0.7):
        """Beam search pour UNE image ([1, ...]). Renvoie [(ids, logprob), ...] triés (meilleur d'abord).

        Score de classement = logprob / longueur**len_norm (pénalise moins les phrases longues).
        """
        img = self.encoder(image)  # [1, D]
        h, c = self.decoder.init_state(img)
        beams = [([], 0.0, (h, c))]
        finished = []
        for _ in range(max_len):
            if not beams:
                break
            toks = torch.tensor([b[0][-1] if b[0] else start_id for b in beams], device=img.device)
            hs = torch.cat([b[2][0] for b in beams], 1)
            cs = torch.cat([b[2][1] for b in beams], 1)
            logp, (hs, cs) = self.decoder.step(toks, (hs, cs))
            scores = torch.tensor([b[1] for b in beams], device=img.device).unsqueeze(1) + logp
            top_v, top_i = scores.view(-1).topk(min(beam * 2, scores.numel()))
            V = logp.size(1)
            new = []
            for v, i in zip(top_v.tolist(), top_i.tolist()):
                b, w = divmod(i, V)
                seq = beams[b][0] + [w]
                st = (hs[:, b : b + 1], cs[:, b : b + 1])
                if w == end_id:
                    finished.append((seq[:-1], v))
                else:
                    new.append((seq, v, st))
                if len(new) == beam:
                    break
            beams = new
            if len(finished) >= beam:
                break
        finished += [(b[0], b[1]) for b in beams]
        finished.sort(key=lambda x: x[1] / max(1, len(x[0])) ** len_norm, reverse=True)
        return finished[:beam]

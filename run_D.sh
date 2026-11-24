#!/bin/bash
# Partie D, semaine 2 : réglage des hyperparamètres + comparaison avec / sans attention.
# Chaque run est enregistré dans results/runs/ (voir : python runlog.py)
set -e

EPOCHS=10
# Sans Java installé (Mac) METEOR bloque : on le désactive. Mettre EXTRA="" si Java est installé.
EXTRA="--no_meteor"

# 1) comparaison principale : attention vs LSTM simple (même décodeur, contexte = moyenne)
python attention_lstm.py --epochs $EPOCHS $EXTRA --name D_attn
python attention_lstm.py --epochs $EPOCHS $EXTRA --no_attention --name D_noattn

# 2) réglages (un paramètre à la fois par rapport au run D_attn)
python attention_lstm.py --epochs $EPOCHS $EXTRA --att_reg 0 --name D_attn_noreg
python attention_lstm.py --epochs $EPOCHS $EXTRA --lr 1e-3 --name D_attn_lr1e-3
python attention_lstm.py --epochs $EPOCHS $EXTRA --dropout 0.3 --name D_attn_drop0.3
python attention_lstm.py --epochs $EPOCHS $EXTRA --hidden 1024 --name D_attn_h1024

# 3) cartes d'attention du meilleur modèle
python attention_maps.py --ckpt checkpoints/D_attn.pt --n 10

python runlog.py

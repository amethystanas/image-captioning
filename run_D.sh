#!/bin/bash
# Partie D, semaine 2 : réglages + seeds supplémentaires
set -e
EPOCHS=10

# réglages (un paramètre à la fois par rapport à D_attn)
python attention_lstm.py --epochs $EPOCHS --att_reg 0 --name D_attn_noreg
python attention_lstm.py --epochs $EPOCHS --lr 1e-3 --name D_attn_lr1e-3
python attention_lstm.py --epochs $EPOCHS --dropout 0.3 --name D_attn_drop0.3
python attention_lstm.py --epochs $EPOCHS --hidden 1024 --name D_attn_h1024

# autres seeds pour la comparaison avec / sans attention
python attention_lstm.py --epochs $EPOCHS --seed 1 --name D_attn_s1
python attention_lstm.py --epochs $EPOCHS --seed 1 --no_attention --name D_noattn_s1
python attention_lstm.py --epochs $EPOCHS --seed 2 --name D_attn_s2
python attention_lstm.py --epochs $EPOCHS --seed 2 --no_attention --name D_noattn_s2

python runlog.py
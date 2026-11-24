# image-captioning
Image captioning on Flickr8k: comparing a CNN+LSTM baseline, an attention-based LSTM (Show, Attend and Tell) and a CLIP + GPT-2 prefix model (ClipCap-style). Computer vision project, Bordeaux INP – ENSEIRB-MATMECA, 2026–27.


## CNN + LSTM (sans attention) — branche `aziz`

- `cnn_lstm.py` : encodeur CNN (`scratch` = entraîné de zéro, ou `resnet` = features ResNet-50 pré-calculées) + décodeur LSTM (Show and Tell), greedy et beam search.
- `train_cnn_lstm.py` : entraînement, validation à chaque epoch (loss, perplexité, BLEU, CIDEr), early stopping sur le CIDEr val, évaluation test avec `evaluate.py`.

```bash
python train_cnn_lstm.py --encoder scratch --img_dir data/Flicker8k_Dataset --name A_cnn_scratch_lstm
python train_cnn_lstm.py --encoder resnet --name B_lstm_resnet
```
Résultats : `results/cnn_lstm/<run>/` (`curves.png`, `history.csv`, `test_metrics.json`, `examples.png`).

# image-captioning
Image captioning on Flickr8k: comparing a CNN+LSTM baseline, an attention-based LSTM (Show, Attend and Tell) and a CLIP + GPT-2 prefix model (ClipCap-style). Computer vision project, Bordeaux INP – ENSEIRB-MATMECA, 2026–27.

The goal: give the model a photo, it writes a sentence describing it.

## Files

### Shared (everyone uses these)
| File | What it does |
|---|---|
| `common.py` | Small tools used by every script: the word list (vocabulary), how sentences are cut into words, fixed random seed. |
| `prepare_data.py` | Reads the Flickr8k captions, splits images into train / val / test, builds the vocabulary. Writes to `data/`. |
| `extract_features.py` | Runs each image once through ResNet-50 and CLIP and saves the result in `data/features/`. Models then train on these saved features instead of raw images (much faster). |
| `evaluate.py` | Gives scores to generated captions by comparing them with the human ones (BLEU, METEOR, ROUGE-L, CIDEr, CLIPScore). Higher is better. |
| `clip_rerank.py` | Takes several candidate captions per image and keeps the one CLIP finds closest to the image. |
| `runlog.py` | Saves a small record of every training run (settings, time, scores) in `results/runs/`. `python runlog.py` shows all runs in a table. |

### D · Attention LSTM
| File | What it does |
|---|---|
| `attention_lstm.py` | The attention model. The image is cut into a 7×7 grid; for each word it writes, the model chooses which squares to look at. Trains, picks the best epoch on val, then writes and scores captions for the test images. `--no_attention` gives the same model without attention, to compare. |
| `attention_maps.py` | Draws, for each word of a caption, where the model was looking in the image. Images go to `results/attention_maps/`. |
| `run_D.sh` | Runs all of part D at once: with/without attention, a few settings to tune, then the attention maps. |

## How to run

**1. Install**
```bash
pip install -r requirements.txt matplotlib
```

**2. Get the data** (put them at the root of the repo)
- the Flickr8k images in a folder `Images/`
- `dataset_flickr8k.json` (Karpathy splits)

**3. Prepare the data** (once, for everyone)
```bash
python prepare_data.py --karpathy_json dataset_flickr8k.json
python extract_features.py --img_dir Images
```

**4. Train + test + evaluate** (example with part D)
```bash
python attention_lstm.py --epochs 10                  # with attention
python attention_lstm.py --epochs 10 --no_attention   # without attention
```
This trains, then writes the test captions to `results/preds_<name>_test.json`, prints their scores and saves the run.

<!-- To score any captions file yourself:
```bash
python evaluate.py --preds results/preds_D_attn_test.json
``` -->

**5. See the results**
```bash
python runlog.py                                   # table of all runs
python attention_maps.py --n 10                    # attention pictures (part D)
```

Or everything of part D in one go: `bash run_D.sh`

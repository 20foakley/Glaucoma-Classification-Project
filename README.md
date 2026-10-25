## RNFLT CNN & B-SCAN CNN

### Repository layout (CNN branch)

```
├── data/                       # NOT in git
│   ├── raw/harvard_gf/         # raw .npz download (Training/, Validation/, Test/)
│   └── processed/
│       ├── rnflt_raw.npy       (3300, 200, 200) float32, raw values
│       ├── rnflt_norm.npy      (3300, 200, 200) float32, cleaned + normalized
│       ├── rnflt_mask.npy      (3300, 200, 200) float32, 1 = valid pixel
│       ├── meta.csv            one row per patient (split, dev_split, label, race, ...)
│       └── norm_stats.json     mean, std, clip, and the split they were fit on
├── scripts/
│   ├── extract.py
│   ├── verify.py
│   ├── split.py
│   ├── preprocess.py
│   ├── dataset.py
│   └── train.py
├── evaluation/
│   ├── evaluate.py
│   └── make_figures.py
├── results/                    # run outputs (weights are git-ignored)
│   ├── runs/  runs_a/  runs_b/  runs_c/   # config comparison runs
│   ├── runs_final/                        # 5-seed dev ensemble
│   └── figs/                              # evaluation figures
├── requirements.txt
└── README.md
```

Rows are in the same order in every file in `data/processed/`. Run all commands from the
repo root.

### Data pipeline

| Step | Command | Input | Output |
|---|---|---|---|
| 1 | `python scripts/extract.py` | raw `.npz` files | `rnflt_raw.npy`, `meta.csv` |
| 2 | `python scripts/verify.py` | raw files, step 1 output | PASS/FAIL report |
| 3 | `python scripts/split.py` | `meta.csv` | `meta.csv` with `dev_split` column |
| 4 | `python scripts/preprocess.py` | `rnflt_raw.npy`, `meta.csv` | `rnflt_norm.npy`, `rnflt_mask.npy`, `norm_stats.json` |

**extract.py** reads all 3,300 patient files and keeps only the 2D data, skipping the large
3D `oct_bscans` volume. It saves the RNFLT maps as one array and a metadata table with one
row per patient: filename, official split, glaucoma label, race, sex, age, and md. Race is
stored as a string (Asian, Black or African American, White or Caucasian) and is used only
for per-group fairness metrics. md is for analysis only. Neither is a model input.

**verify.py** checks the extraction against the raw files:
- 3,300 files, with array and table lengths that match, and row order matching the raw files
- no NaN, inf, or null values
- split sizes of 2,100 / 300 / 900, binary labels, race balanced within each split
- RNFLT negative values are only the codes -1 and -2
- every RNFLT map and metadata row is identical to a fresh re-read of its raw file

It prints PASS or FAIL for each check. Do not continue if any check fails.

**split.py** divides only the official training rows (2,100) into 80% `dev_train` (1,680)
and 20% `dev_val` (420), stratified by glaucoma label and race, seed 42. The result is a new
`dev_split` column in `meta.csv`. The official val (300) and test (900) rows are unchanged.

| `dev_split` | Rows | Use |
|---|---|---|
| `dev_train` | 1,680 | fit the model during development |
| `dev_val` | 420 | tuning and model selection during development |
| `val` | 300 | official val, spare sanity check |
| `test` | 900 | official test, used once at the end |

**preprocess.py** cleans and normalizes the maps:
- Pixels with values -2, -1, or 0 are set to 0 and recorded in a validity mask. These are
  placeholder codes, the optic disc, or areas outside the scan.
- Values are clipped at 400 µm to remove segmentation artifacts (raw max is 1,032 µm).
- Mean and std are computed on the valid pixels of one fit split only, then applied to all
  rows. Invalid pixels stay 0. The fit split is set with `--fit_col` and `--fit_split`
  (default: official `train`). The development runs below used `dev_train`
  (mean 73.04, std 40.23). The fit split is recorded in `norm_stats.json`.

### Model code

**dataset.py** defines `RNFLTDataset`, a PyTorch Dataset. It selects rows by a split column
and value, and returns the normalized map (shape `(1, 200, 200)`) and the glaucoma label.
Options: `use_mask` adds the validity mask as a second channel, and `aug` applies a random
shift of up to 10 pixels (training only). Race is kept on the object for per-group metrics
and is never returned as a model input.

**train.py** trains a ResNet (18 by default) with a 1-channel input and a single-logit
output, using BCE loss, AdamW, and a cosine learning-rate schedule. After each epoch it
evaluates on the eval split and prints AUROC and ECE. Outputs per seed go to the `--out`
folder: weights (`.pt`), predicted probabilities (`_probs.npy`, used for the ensemble), and
a metrics file (`.json`).

| Flag | Meaning |
|---|---|
| `--epochs`, `--lr`, `--bs`, `--wd` | training settings (defaults 30, 3e-4, 32, 1e-4) |
| `--arch` | torchvision ResNet name (default `resnet18`) |
| `--aug` | random-shift augmentation on the training set |
| `--pretrained` | start from ImageNet weights |
| `--use_mask` | add the validity mask as a second input channel |
| `--save_last` | keep the final-epoch weights and probabilities instead of the best epoch |
| `--split_col`, `--train_split`, `--eval_split` | which rows to train and evaluate on (default `dev_split`, `dev_train`, `dev_val`) |
| `--seed`, `--out` | seed and output folder |
| `--final` | required to evaluate on the `test` split |

### Evaluation code

**evaluate.py** loads the saved `_probs.npy` files from a run folder and reports, per seed
(mean ± std) and for the 5-seed ensemble: AUROC, ECE, and AUROC per race, with 95%
bootstrap confidence intervals. It also reports two uncertainty measures, predictive
entropy and disagreement across seeds, how well each flags wrong predictions, and the error
rate by uncertainty quartile.

**make_figures.py** draws six panels from the same predictions, saved as one dashboard and
as individual PNGs in `results/figs/`: ROC curves (overall and per race), calibration
diagram, AUROC by race with confidence intervals, per-seed vs. ensemble AUROC, error rate
by uncertainty quartile, and a probability histogram by true class.

Both take `--runs`, `--split_col`, and `--eval_split`.

### Development results (5 seeds, scored on dev_val)

Models were trained on `dev_train` and scored on `dev_val` (n=420). Configuration choices
were made on this set, so these numbers are slightly optimistic. They check the pipeline and
are not the reported result.

| Config | Best-epoch AUROC | ECE |
|---|---|---|
| a: ResNet-18 from scratch, aug, 15 epochs (3 seeds) | 0.858 ± 0.003 | 0.086 |
| b: pretrained, aug, 6 epochs, lr 1e-4 (3 seeds) | 0.849 ± 0.007 | 0.144 |
| c: pretrained, aug, mask channel, 15 epochs (1 seed) | 0.846 | 0.096 |

**Frozen configuration:** ResNet-18 from scratch, RNFLT only (no mask channel), `--aug`,
15 epochs, lr 3e-4, batch size 32, final-epoch weights (`--save_last`).

| Final-epoch results, 5 seeds on dev_val | Single seed (mean ± std) | Ensemble |
|---|---|---|
| AUROC | 0.846 ± 0.004 | 0.853 (95% CI 0.810 to 0.889) |
| ECE | 0.077 ± 0.013 | 0.056 |
| AUROC, Asian | 0.873 | 0.882 |
| AUROC, Black | 0.840 | 0.848 |
| AUROC, White | 0.826 | 0.828 |

- Race confidence intervals overlap heavily (about 140 patients per race), so the gap is
  not established on this data.
- Error rate rises from 3.8% to 42.9% from the most to the least confident quartile. Seed
  disagreement detects errors with AUROC 0.74.
- Pretrained models overfit quickly and were worse calibrated. The mask channel did not
  help, since invalid pixels are already 0 after normalization.

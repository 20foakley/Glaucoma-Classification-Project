## RNFLT CNN & B-SCAN CNN

CNN baseline for glaucoma vs. no glaucoma on the 2D RNFLT map (200×200 thickness
image) from Harvard-GF. Final result: 5-seed ResNet-18 ensemble, test AUROC 0.846
(95% CI 0.820 to 0.872), ECE 0.072.

### Repository layout (CNN branch)

```
├── data/                       # NOT in git
│   ├── raw/harvard_gf/         # raw .npz download (Training/, Validation/, Test/)
│   └── processed/
│       ├── rnflt_raw.npy      (3300, 200, 200) float32, raw values
│       ├── rnflt_norm.npy     (3300, 200, 200) float32, cleaned + normalized
│       ├── rnflt_mask.npy     (3300, 200, 200) float32, 1 = valid pixel
│       ├── meta.csv           one row per patient (split, dev_split, label, race, ...)
│       └── norm_stats.json    mean, std, clip, and the split they were fit on
├── scripts/
│   ├── extract.py
│   ├── verify.py
│   ├── split.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── train.py
│   └── predict.py
├── evaluation/
│   ├── evaluate.py
│   └── make_figures.py
├── results/                    # run outputs (weights are git-ignored)
│   ├── runs/  runs_a/  runs_b/  runs_c/   # config comparison runs (development)
│   ├── runs_final/                        # 5-seed dev ensemble (trained on dev_train)
│   ├── runs_full/                         # 5-seed final ensemble (trained on official train)
│   ├── runs_full_val/                     # runs_full re-scored on val (reload check)
│   ├── runs_full_test/                    # runs_full scored on test
│   └── figs/                              # evaluation figures
├── requirements.txt
└── README.md
```

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
| `val` | 300 | official val, sanity check for the final models |
| `test` | 900 | official test, scored once |

**preprocess.py** cleans and normalizes the maps:
- Pixels with values -2, -1, or 0 are set to 0 and recorded in a validity mask. These are
  placeholder codes, the optic disc, or areas outside the scan.
- Values are clipped at 400 µm to remove segmentation artifacts (raw max is 1,032 µm).
- Mean and std are computed on the valid pixels of one fit split only, then applied to all
  rows. Invalid pixels stay 0. The fit split is set with `--fit_col` and `--fit_split`
  (default: official `train`) and is recorded in `norm_stats.json`.
- Development runs used `dev_train` stats (mean 73.04, std 40.23). Final runs used the
  official train stats (mean 73.10, std 40.09). Re-run `preprocess.py` with the matching
  fit split before training or scoring a model.

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

**predict.py** loads the saved `.pt` weights from a run folder and scores one split, with no
training. It saves per-seed probabilities in the same format `evaluate.py` and
`make_figures.py` read, to `<runs>_<split>` by default. Scoring the `test` split requires
`--final`. Use it to score the test set once, and to check that reloaded weights reproduce
the training-time results.

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

### Experiments

**Development phase.** Models were trained on `dev_train` and scored on `dev_val` (n=420).
Configuration choices were made on this set, so its numbers are slightly optimistic.

| Config | Best-epoch AUROC | ECE |
|---|---|---|
| a: ResNet-18 from scratch, aug, 15 epochs (3 seeds) | 0.858 ± 0.003 | 0.086 |
| b: pretrained, aug, 6 epochs, lr 1e-4 (3 seeds) | 0.849 ± 0.007 | 0.144 |
| c: pretrained, aug, mask channel, 15 epochs (1 seed) | 0.846 | 0.096 |

Pretrained models overfit quickly and were worse calibrated. The mask channel did not help,
since invalid pixels are already 0 after normalization.

**Frozen configuration:** ResNet-18 from scratch, RNFLT only (no mask channel), `--aug`,
15 epochs, lr 3e-4, batch size 32, final-epoch weights (`--save_last`), 5 seeds.

**Final runs.** The frozen configuration was retrained on the official train set (2,100)
with 5 seeds, then checked on the official val (300) and scored once on the test set (900).

```
python scripts/preprocess.py
python scripts/train.py --epochs 15 --aug --save_last --seed 0 \
    --split_col split --train_split train --eval_split val --out results/runs_full
    # repeat for seeds 1 to 4
python scripts/predict.py --runs results/runs_full --split_col split --eval_split val
python evaluation/evaluate.py --runs results/runs_full_val --split_col split --eval_split val
python scripts/predict.py --runs results/runs_full --eval_split test --final
python evaluation/evaluate.py --runs results/runs_full_test --split_col split --eval_split test
python evaluation/make_figures.py --runs results/runs_full_test --split_col split --eval_split test
```

### Results

Ensemble of 5 seeds (trained on the official train set). The test row is the reported
result. Val was a sanity check and was not used for any tuning.

| Split | AUROC (ensemble) | 95% CI | ECE (ensemble) | AUROC single seed (mean ± std) | ECE single seed |
|---|---|---|---|---|---|
| val (n=300) | 0.832 | 0.780 to 0.874 | 0.077 | 0.826 ± 0.003 | 0.097 ± 0.008 |
| **test (n=900)** | **0.846** | **0.820 to 0.872** | **0.072** | 0.840 ± 0.004 | 0.082 ± 0.013 |

**AUROC by race, test set**

| Race | Ensemble | 95% CI | Single seed (mean ± std) |
|---|---|---|---|
| Asian | 0.873 | 0.830 to 0.912 | 0.865 ± 0.004 |
| White | 0.851 | 0.806 to 0.891 | 0.844 ± 0.010 |
| Black | 0.798 | 0.748 to 0.851 | 0.792 ± 0.005 |

The worst-to-best gap is 0.075, and Black patients had the lowest AUROC in every seed. The
Asian and Black confidence intervals overlap slightly, and the cause of the gap is not
identified. On the val set, with 100 patients per race, the ordering differed and the gap
was 0.022, so small-sample noise is large there.

**Uncertainty, test set**
- Ensemble accuracy at a 0.5 threshold is 0.777.
- Predictive entropy flags wrong predictions with AUROC 0.717, and seed disagreement with
  0.709. The two correlate at 0.82.
- Error rate by entropy quartile, from most to least confident: 4.0%, 19.1%, 28.0%, 38.2%.

**Development-phase ensemble** (`runs_final`, trained on `dev_train`, scored on `dev_val`):
AUROC 0.853 (95% CI 0.810 to 0.889), ECE 0.056, single-seed AUROC 0.846 ± 0.004. The test
result is slightly lower, as expected for numbers that were not used for any selection.

Figures: `results/figs/evaluation_dashboard_test.png` and the individual panels.

### Rules for this branch
- Never use `md`, `tds`, or `race` as model inputs (label leakage and shortcut).
- Compute normalization statistics from training data only.
- Tune on a validation split. Touch the official test set once, at the end.
- Report AUROC, ECE, and AUROC per race, with 5 seeds.
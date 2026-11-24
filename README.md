## RNFLT CNN & B-scan CNN

Two image baselines for glaucoma vs. no glaucoma on Harvard-GF, compared on the same splits:
a 2D ResNet-18 on the RNFLT thickness map (200×200), and a 3D ResNet-18 on the OCT volume
(downsampled to 64×64×64). Each is a 5-seed ensemble.

| Test set (n=900) | AUROC (ensemble, 95% CI) | ECE (ensemble) |
|---|---|---|
| 2D RNFLT, ResNet-18 | 0.846 (0.820 to 0.872) | 0.072 |
| 3D OCT, r3d_18 | 0.853 (0.829 to 0.877) | 0.039 |

The AUROC difference is not significant (paired bootstrap, p = 0.505). See Comparison.

### Repository layout (CNN branches)

```
├── data/                       # NOT in git
│   ├── raw/harvard_gf/         # raw .npz download (Training/, Validation/, Test/)
│   └── processed/
│       ├── rnflt_raw.npy      (3300, 200, 200) float32, raw values
│       ├── rnflt_norm.npy     (3300, 200, 200) float32, cleaned + normalized
│       ├── rnflt_mask.npy     (3300, 200, 200) float32, 1 = valid pixel
│       ├── norm_stats.json    RNFLT mean, std, clip, and the split they were fit on
│       ├── oct_vol64.npy      (3300, 64, 64, 64) uint8, downsampled OCT volumes
│       ├── oct_files.txt      filenames in the row order of oct_vol64.npy
│       ├── oct_norm_stats.json  OCT mean, std, and the split they were fit on
│       └── meta.csv           one row per patient (split, dev_split, label, race, ...)
├── scripts/
│   ├── extract.py             # 2D: RNFLT maps + metadata
│   ├── verify.py
│   ├── split.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── train.py
│   ├── predict.py
│   ├── extract_oct.py         # 3D: OCT volumes
│   ├── verify_oct.py
│   ├── norm_oct.py
│   ├── dataset3d.py
│   ├── train3d.py
│   └── predict3d.py
├── evaluation/
│   ├── evaluate.py
│   ├── make_figures.py
│   └── compare_models.py
├── results/                    # run outputs (weights are git-ignored)
│   ├── runs/  runs_a/  runs_b/  runs_c/        # 2D config comparison runs (development)
│   ├── runs_final/                             # 2D 5-seed dev ensemble (dev_train)
│   ├── runs_full/                              # 2D 5-seed final ensemble (official train)
│   ├── runs_full_val/  runs_full_test/         # runs_full re-scored on val / test
│   ├── runs3d_smoke/  runs3d_r3d/  runs3d_mc3/  runs3d_r3d_e25/   # 3D config comparison
│   ├── runs3d_dev/                             # 3D 5-seed dev ensemble (dev_train)
│   ├── runs3d_full/                            # 3D 5-seed final ensemble (official train)
│   ├── runs3d_full_val/  runs3d_full_test/     # runs3d_full re-scored on val / test
│   └── figs/  figs3d/                          # 2D / 3D evaluation figures
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
| 5 | `python scripts/extract_oct.py` | raw `.npz` files | `oct_vol64.npy`, `oct_files.txt` |
| 6 | `python scripts/verify_oct.py` | raw files, step 5 output, `meta.csv` | PASS/FAIL report, `oct_check.png` |
| 7 | `python scripts/norm_oct.py` | `oct_vol64.npy`, `meta.csv` | `oct_norm_stats.json` |

Steps 1 to 4 serve the 2D branch and steps 5 to 7 the 3D branch. Step 7 reads the
`dev_split` column by default, so run step 3 first.

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

**preprocess.py** cleans and normalizes the RNFLT maps:
- Pixels with values -2, -1, or 0 are set to 0 and recorded in a validity mask. These are
  placeholder codes, the optic disc, or areas outside the scan.
- Values are clipped at 400 µm to remove segmentation artifacts (raw max is 1,032 µm).
- Mean and std are computed on the valid pixels of one fit split only, then applied to all
  rows. Invalid pixels stay 0. The fit split is set with `--fit_col` and `--fit_split`
  (default: official `train`) and is recorded in `norm_stats.json`.
- Development runs used `dev_train` stats (mean 73.04, std 40.23). Final runs used the
  official train stats (mean 73.10, std 40.09). Re-run `preprocess.py` with the matching
  fit split before training or scoring a model.

**extract_oct.py** reads the `oct_bscans` array (200×200×200, uint8) of every patient and
downsamples it to 64×64×64 by 3D area averaging, keeping the axis order (B-scans along
axis 0, depth along axis 1). It saves one uint8 array `oct_vol64.npy` and `oct_files.txt`,
the filenames in row order. A full 200³ volume is too large to train a 3D ResNet on a
16 GB GPU.

**verify_oct.py** checks the OCT extraction:
- shape `(3300, 64, 64, 64)` and dtype uint8
- file order matches the raw files and `meta.csv`
- no empty volumes
- 100 random volumes identical to a fresh re-read and downsample of the raw file
- saves `oct_check.png`, raw and downsampled B-scans, to confirm the axis order visually

**norm_oct.py** computes one mean and std over all voxels of the fit split's volumes and
saves them to `oct_norm_stats.json`. No normalized copy is stored; `dataset3d.py` rescales
each volume as it is loaded. The fit split is set with `--fit_col` and `--fit_split`
(default `dev_split` / `dev_train`). Development runs used mean 35.22, std 20.41. Final runs
used the official train stats (mean 35.16, std 20.39). Re-run it with the matching fit split
before training or scoring a model.

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

**dataset3d.py** defines `OCTVolDataset`, the 3D counterpart of `RNFLTDataset`. It selects
rows by split, loads the volumes from `oct_vol64.npy`, rescales them with the saved mean and
std, and returns a `(1, 64, 64, 64)` tensor and the label. With `aug=True` it applies a
random 3D shift of up to 4 voxels per axis (training only). Race is kept for per-group
metrics only. It never returns `md`, `tds`, or race as inputs.

**train3d.py** is the 3D version of `train.py`: a torchvision video ResNet (`r3d_18` or
`mc3_18`) with the first layer changed to one input channel and a single-logit output. It
uses the same loss, optimizer, schedule, metrics, and test-set guard as `train.py`, plus
mixed precision. Defaults: 15 epochs, batch size 8, lr 3e-4, weight decay 1e-4, output
folder `results/runs3d`. It takes the same `--aug`, `--pretrained`, `--save_last`,
`--split_col`, `--train_split`, `--eval_split`, `--seed`, `--final`, and `--out` flags, plus
`--arch` (`r3d_18` or `mc3_18`) and `--size`.

**predict3d.py** is the 3D version of `predict.py`: it reloads the saved weights from a run
folder, scores one split, and saves per-seed probabilities. Scoring `test` requires
`--final`.

### Evaluation code

**evaluate.py** loads the saved `_probs.npy` files from a run folder and reports, per seed
(mean ± std) and for the 5-seed ensemble: AUROC, ECE, and AUROC per race, with 95%
bootstrap confidence intervals. It also reports two uncertainty measures, predictive
entropy and disagreement across seeds, how well each flags wrong predictions, and the error
rate by uncertainty quartile. It saves the ensemble probabilities as
`ensemble_<split>_probs.npy` in the run folder.

**make_figures.py** draws six panels from the same predictions, saved as one dashboard and
as individual PNGs: ROC curves (overall and per race), calibration diagram, AUROC by race
with confidence intervals, per-seed vs. ensemble AUROC, error rate by uncertainty quartile,
and a probability histogram by true class.

**compare_models.py** compares models on the same patients. It takes one ensemble
probability file per model (`--models name=path.npy`), reports AUROC and ECE with bootstrap
95% intervals, and runs a paired bootstrap on the AUROC difference between each pair of
models (difference, 95% interval, p-value). It does not test ECE differences.

`evaluate.py` and `make_figures.py` take `--runs`, `--arch` (default `resnet18`; use
`r3d_18` for the 3D runs), `--split_col`, and `--eval_split`. `make_figures.py` also takes
`--out` and `--title`.

### Experiments, 2D RNFLT

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

### Experiments, 3D OCT

**Development phase.** Models were trained on `dev_train` and scored on `dev_val` (n=420),
one seed each, final-epoch scores.

| Config | AUROC | ECE |
|---|---|---|
| r3d_18, 15 epochs | 0.822 | 0.055 |
| mc3_18, 15 epochs | 0.820 | 0.068 |
| r3d_18, 25 epochs | 0.825 | 0.061 |

The three configurations are within 0.005 of each other, which is inside epoch-to-epoch
noise. Longer training and the lighter `mc3_18` did not help.

**Frozen configuration:** `r3d_18` from scratch, OCT volume at 64³, `--aug`, 15 epochs,
lr 3e-4, batch size 8, final-epoch weights (`--save_last`), 5 seeds.

**Development ensemble** (`runs3d_dev`, trained on `dev_train`, scored on `dev_val`):
AUROC 0.825 (95% CI 0.782 to 0.864), ECE 0.053, single-seed AUROC 0.824 ± 0.001. The 2D
development ensemble scored 0.853 on the same patients; this difference was not tested.

**Final runs.** The frozen configuration was retrained on the official train set (2,100)
with 5 seeds, checked on the official val (300), and scored once on the test set (900).

```
python scripts/norm_oct.py --fit_col split --fit_split train
python scripts/train3d.py --epochs 15 --aug --save_last --seed 0 \
    --split_col split --train_split train --eval_split val --out results/runs3d_full
    # repeat for seeds 1 to 4
python scripts/predict3d.py --runs results/runs3d_full --split_col split --eval_split val
python evaluation/evaluate.py --runs results/runs3d_full_val --arch r3d_18 --split_col split --eval_split val
python scripts/predict3d.py --runs results/runs3d_full --eval_split test --final
python evaluation/evaluate.py --runs results/runs3d_full_test --arch r3d_18 --split_col split --eval_split test
python evaluation/make_figures.py --runs results/runs3d_full_test --arch r3d_18 \
    --split_col split --eval_split test --out results/figs3d --title "Glaucoma 3D OCT CNN (r3d_18)"
```

### Results

Ensembles of 5 seeds, trained on the official train set. The test rows are the reported
results. Val was a sanity check and was not used for any tuning.

| Model | Split | AUROC (ensemble) | 95% CI | ECE (ensemble) | AUROC single seed (mean ± std) | ECE single seed |
|---|---|---|---|---|---|---|
| 2D RNFLT | val (n=300) | 0.832 | 0.780 to 0.874 | 0.077 | 0.826 ± 0.003 | 0.097 ± 0.008 |
| 2D RNFLT | **test (n=900)** | **0.846** | **0.820 to 0.872** | **0.072** | 0.840 ± 0.004 | 0.082 ± 0.013 |
| 3D OCT | val (n=300) | 0.863 | 0.821 to 0.901 | 0.056 | 0.861 ± 0.005 | 0.067 ± 0.008 |
| 3D OCT | **test (n=900)** | **0.853** | **0.829 to 0.877** | **0.039** | 0.850 ± 0.005 | 0.034 ± 0.005 |

**AUROC by race, test set** (ensemble, 95% CI)

| Race | 2D RNFLT | 3D OCT |
|---|---|---|
| Asian | 0.873 (0.830 to 0.912) | 0.873 (0.831 to 0.912) |
| White | 0.851 (0.806 to 0.891) | 0.872 (0.832 to 0.912) |
| Black | 0.798 (0.748 to 0.851) | 0.804 (0.756 to 0.852) |
| Worst-to-best gap | 0.075 | 0.069 |

Black patients had the lowest AUROC in both models, and in every seed of the 2D model. Race
is not a model input, and the cause of the gap is not identified. On the val set, with 100
patients per race, the ordering differed for the 2D model and the confidence intervals were
wide, so small-sample noise is large there.

**Uncertainty, test set**

| | 2D RNFLT | 3D OCT |
|---|---|---|
| Ensemble accuracy at a 0.5 threshold | 0.777 | 0.769 |
| Predictive entropy flags errors (AUROC) | 0.717 | 0.744 |
| Seed disagreement flags errors (AUROC) | 0.709 | 0.689 |
| Error rate by entropy quartile, most to least confident | 4.0%, 19.1%, 28.0%, 38.2% | 4.0%, 17.8%, 24.4%, 46.2% |

**Development ensembles** (trained on `dev_train`, scored on `dev_val`, n=420):

| | AUROC (ensemble) | 95% CI | ECE | AUROC single seed |
|---|---|---|---|---|
| 2D RNFLT (`runs_final`) | 0.853 | 0.810 to 0.889 | 0.056 | 0.846 ± 0.004 |
| 3D OCT (`runs3d_dev`) | 0.825 | 0.782 to 0.864 | 0.053 | 0.824 ± 0.001 |

The 2D test result is slightly lower than its development result, as expected for numbers
that were not used for any selection.

Figures: `results/figs/evaluation_dashboard_test.png` (2D) and
`results/figs3d/evaluation_dashboard_test.png` (3D), plus the individual panels.

### Comparison: 2D RNFLT vs. 3D OCT

Paired bootstrap on the AUROC difference (2D minus 3D), 2,000 resamples of the same
patients, with `evaluation/compare_models.py`:

| Split | Difference | 95% CI | p |
|---|---|---|---|
| val (n=300) | -0.031 | -0.070 to 0.010 | 0.121 |
| test (n=900) | -0.007 | -0.027 to 0.014 | 0.505 |

- On AUROC the two representations are statistically indistinguishable on both splits. The
  ordering changes between splits (2D ahead on `dev_val`, 3D ahead on val and test), which
  is what a tie looks like.
- The 3D model has a lower ECE on test (0.039 vs. 0.072), and the bootstrap intervals
  barely overlap (0.033 to 0.073 vs. 0.056 to 0.102). ECE differences were not tested with a
  paired test, so this is suggestive, not established.
- The 3D volume is downsampled to 64³, so a higher resolution may behave differently. This
  was not tested.

```
python evaluation/compare_models.py --split test --models \
    rnflt2d=results/runs_full_test/ensemble_test_probs.npy \
    oct3d=results/runs3d_full_test/ensemble_test_probs.npy
```

### Rules for these branches
- Never use `md`, `tds`, or `race` as model inputs (label leakage and shortcut).
- Compute normalization statistics from training data only.
- Tune on a validation split. Touch the official test set once, at the end.
- Report AUROC, ECE, and AUROC per race, with 5 seeds.
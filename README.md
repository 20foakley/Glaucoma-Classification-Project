# Glaucoma-Classification-Project (GRAFT)

Glaucoma detection on the **Harvard-GF** dataset with **attention fusion**, **uncertainty / calibration**, and **fairness** analysis.

The goal is not only to classify glaucoma vs. no glaucoma, but also to answer:

1. **Trust** – when the model combines several inputs (RNFLT map, OCT, tabular features), which one does it rely on for each patient?
2. **Uncertainty** – how sure is the model, and is that confidence honest (calibration, ECE)?
3. **Fairness** – does it work equally well for Asian, Black, and White patients?

---

## Dataset: Harvard-GF

- 3,300 patients, one eye each, one `.npz` file per patient.
- Official split (from `ReadMe/data_summary.csv`, column `use`): **train 2,100 / val 300 / test 900**.
- Glaucoma rate ≈ 52% / 59% / 54% (train / val / test), so classes are roughly balanced.
- No fundus photos. Each file contains:

| Key | Shape | Meaning |
|---|---|---|
| `oct_bscans` | (200, 200, 200) uint8 | 3D OCT volume. **B-scans = axis 0, depth = axis 1** |
| `rnflt` | (200, 200) float | RNFL thickness map (µm) |
| `md` | scalar | Visual field mean deviation |
| `tds` | (52,) | Visual field total deviation values |
| `glaucoma` | scalar | Label (1 = glaucoma) |
| `race`, `male`, `hispanic`, `language`, `maritalstatus`, `age` | scalar | Demographics |

Download (requires a Hugging Face account and accepting the dataset terms):

```bash
hf auth login
hf download harvardairobotics/Harvard-GF --repo-type dataset --local-dir data/raw/harvard_gf
```

Then unzip `dataset.zip` inside `data/raw/harvard_gf/` so the files sit in `Dataset/Training/`, `Dataset/Validation/`, `Dataset/Test/`.

---

## Rules every model must follow

| Rule | Why |
|---|---|
| **Never use `md` or `tds` as model inputs.** | The glaucoma label was defined from the visual field test, so these leak the answer. Use them only for analysis (e.g. mild / moderate / severe groups). |
| **Never use `race` as a model input.** | It is a shortcut (Black patients have a higher glaucoma rate). Use it only for per-group fairness metrics. |
| **Use the official split. Touch `test` once, at the end.** | Avoid leakage and overfitting to the test set. |
| **Compute normalization stats from training data only.** | Avoid leakage and overfitting to the test set. |
| **Report AUROC, ECE, and AUROC per race.** | So all models are comparable. Use AUROC (not AUPRC) across races, because glaucoma prevalence differs between groups. |
| **Train 5 seeds per model (deep ensemble).** | Ensemble disagreement is our uncertainty estimate. |

---

## Pipeline

Run from the repo root. Scripts that import `src` need `PYTHONPATH=.` (or use `python -m scripts.<name>`).

| Step | Command | Output |
|---|---|---|
| 1. Inspect raw files | `python scripts/inspect_data.py` | Keys, shapes, dtypes, CSV summary |
| 2. Visualize OCT axes | `python scripts/view_oct.py` | `results/oct_axes.png`, `results/oct_mean_projections.png` |
| 3. Preprocess (≈40 s) | `python scripts/preprocess.py` | `data/processed/` arrays + `meta.csv` |
| 4. Check RNFLT values | `python scripts/check_rnflt.py` | Percentiles, `results/rnflt_negative_codes.png` |
| 5. Cup-to-disc ratio | `python scripts/compute_cdr.py` | `structural_features.csv` |
| 6. RNFL features | `python scripts/compute_rnflt_features.py` | `rnflt_features.csv` |
| 7. Scan quality | `python scripts/compute_quality.py` | `quality.csv` |
| 8. Test dataset class | `python -m scripts.test_dataset` | Shape checks |
| 9. Feature baselines | `PYTHONPATH=. python scripts/baseline_features.py` | Val metrics, `results/preds/logreg_{all12,no_age}_seed0_val.csv` |

| 10. Train one model, one seed | `PYTHONPATH=. python scripts/train_fusion.py --fusion single --branches rnflt --seed 0` | `results/preds/single_rnflt_seed0_{val,test}.csv`, `results/checkpoints/single_rnflt_seed0.pt` |
| 11. Late fusion | `python scripts/late_fusion.py --inputs single_rnflt single_tab --name late_rnflt+tab` | `results/preds/late_rnflt+tab_seed{k}_{val,test}.csv` |
| 12. Results table | `PYTHONPATH=. python scripts/summarize.py --ref single_tab --out results/summary_ref_tab.csv` | One row per model: AUROC with 95% CI, ECE, per-race AUROC, paired test vs `--ref` |

Steps 5–7 must run before `GFDataset` can be used.

### Fusion pipeline (steps 10–12)

Every model is trained with `train_fusion.py`, so models with different inputs and fusion heads use the same data, split, recipe, output format and evaluation.

| File | What it does |
|---|---|
| `src/models/branch_encoders.py` | One encoder per input: ResNet-18 from scratch for `rnflt` (1 channel), MLP for `tab` (12 features) |
| `src/models/fusion.py` | `FusionModel`: the encoders plus a fusion head (`single` so far; `gmu` and `concat` are next) |
| `scripts/train_fusion.py` | Trains encoders and head end to end for one seed; writes val/test predictions and a checkpoint |
| `scripts/late_fusion.py` | Averages the probabilities of already trained models, seed by seed |
| `scripts/summarize.py` | Seed mean ± SD, seed-ensemble AUROC with 95% bootstrap CI, ECE, per-race AUROC, paired bootstrap test against `--ref` |

Training recipe (same for every model, from the 2D RNFLT CNN): AdamW, lr 3e-4, weight decay 1e-4, cosine schedule stepped per epoch, batch 32, 15 epochs, **last epoch kept** (val is only monitored, it never picks the model), random shifts of up to 8 px for images (no flips or rotations). `--early_stop` keeps the best-val epoch instead.

Run 5 seeds of each model, then late fusion and the summary:

```bash
for s in 0 1 2 3 4; do
  PYTHONPATH=. python scripts/train_fusion.py --fusion single --branches rnflt --seed $s
  PYTHONPATH=. python scripts/train_fusion.py --fusion single --branches tab   --seed $s
done
python scripts/late_fusion.py --inputs single_rnflt single_tab --name late_rnflt+tab
PYTHONPATH=. python scripts/summarize.py --ref single_tab   --out results/summary_ref_tab.csv
PYTHONPATH=. python scripts/summarize.py --ref single_rnflt --out results/summary_ref_rnflt.csv
```

### Processed files (`data/processed/`)

| File | Shape | Content |
|---|---|---|
| `rnflt.npy` | (3300, 200, 200) float32 | RNFLT maps (raw values, cleaned at load time) |
| `oct_slices.npy` | (3300, 16, 200, 200) uint8 | 16 evenly spaced B-scans per patient |
| `oct_enface.npy` | (3300, 200, 200) float32 | En-face image (OCT averaged over depth) |
| `tds.npy` | (3300, 52) float32 | Visual field values (**analysis only**) |
| `meta.csv` | 3300 rows | Demographics, `md`, `label`, `split` (`train` / `val` / `test`) |
| `structural_features.csv` | 3300 rows | `cup_px`, `disc_px`, `cdr_area`, `vcdr` |
| `rnflt_features.csv` | 3300 rows | `hcdr`, `rim_px`, RNFL mean/p10, ring thickness (mean, sup, inf, horiz), `si_asym` |
| `quality.csv` | 3300 rows | `zero_frac`: share of missing RNFLT pixels (**analysis only**) |
| `rnflt_stats.json`, `tab_stats.json` | — | Normalization stats from train, created on first `GFDataset` use. Delete them if features change. |

RNFLT pixel values: `> 0` thickness (µm), `0` missing, `-1` disc rim, `-2` cup.

Rows are in the same order in every file.

## Using `GFDataset`

```python
from src.data.dataset import GFDataset, train_augment

train_ds = GFDataset("train", modalities=("rnflt", "tab"),
                     mask_channels="rim_cup", transform=train_augment)
val_ds   = GFDataset("val",   modalities=("rnflt", "tab"), mask_channels="rim_cup")
```

| Option | Values |
|---|---|
| `modalities` | any of `rnflt` (C×200×200), `oct` (16×200×200), `enface` (1×200×200), `tab` (12) |
| `mask_channels` | `none` (C=1), `rim_cup` (C=3), `rim_cup_valid` (C=4) |
| `transform` | `train_augment` for the train split only |

Each item also returns `label`, `race`, `md`, `zero_frac`, `filename`; `race` and `md` are for analysis, never model input.

## Results (val, n = 300)

| Model | AUROC | ECE | AUROC Asian / Black / White |
|---|---|---|---|
| Rule: vCDR | 0.685 | – | – |
| Rule: RNFL ring mean | 0.731 | – | – |
| Rule: RNFL inferior ring | 0.765 | – | – |
| Logistic regression (12 features) | 0.781 | 0.087 | 0.809 / 0.776 / 0.790 |
| Logistic regression (no age) | 0.784 | 0.098 | 0.790 / 0.773 / 0.801 |

Val AUROC is noisy (±≈0.05).

## Results (test, n = 900)

From `results/summary_ref_tab.csv` and `results/summary_ref_rnflt.csv`. Deep models: 5 seeds each. "Ensemble" averages the 5 seeds' probabilities; CIs are 95% percentile bootstrap (2000 resamples).

| Model | Ensemble AUROC [95% CI] | ECE | Test AUROC, seed mean ± SD | Val AUROC, seed mean |
|---|---|---|---|---|
| Logistic regression (12 features) | 0.821 [0.794, 0.848] | 0.063 | – | 0.781 |
| Logistic regression (no age) | 0.808 [0.779, 0.836] | 0.070 | – | 0.784 |
| `single_rnflt` (ResNet-18 on RNFLT) | 0.842 [0.815, 0.867] | 0.069 | 0.835 ± 0.005 | 0.826 |
| `single_tab` (MLP on 12 features) | 0.844 [0.818, 0.868] | 0.074 | 0.842 ± 0.003 | 0.808 |
| **`late_rnflt+tab`** (average of the two) | **0.864 [0.840, 0.887]** | **0.042** | 0.861 ± 0.002 | 0.843 |

Paired bootstrap tests (same 900 patients):

| Comparison | AUROC difference [95% CI] | p |
|---|---|---|
| late fusion − `single_tab` | +0.020 [+0.004, +0.036] | 0.016 |
| late fusion − `single_rnflt` | +0.022 [+0.013, +0.032] | < 0.001 |
| `single_tab` − `single_rnflt` | +0.002 [−0.022, +0.028] | 0.87 |
| logistic regression (12) − `single_tab` | −0.023 [−0.037, −0.010] | 0.002 |

AUROC by race (ensemble, test):

| Model | Asian | Black | White |
|---|---|---|---|
| `single_rnflt` | 0.866 | 0.787 | 0.852 |
| `single_tab` | 0.880 | 0.790 | 0.868 |
| `late_rnflt+tab` | 0.892 [0.854, 0.926] | 0.803 [0.753, 0.849] | 0.885 [0.847, 0.918] |

Late fusion beats both single-input models and is the best calibrated, but every model is weakest for Black patients, and fusion does not close that gap.

## Repository layout

```
GRAFT-Glaucoma/
├── data/            # NOT in git: raw download + processed arrays
│   ├── raw/harvard_gf/
│   └── processed/
├── scripts/         # one-off scripts (inspect, preprocess, checks)
├── src/
│   ├── data/dataset.py   # GFDataset: shared PyTorch Dataset
│   ├── eval/metrics.py   # AUROC, AUPRC, ECE, per-race AUROC
│   ├── models/           # CNN, MLP, fusion models
│   └── utils/            # calibration, fairness helpers
├── configs/         # experiment configs (YAML)
├── notebooks/       # exploration (clear outputs before committing)
├── results/         # figures, summary_ref_*.csv; preds/ and checkpoints/ are git-ignored
├── weights/         # NOT in git: pretrained weights (e.g. RETFound)
└── requirements.txt
```

---

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
| **Compute normalization stats from training data only.** | Same reason. |
| **Report AUROC, ECE, and AUROC per race.** | So all models are comparable. Use AUROC (not AUPRC) across races, because glaucoma prevalence differs between groups. |
| **Train 5 seeds per model (deep ensemble).** | Ensemble disagreement is our uncertainty estimate. |

---

## Setup

Python 3.11 (the macOS system Python 3.9 is too old for current PyTorch).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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
| 9. Feature baselines | `PYTHONPATH=. python scripts/baseline_features.py` | Val metrics, `results/preds/logreg_seed0_val.csv` |

Steps 5–7 must run before `GFDataset` can be used.

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

Val AUROC is noisy (±≈0.05); test is evaluated once at the end.

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
├── results/         # figures; preds/ (git-ignored) holds per-patient predictions
├── weights/         # NOT in git: pretrained weights (e.g. RETFound)
└── requirements.txt
```

---

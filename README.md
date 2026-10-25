## RNFLT CNN & B-SCAN CNN

CNN baseline on the 2D RNFLT map (200×200 thickness image) from Harvard-GF.

### Repository layout (CNN branch)

```
├── data/                  # NOT in git
│   ├── raw/harvard_gf/    # raw .npz download (Training/, Validation/, Test/)
│   └── processed/
│       ├── rnflt_raw.npy      (3300, 200, 200) float32, raw values
│       ├── rnflt_norm.npy     (3300, 200, 200) float32, cleaned + normalized
│       ├── rnflt_mask.npy     (3300, 200, 200) float32, 1 = valid pixel
│       ├── meta.csv           one row per patient (split, dev_split, label, race, ...)
│       └── norm_stats.json    mean, std, clip used for normalization
├── scripts/
│   ├── extract.py
│   ├── verify.py
│   ├── split.py
│   ├── preprocess.py
│   ├── dataset.py
│   └── train.py
├── evaluation/            # evaluation code
├── results/               # metrics, probabilities, figures
├── requirements.txt
└── README.md
```

Rows are in the same order in every file in `data/processed/`.

### Pipeline

Run from the repo root, in order. Each script reads the previous script's output.

| Step | Command | Input | Output |
|---|---|---|---|
| 1 | `python scripts/extract.py` | raw `.npz` files | `rnflt_raw.npy`, `meta.csv` |
| 2 | `python scripts/verify.py` | raw files, step 1 output | PASS/FAIL report |
| 3 | `python scripts/split.py` | `meta.csv` | `meta.csv` with `dev_split` column |
| 4 | `python scripts/preprocess.py` | `rnflt_raw.npy`, `meta.csv` | `rnflt_norm.npy`, `rnflt_mask.npy`, `norm_stats.json` |
| 5 | `python scripts/train.py` | step 4 output | weights, probabilities, metrics in `results/` |

### extract.py
Reads all 3,300 patient `.npz` files and keeps only the 2D data, skipping the large
3D `oct_bscans` volume. Saves the RNFLT maps as one array and a metadata table with
one row per patient: filename, official split (train/val/test), glaucoma label, race,
sex, age, and md. Race is stored as a string (Asian, Black or African American, White
or Caucasian) and is used only for per-group fairness metrics. md is for analysis
only. Neither is a model input.

### verify.py
Checks that the extraction is complete and correct:
- 3,300 files, with array and table lengths that match
- row order matches the raw files
- no NaN, inf, or null values
- split sizes of 2,100 / 300 / 900, binary labels, race balanced within each split
- RNFLT negative values are only the codes -1 and -2
- every RNFLT map and metadata row is identical to a fresh re-read of its raw file

It prints PASS or FAIL for each check. Do not continue if any check fails.

### split.py
Divides only the official training rows (2,100) into 80% `dev_train` (1,680) and 20%
`dev_val` (420), stratified by glaucoma label and race, seed 42. The result is written
to a new `dev_split` column in `meta.csv`. The official val (300) and test (900) rows
are unchanged.

| `dev_split` | Rows | Use |
|---|---|---|
| `dev_train` | 1,680 | fit the model |
| `dev_val` | 420 | tuning and early stopping during development |
| `val` | 300 | official val, spare sanity check |
| `test` | 900 | official test, used once at the end |

### preprocess.py
Cleans and normalizes the RNFLT maps:
- Pixels with values -2, -1, or 0 are set to 0 and recorded in a validity mask. These
  are placeholder codes, the optic disc, or areas outside the scan.
- Values are clipped at 400 µm to remove segmentation artifacts (raw max is 1,032 µm).
- Mean and std are computed on `dev_train` valid pixels only, then applied to all
  rows. Invalid pixels stay 0.

The stats are saved in `norm_stats.json`.

### dataset.py
Defines `RNFLTDataset`, a PyTorch Dataset. It selects rows by a split column and
value, and returns the normalized map (shape `(1, 200, 200)`) and the glaucoma label.
Options: `use_mask` adds the validity mask as a second channel, and `aug` applies a
random shift of up to 10 pixels (training only). Race is kept on the object for
per-group metrics and is never returned as a model input.

### train.py
Trains a ResNet (18 by default) with a 1-channel input and a single-logit output, using
BCE loss, AdamW, and a cosine learning-rate schedule. After each epoch it evaluates on
the eval split and keeps the best epoch by AUROC. It reports AUROC, ECE, and AUROC per
race, and saves the weights, predicted probabilities (for the 5-seed ensemble), and a
metrics JSON. Evaluating on `test` requires the `--final` flag.

```
python scripts/train.py --epochs 30 --aug --seed 0
```

Other options: `--arch resnet34`, `--pretrained`, `--use_mask`, `--lr`, `--bs`,
`--out`.

### Rules for this branch
- Never use `md`, `tds`, or `race` as model inputs (label leakage and shortcut).
- Compute normalization stats from training data only.
- Tune on `dev_val`. Touch the official test set once, at the end.
- Report AUROC, ECE, and AUROC per race, with 5 seeds.
import glob, os
import numpy as np
import pandas as pd
from tqdm import tqdm

RAW = "data/raw/harvard_gf"
OUT = "data/processed"
AXIS = 0
N_SLICES = 16
os.makedirs(OUT, exist_ok=True)

summary = pd.read_csv(f"{RAW}/ReadMe/data_summary.csv")
paths = {os.path.basename(p): p for p in glob.glob(f"{RAW}/**/*.npz", recursive=True)}
assert len(paths) == len(summary) == 3300

N = len(summary)
rnflt = np.zeros((N, 200, 200), dtype=np.float32)
oct_slices = np.zeros((N, N_SLICES, 200, 200), dtype=np.uint8)
tds = np.zeros((N, 52), dtype=np.float32)
enface = np.zeros((N, 200, 200), dtype=np.float32)
extra = []

idx = np.linspace(0, 199, N_SLICES).astype(int)

for i, fn in enumerate(tqdm(summary["filename"])):
    d = np.load(paths[fn], allow_pickle=True)
    vol = np.moveaxis(d["oct_bscans"], AXIS, 0)
    oct_slices[i] = vol[idx]
    rnflt[i] = d["rnflt"]
    tds[i] = d["tds"]
    enface[i] = d["oct_bscans"].mean(axis=1) 
    extra.append({
        "md": float(d["md"]),
        "glaucoma_npz": int(d["glaucoma"]),
        "male": int(d["male"]),
        "hispanic": int(d["hispanic"]),
        "language_code": int(d["language"]),
        "marital_code": int(d["maritalstatus"]),
    })

meta = pd.concat([summary, pd.DataFrame(extra)], axis=1)
meta["label"] = (meta["glaucoma"] == "yes").astype(int)
meta["split"] = meta["use"].map({"training": "train", "validation": "val", "test": "test"})

# sanity checks
assert (meta["label"] == meta["glaucoma_npz"]).all(), "CSV and npz labels disagree"
assert meta["split"].notna().all(), f"unexpected 'use' values: {meta['use'].unique()}"

np.save(f"{OUT}/rnflt.npy", rnflt)
np.save(f"{OUT}/oct_slices.npy", oct_slices)
np.save(f"{OUT}/tds.npy", tds)
np.save(f"{OUT}/oct_enface.npy", enface)  
meta.to_csv(f"{OUT}/meta.csv", index=False)

print(meta["split"].value_counts())
print(meta.groupby("split")["label"].mean().round(3))
print(meta.groupby("race")["label"].mean().round(3))
print("RNFLT range:", rnflt.min(), rnflt.max())

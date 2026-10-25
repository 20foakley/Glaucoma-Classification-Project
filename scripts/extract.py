import os, glob, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

RAW = "Harvard-GF"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)
split_map = {"training": "train", "validation": "val", "test": "test"}

def load(f):
    with np.load(f, allow_pickle=True) as d:   # lazy: oct_bscans never loaded
        return dict(
            file=os.path.basename(f),
            split=split_map[os.path.basename(os.path.dirname(f)).lower()],
            rnflt=d["rnflt"].astype(np.float32),
            glaucoma=int(d["glaucoma"]),
            race=str(d["race"]),       # fairness metrics only, never a model input
            male=int(d["male"]),
            age=float(d["age"]),
            md=float(d["md"]),         # analysis only, never a model input
        )

files = sorted(glob.glob(f"{RAW}/**/data_*.npz", recursive=True))
with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(load, files))

rnflt = np.stack([r.pop("rnflt") for r in rows])
meta = pd.DataFrame(rows)
np.save(f"{OUT}/rnflt_raw.npy", rnflt)
meta.to_csv(f"{OUT}/meta.csv", index=False)
print(rnflt.shape, meta.split.value_counts().to_dict())
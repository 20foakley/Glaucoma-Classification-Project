import os, glob, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

RAW, D = "Harvard-GF", "data/processed"
split_map = {"training": "train", "validation": "val", "test": "test"}
rnflt = np.load(f"{D}/rnflt_raw.npy")
meta = pd.read_csv(f"{D}/meta.csv")
files = sorted(glob.glob(f"{RAW}/**/data_*.npz", recursive=True))

def ok(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return cond

results = []

# --- structure ---
results.append(ok("file count", len(files) == 3300 == len(meta) == len(rnflt),
                  f"(files={len(files)}, meta={len(meta)}, rnflt={len(rnflt)})"))
results.append(ok("rnflt shape/dtype", rnflt.shape == (3300, 200, 200) and rnflt.dtype == np.float32, str(rnflt.shape)))
results.append(ok("row order matches raw files", list(meta.file) == [os.path.basename(f) for f in files]))
results.append(ok("unique filenames", meta.file.is_unique))
results.append(ok("no NaN/inf in rnflt", bool(np.isfinite(rnflt).all())))
results.append(ok("no nulls in meta", int(meta.isnull().sum().sum()) == 0))

# --- splits, labels, race ---
results.append(ok("split sizes", meta.split.value_counts().to_dict() == {"train": 2100, "test": 900, "val": 300},
                  str(meta.split.value_counts().to_dict())))
results.append(ok("labels binary", set(meta.glaucoma.unique()) == {0, 1}))
results.append(ok("race values", set(meta.race.unique()) ==
                  {"Asian", "Black or African American", "White or Caucasian"}))
bal = meta.groupby(["split", "race"]).size().unstack()
results.append(ok("race balanced per split", bool((bal.nunique(axis=1) == 1).all())))
print(bal.to_string())
print(meta.groupby("split").glaucoma.mean().round(3).to_string())

# --- value sanity ---
vals = set(np.unique(rnflt[rnflt < 0]).tolist())
results.append(ok("negative codes only -1/-2", vals <= {-1.0, -2.0}, str(vals)))
results.append(ok("no all-invalid maps", bool(((rnflt > 0).any(axis=(1, 2))).all())))
print("age range:", meta.age.min(), meta.age.max())

# --- compare against raw files: every sample, independently re-read ---
def reread(i):
    with np.load(files[i], allow_pickle=True) as d:
        same_map = np.array_equal(d["rnflt"].astype(np.float32), rnflt[i])
        row = meta.iloc[i]
        same_meta = (int(d["glaucoma"]) == row.glaucoma and str(d["race"]) == row.race
                     and int(d["male"]) == row.male and abs(float(d["age"]) - row.age) < 1e-6
                     and split_map[os.path.basename(os.path.dirname(files[i])).lower()] == row.split)
        return same_map, same_meta

with ThreadPoolExecutor(8) as ex:
    out = list(ex.map(reread, range(len(files))))
bad_map = [i for i, (m, _) in enumerate(out) if not m]
bad_meta = [i for i, (_, m) in enumerate(out) if not m]
results.append(ok("rnflt identical to raw (all 3300)", not bad_map, f"mismatches={len(bad_map)} {bad_map[:5]}"))
results.append(ok("meta identical to raw (all 3300)", not bad_meta, f"mismatches={len(bad_meta)} {bad_meta[:5]}"))

print("\nALL CHECKS PASSED" if all(results) else "\nSOME CHECKS FAILED, do not continue")
import json, numpy as np, pandas as pd

D = "data/processed"
CLIP = 400.0

rnflt = np.load(f"{D}/rnflt_raw.npy")
meta = pd.read_csv(f"{D}/meta.csv")
assert "dev_split" in meta.columns, "run split.py first"

# 1. Fixed per-sample cleaning (uses no dataset statistics)
mask = (rnflt > 0).astype(np.float32)                # -2, -1, 0 -> invalid
clean = np.clip(np.where(rnflt > 0, rnflt, 0), 0, CLIP)

# 2. Normalization stats from dev_train valid pixels only
tr = (meta.dev_split == "dev_train").values
vals = clean[tr][mask[tr] > 0]
mu, sd = float(vals.mean()), float(vals.std())

# 3. Apply to every sample; invalid pixels stay 0
norm = np.where(mask > 0, (clean - mu) / sd, 0).astype(np.float32)

np.save(f"{D}/rnflt_norm.npy", norm)
np.save(f"{D}/rnflt_mask.npy", mask)
json.dump(dict(mu=mu, sd=sd, clip=CLIP), open(f"{D}/norm_stats.json", "w"))
print(f"mu={mu:.2f} sd={sd:.2f} norm shape={norm.shape}")
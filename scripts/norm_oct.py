import argparse, json, numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--fit_col", default="dev_split")
p.add_argument("--fit_split", default="dev_train")
p.add_argument("--size", type=int, default=64)
a = p.parse_args()

D = "data/processed"
meta = pd.read_csv(f"{D}/meta.csv")
vol = np.load(f"{D}/oct_vol{a.size}.npy", mmap_mode="r")
assert len(meta) == len(vol), "meta rows and volumes differ"

idx = np.where(meta[a.fit_col] == a.fit_split)[0]
s = s2 = 0.0; n = 0
for i in idx:
    v = vol[i].astype(np.float64)
    s += v.sum(); s2 += (v * v).sum(); n += v.size
mu = s / n; sd = float(np.sqrt(s2 / n - mu ** 2))

json.dump(dict(mu=mu, sd=sd, size=a.size, fit=f"{a.fit_col}=={a.fit_split}"),
          open(f"{D}/oct_norm_stats.json", "w"))
print(f"fit on {a.fit_col}=={a.fit_split} ({len(idx)} patients): mu={mu:.2f} sd={sd:.2f}")

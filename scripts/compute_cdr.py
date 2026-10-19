import numpy as np
import pandas as pd

r = np.load("data/processed/rnflt.npy", mmap_mode="r")
meta = pd.read_csv("data/processed/meta.csv")

cup_px, disc_px, cup_h, disc_h = [], [], [], []

for i in range(len(meta)):
    m = np.array(r[i])
    cup = m == -2
    disc = m < 0
    cup_px.append(cup.sum())
    disc_px.append(disc.sum())
    cup_h.append(cup.any(axis=1).sum())
    disc_h.append(disc.any(axis=1).sum())

f = pd.DataFrame({"filename": meta["filename"],
                  "cup_px": cup_px, "disc_px": disc_px,
                  "cdr_area": np.array(cup_px) / np.maximum(disc_px, 1),
                  "vcdr": np.array(cup_h) / np.maximum(disc_h, 1)})
f.loc[f["disc_px"] == 0, ["cdr_area", "vcdr"]] = np.nan
f.to_csv("data/processed/structural_features.csv", index=False)

meta = meta.merge(f, on="filename")
print("patients with no disc found:", (f["disc_px"] == 0).sum())
print(meta.groupby("label")[["cdr_area", "vcdr", "disc_px"]].mean().round(3))
print("corr(vcdr, md):", round(meta["vcdr"].corr(meta["md"]), 3))

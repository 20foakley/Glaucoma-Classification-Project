import json, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = "data/processed"
CLIP = 400.0
CUP, RIM = -2, -1
TAB_FEATURES = ["vcdr", "cdr_area", "hcdr", "rim_px", "rnfl_mean", "rnfl_p10",
                "ring_mean", "ring_sup", "ring_inf", "ring_horiz", "si_asym", "age"]

def _rnflt_stats(root):
    p = f"{root}/rnflt_stats.json"
    if os.path.exists(p):
        return json.load(open(p))
    meta = pd.read_csv(f"{root}/meta.csv")
    r = np.load(f"{root}/rnflt.npy", mmap_mode="r")
    vals = []
    for i in np.where(meta["split"] == "train")[0]:
        m = np.array(r[i])
        vals.append(np.clip(m[m > 0], 0, CLIP))      # real thickness only, TRAIN only
    vals = np.concatenate(vals)
    s = {"mean": float(vals.mean()), "std": float(vals.std()), "clip": CLIP}
    json.dump(s, open(p, "w"))
    return s


def _tab_stats(meta, root):
    p = f"{root}/tab_stats.json"
    if os.path.exists(p):
        return json.load(open(p))
    tr = meta[meta["split"] == "train"][TAB_FEATURES].astype(float)
    med = tr.median()
    filled = tr.fillna(med)
    s = {"median": med.tolist(), "mean": filled.mean().tolist(),
         "std": (filled.std() + 1e-8).tolist()}
    json.dump(s, open(p, "w"))
    return s


class GFDataset(Dataset):
    def __init__(self, split, root=ROOT, modalities=("rnflt",), mask_channels="none"):
        self.modalities, self.mask_channels = modalities, mask_channels
        meta = pd.read_csv(f"{root}/meta.csv")
        for name in ["structural_features", "rnflt_features", "quality"]:
            meta = meta.merge(pd.read_csv(f"{root}/{name}.csv"), on="filename", how="left")
        assert len(meta) == 3300
        self.meta = meta
        self.idx = np.where(meta["split"] == split)[0]

        if "rnflt" in modalities:
            self.rnflt = np.load(f"{root}/rnflt.npy", mmap_mode="r")
            self.rs = _rnflt_stats(root)
        if "oct" in modalities:
            self.oct = np.load(f"{root}/oct_slices.npy", mmap_mode="r")
        if "tab" in modalities:
            ts = _tab_stats(meta, root)
            v = meta[TAB_FEATURES].astype(float).values
            v = np.where(np.isnan(v), np.array(ts["median"]), v)      # fill with train median
            self.tab_matrix = ((v - np.array(ts["mean"])) / np.array(ts["std"])).astype(np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        row = self.meta.iloc[i]
        item = {"label": torch.tensor(float(row["label"])),
                "race": row["race"],              # analysis only
                "md": float(row["md"]),           # analysis only
                "zero_frac": float(row["zero_frac"]),
                "filename": row["filename"], "index": i}

        if "rnflt" in self.modalities:
            m = np.array(self.rnflt[i])
            valid = m > 0                          # not a code, not missing
            x = (np.clip(m, 0, CLIP) - self.rs["mean"]) / self.rs["std"]
            x[~valid] = 0.0
            chans = [x]
            if self.mask_channels == "rim_cup":
                chans += [(m == RIM).astype(np.float32), (m == CUP).astype(np.float32)]
            item["rnflt"] = torch.from_numpy(np.stack(chans).astype(np.float32))

        if "oct" in self.modalities:
            item["oct"] = torch.from_numpy(np.array(self.oct[i]).astype(np.float32) / 255.0)

        if "tab" in self.modalities:
            item["tab"] = torch.from_numpy(self.tab_matrix[i])
        return item

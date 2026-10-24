import json, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = os.path.join(os.path.dirname(__file__), "../../data/processed")
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
        vals.append(np.clip(m[m > 0], 0, CLIP))
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
    def __init__(self, split, root=ROOT, modalities=("rnflt",), mask_channels="none", transform=None):
        self.modalities, self.mask_channels, self.transform = modalities, mask_channels, transform
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
        if "enface" in modalities:
            self.enface = np.load(f"{root}/oct_enface.npy", mmap_mode="r")
        if "tab" in modalities:
            ts = _tab_stats(meta, root)
            v = meta[TAB_FEATURES].astype(float).values
            v = np.where(np.isnan(v), np.array(ts["median"]), v)
            self.tab_matrix = ((v - np.array(ts["mean"])) / np.array(ts["std"])).astype(np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        row = self.meta.iloc[i]
        item = {"label": torch.tensor(float(row["label"])),
                "race": row["race"],
                "md": float(row["md"]),
                "zero_frac": float(row["zero_frac"]),
                "filename": row["filename"], "index": i}

        if "rnflt" in self.modalities:
            m = np.array(self.rnflt[i])
            valid = m > 0
            x = (np.clip(m, 0, CLIP) - self.rs["mean"]) / self.rs["std"]
            x[~valid] = 0.0
            chans = [x]
            if self.mask_channels in ("rim_cup", "rim_cup_valid"):
                chans += [(m == RIM).astype(np.float32), (m == CUP).astype(np.float32)]
            if self.mask_channels == "rim_cup_valid":
                chans.append(valid.astype(np.float32))
            item["rnflt"] = torch.from_numpy(np.stack(chans).astype(np.float32))

        if "oct" in self.modalities:
            item["oct"] = torch.from_numpy(np.array(self.oct[i]).astype(np.float32) / 255.0)

        if "enface" in self.modalities:
            e = np.array(self.enface[i], dtype=np.float32)
            e = (e - e.mean()) / (e.std() + 1e-6)
            item["enface"] = torch.from_numpy(e[None])

        if "tab" in self.modalities:
            item["tab"] = torch.from_numpy(self.tab_matrix[i])
        
        if self.transform is not None:
            item = self.transform(item)

        return item

def train_augment(item, max_shift=10):
    dy, dx = np.random.randint(-max_shift, max_shift + 1, size=2)
    for key in ["rnflt", "oct", "enface"]:
        if key in item:
            item[key] = torch.roll(item[key], shifts=(int(dy), int(dx)), dims=(-2, -1))
    if "rnflt" in item:
        item["rnflt"][0] = item["rnflt"][0] * (1 + 0.05 * np.random.randn())

    return item

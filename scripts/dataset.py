import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import Dataset

D = "data/processed"

class RNFLTDataset(Dataset):
    """RNFLT maps, already cleaned and normalized by preprocess.py.
    Inputs: the map only (optionally + validity mask). Never md, tds, or race.
    race is returned separately for per-group fairness metrics."""

    def __init__(self, split_col, split_name, aug=False, use_mask=False, max_shift=10):
        meta = pd.read_csv(f"{D}/meta.csv")
        idx = np.where(meta[split_col] == split_name)[0]
        self.x = np.load(f"{D}/rnflt_norm.npy", mmap_mode="r")[idx].copy()
        self.m = np.load(f"{D}/rnflt_mask.npy", mmap_mode="r")[idx].copy() if use_mask else None
        self.y = meta.glaucoma.values[idx].astype(np.float32)
        self.race = meta.race.values[idx]
        self.aug, self.max_shift = aug, max_shift

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i]).unsqueeze(0)           # (1, 200, 200)
        if self.m is not None:
            x = torch.cat([x, torch.from_numpy(self.m[i]).unsqueeze(0)], 0)
        if self.aug:                                           # small random translation, zero-filled
            s = self.max_shift
            dx, dy = np.random.randint(-s, s + 1, 2)
            x = F.pad(x, (s, s, s, s))[:, s + dy: s + dy + 200, s + dx: s + dx + 200]
        return x, torch.tensor(self.y[i])
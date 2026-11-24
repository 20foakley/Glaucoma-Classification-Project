import json, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import Dataset

D = "data/processed"

class OCTVolDataset(Dataset):
    """Downsampled OCT volumes (uint8), normalized on the fly with train-only stats.
    Returns (1, S, S, S) float and the glaucoma label. Never md, tds, or race;
    race is kept on the object for per-group metrics only."""

    def __init__(self, split_col, split_name, aug=False, size=64, max_shift=4):
        meta = pd.read_csv(f"{D}/meta.csv")
        idx = np.where(meta[split_col] == split_name)[0]
        self.v = np.load(f"{D}/oct_vol{size}.npy", mmap_mode="r")[idx]
        st = json.load(open(f"{D}/oct_norm_stats.json"))
        assert st["size"] == size, "norm stats were fit at a different size"
        self.mu, self.sd = st["mu"], st["sd"]
        self.y = meta.glaucoma.values[idx].astype(np.float32)
        self.race = meta.race.values[idx]
        self.aug, self.s, self.size = aug, max_shift, size

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = (torch.from_numpy(np.ascontiguousarray(self.v[i])).float() - self.mu) / self.sd
        x = x.unsqueeze(0)
        if self.aug:
            s, S = self.s, self.size
            xp = F.pad(x, (s,) * 6)
            dz, dy, dx = np.random.randint(0, 2 * s + 1, 3)
            x = xp[:, dz:dz + S, dy:dy + S, dx:dx + S]
        return x, torch.tensor(self.y[i])

import glob, os, numpy as np, torch, torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

RAW, OUT, S = "Harvard-GF", "data/processed", 64
os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(f"{RAW}/**/data_*.npz", recursive=True))
assert len(files) == 3300, len(files)

def down(v):                                  
    t = torch.from_numpy(v).float()[None, None]
    t = F.adaptive_avg_pool3d(t, (S, S, S))
    return t[0, 0].round().clamp(0, 255).byte().numpy()

def load(f):
    with np.load(f, allow_pickle=True) as d:
        return down(d["oct_bscans"])

vol = np.lib.format.open_memmap(f"{OUT}/oct_vol{S}.npy", mode="w+",
                                dtype=np.uint8, shape=(len(files), S, S, S))
with ThreadPoolExecutor(4) as ex:
    for i, v in enumerate(ex.map(load, files)):
        vol[i] = v
        if i % 300 == 0: print(i, "/", len(files), flush=True)
vol.flush()
open(f"{OUT}/oct_files.txt", "w").write("\n".join(os.path.basename(f) for f in files))
print("saved", vol.shape, vol.dtype)

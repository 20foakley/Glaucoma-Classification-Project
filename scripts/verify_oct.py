import glob, os, numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW, D, S = "Harvard-GF", "data/processed", 64
vol = np.load(f"{D}/oct_vol{S}.npy", mmap_mode="r")
files = sorted(glob.glob(f"{RAW}/**/data_*.npz", recursive=True))
saved = open(f"{D}/oct_files.txt").read().split("\n")

def ok(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}"); return cond
res = []
res.append(ok("shape/dtype", vol.shape == (3300, S, S, S) and vol.dtype == np.uint8, str(vol.shape)))
res.append(ok("file order matches raw", saved == [os.path.basename(f) for f in files]))
if os.path.exists(f"{D}/meta.csv"):
    meta = pd.read_csv(f"{D}/meta.csv")
    res.append(ok("row order matches meta.csv", list(meta.file) == saved))
else:
    print("[SKIP] meta.csv not found, order check against meta skipped")

means = np.array([float(vol[i].mean()) for i in range(len(vol))])
res.append(ok("no empty volumes", bool((means > 0).all()), f"min mean {means.min():.1f}"))
print(f"volume mean intensity: min {means.min():.1f}  median {np.median(means):.1f}  max {means.max():.1f}")

def down(v):
    t = torch.from_numpy(v).float()[None, None]
    return F.adaptive_avg_pool3d(t, (S, S, S))[0, 0].round().clamp(0, 255).byte().numpy()

rng = np.random.default_rng(0)
idx = rng.choice(len(files), 100, replace=False)
bad = []
for i in idx:
    with np.load(files[i], allow_pickle=True) as d:
        if not np.array_equal(down(d["oct_bscans"]), vol[i]): bad.append(int(i))
res.append(ok("100 random volumes identical to fresh re-read", not bad, f"mismatches={bad[:5]}"))
print("\nALL CHECKS PASSED" if all(res) else "\nSOME CHECKS FAILED, do not continue")

# picture: B-scan along axis 0 (README says depth is axis 1)
fig, ax = plt.subplots(2, 3, figsize=(11, 7))
for j, i in enumerate(idx[:3]):
    with np.load(files[i], allow_pickle=True) as d: raw = d["oct_bscans"]
    ax[0, j].imshow(raw[100], cmap="gray"); ax[0, j].set_title(f"raw B-scan [100], patient {i}")
    ax[1, j].imshow(vol[i][32], cmap="gray"); ax[1, j].set_title(f"downsampled [32]")
    for a in ax[:, j]: a.axis("off")
plt.tight_layout(); plt.savefig("oct_check.png", dpi=110)
print("saved oct_check.png")

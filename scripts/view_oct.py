import glob
import numpy as np
import matplotlib.pyplot as plt

f = sorted(glob.glob("data/raw/harvard_gf/**/*.npz", recursive=True))[0]
d = np.load(f, allow_pickle=True)
vol = d["oct_bscans"]

fig, ax = plt.subplots(1, 4, figsize=(13, 4.5))
ax[0].imshow(d["rnflt"], cmap="jet")
ax[0].set_title("RNFLT map")

for i in range(3):
    ax[i + 1].imshow(np.take(vol, vol.shape[i] // 2, axis=i), cmap="gray")
    ax[i + 1].set_title(f"OCT middle slice, axis {i}")
for a in ax:
    a.axis("off")

plt.tight_layout()
plt.savefig("results/oct_axes.png", dpi=120)
print("saved results/oct_axes.png")

fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))

for i in range(3):
    ax[i].imshow(vol.mean(axis=i), cmap="gray")
    ax[i].set_title(f"OCT mean projection, axis {i}")
    ax[i].axis("off")

plt.tight_layout()
plt.savefig("results/oct_mean_projections.png", dpi=120)
print("saved results/oct_mean_projections.png")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

r = np.load('data/processed/rnflt.npy', mmap_mode='r')
meta = pd.read_csv('data/processed/meta.csv')
f = pd.read_csv('data/processed/rnflt_features.csv')

print("NaN counts: \n", f.isna().sum()[f.isna().sum() > 0])

meta["zero_frac"] = [(np.array(r[i]) == 0).mean() for i in range(len(r))]
print("\nZero fraction by label: \n", meta.groupby("label")["zero_frac"].describe().round(3))
print("\nZero fraction by race: \n", meta.groupby("race")["zero_frac"].mean().round(3))

worst = meta["zero_frac"].nlargest(3).index
fig, ax = plt.subplots(2, 3, figsize=(19, 7))
for c, i in enumerate(worst):
    m = np.array(r[i])
    ax[0, c].imshow(np.clip(m, 0, 400), cmap="gray")
    ax[0, c].set_title(f"patient {i}")
    ax[1, c].imshow(m == 0, cmap="gray")
    ax[1, c].set_title("== 0 (white)")

for a in ax.flat: a.axis("off")
plt.tight_layout()
plt.savefig("results/zero_pixels.png", dpi=110)

import numpy as np
import matplotlib.pyplot as plt

r = np.load("data/processed/rnflt.npy", mmap_mode="r")
s = np.array(r[:300])

print("percentiles:", np.percentile(s, [0, 0.1, 1, 50, 99, 99.9, 100]).round(1))
print("fraction == -2:", (s == -2).mean().round(4))
print("fraction < 0:", (s < 0).mean().round(4))
print("fraction > 400:", (s > 400).mean().round(6))
print("distinct negatives:", np.unique(s[s < 0])[:10])

fig, ax = plt.subplots(3, 3, figsize=(10, 10))
for row, i in enumerate([0, 1, 2]):
    m = np.array(r[i])
    ax[row, 0].imshow(np.clip(m, 0, 400), cmap="jet"); ax[row, 0].set_title(f"patient {i}: RNFLT")
    ax[row, 1].imshow(m == -1, cmap="gray"); ax[row, 1].set_title("== -1")
    ax[row, 2].imshow(m == -2, cmap="gray"); ax[row, 2].set_title("== -2")
for a in ax.flat:
    a.axis("off")
plt.tight_layout(); plt.savefig("results/rnflt_negative_codes.png", dpi=110)

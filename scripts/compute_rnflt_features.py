import numpy as np
import pandas as pd

r = np.load('data/processed/rnflt.npy', mmap_mode='r')
meta = pd.read_csv('data/processed/meta.csv')
yy, xx = np.mgrid[0:200, 0:200]
RING_R, RING_W = 58, 4

rows = []

for i in range(len(meta)):
    m = np.array(r[i])
    disc, cup, valid = m < 0, m == -2, m > 0
    t = np.clip(m, 0, 400)

    cy, cx = yy[disc].mean(), xx[disc].mean()
    dist = np.hypot(yy - cy, xx - cx)
    ang = np.degrees(np.arctan2(cy - yy, xx - cx)) % 360
    ring = valid & (np.abs(dist - RING_R) <= RING_W)

    def sector(lo, hi):
        return t[ring & (ang >= lo) & (ang < hi)].mean()

    sup, inf = sector(45, 135), sector(225, 315)
    horiz = t[ring & ((ang < 45) | (ang >= 315) | ((ang >= 135) & (ang < 225)))].mean()

    rows.append({
        'filename': meta['filename'][i],
        'hcdr': cup.any(axis=0).sum() / disc.any(axis=0).sum(),
        'rim_px': (m == -1).sum(),
        'rnfl_mean': t[valid].mean(),
        'rnfl_p10': np.percentile(t[valid], 10),
        'ring_mean': t[ring].mean(),
        'ring_sup': sup, 'ring_inf': inf, 'ring_horiz': horiz,
        'si_asym': abs(sup - inf),
    })

f = pd.DataFrame(rows)
f.to_csv('data/processed/rnflt_features.csv', index=False)
print(f.merge(meta[['filename', 'label']]).groupby('label').mean(numeric_only=True).round(2).T)

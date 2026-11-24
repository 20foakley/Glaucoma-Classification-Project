import argparse, itertools, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument("--split_col", default="split")
p.add_argument("--split", default="test")
p.add_argument("--models", nargs="+", required=True,
               help="name=path.npy; one probability per patient, in meta row order for the split")
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()

meta = pd.read_csv("data/processed/meta.csv")
y = meta[meta[a.split_col] == a.split].glaucoma.values
n = len(y)

P = {}
for spec in a.models:
    name, path = spec.split("=", 1)
    s = np.load(path)
    assert s.shape == (n,), f"{name}: expected {n} values, got {s.shape}"
    P[name] = s

def ece(yy, pp, bins=15):
    e, ed = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(ed[:-1], ed[1:]):
        k = (pp > lo) & (pp <= hi)
        if k.any(): e += k.mean() * abs(yy[k].mean() - pp[k].mean())
    return e

rng = np.random.default_rng(0)
idx = rng.integers(0, n, (a.boot, n))
idx = idx[[len(np.unique(y[i])) == 2 for i in idx]]       # drop one-class resamples

auc = {k: np.array([roc_auc_score(y[i], s[i]) for i in idx]) for k, s in P.items()}
rows = []
for k, s in P.items():
    lo, hi = np.percentile(auc[k], [2.5, 97.5])
    r = dict(model=k, auroc=roc_auc_score(y, s), ci=f"[{lo:.3f}, {hi:.3f}]")
    if 0 <= s.min() and s.max() <= 1:
        e = np.array([ece(y[i], s[i]) for i in idx])
        r.update(ece=ece(y, s), ece_ci="[%.3f, %.3f]" % tuple(np.percentile(e, [2.5, 97.5])))
    rows.append(r)
print(f"\n{a.split}, n={n}, {len(idx)} bootstrap samples\n")
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nPAIRED DIFFERENCES (A - B in AUROC)")
out = []
for A, B in itertools.combinations(P, 2):
    d = auc[A] - auc[B]
    lo, hi = np.percentile(d, [2.5, 97.5])
    pv = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    out.append(dict(A=A, B=B, diff=roc_auc_score(y, P[A]) - roc_auc_score(y, P[B]),
                    ci=f"[{lo:.3f}, {hi:.3f}]", p=pv))
print(pd.DataFrame(out).round(3).to_string(index=False))

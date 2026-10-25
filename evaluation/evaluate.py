import argparse, glob, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument("--runs", default="runs_final")
p.add_argument("--arch", default="resnet18")
p.add_argument("--split_col", default="dev_split")
p.add_argument("--eval_split", default="dev_val")
p.add_argument("--boot", type=int, default=1000)
a = p.parse_args()

meta = pd.read_csv("data/processed/meta.csv")
m = meta[meta[a.split_col] == a.eval_split]
y, race = m.glaucoma.values, m.race.values

files = sorted(glob.glob(f"{a.runs}/{a.arch}_seed*_probs.npy"))
P = np.stack([np.load(f) for f in files])            # (seeds, patients)
assert P.shape[1] == len(y), "probs and eval split sizes differ"
print(f"{len(files)} seeds, {len(y)} patients ({a.eval_split})\n")

def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (p > lo) & (p <= hi)
        if k.any(): e += k.mean() * abs(y[k].mean() - p[k].mean())
    return float(e)

def metrics(y, p, race):
    out = {"auroc": roc_auc_score(y, p), "ece": ece(y, p)}
    for r in np.unique(race):
        k = race == r
        out[f"auroc_{r}"] = roc_auc_score(y[k], p[k])
    return out

# ---- per-seed metrics, then mean ± std
per_seed = pd.DataFrame([metrics(y, P[i], race) for i in range(len(P))])
print("PER-SEED (mean ± std)")
for c in per_seed.columns:
    print(f"  {c:34s} {per_seed[c].mean():.3f} ± {per_seed[c].std(ddof=0):.3f}")

# ---- ensemble metrics
pe = P.mean(0)
ens = metrics(y, pe, race)
print("\nENSEMBLE")
rng = np.random.default_rng(0)
idx = [rng.integers(0, len(y), len(y)) for _ in range(a.boot)]
def boot_ci(yy, pp):
    v = []
    for ii in idx:
        if len(np.unique(yy[ii])) == 2: v.append(roc_auc_score(yy[ii], pp[ii]))
    return np.percentile(v, [2.5, 97.5])
for c, v in ens.items():
    line = f"  {c:34s} {v:.3f}"
    if c == "auroc":
        lo, hi = boot_ci(y, pe); line += f"   95% CI [{lo:.3f}, {hi:.3f}]"
    elif c.startswith("auroc_"):
        k = race == c[6:]; yy, pp = y[k], pe[k]
        rr = np.random.default_rng(1)
        v2 = []
        for _ in range(a.boot):
            ii = rr.integers(0, len(yy), len(yy))
            if len(np.unique(yy[ii])) == 2: v2.append(roc_auc_score(yy[ii], pp[ii]))
        lo, hi = np.percentile(v2, [2.5, 97.5]); line += f"   95% CI [{lo:.3f}, {hi:.3f}]"
    print(line)
auc_by_race = {c: v for c, v in ens.items() if c.startswith("auroc_")}
print(f"  worst-to-best race gap: {max(auc_by_race.values()) - min(auc_by_race.values()):.3f}")

# ---- uncertainty: does it flag the wrong predictions?
ent = -(pe * np.log(pe + 1e-9) + (1 - pe) * np.log(1 - pe + 1e-9))   # predictive entropy
dis = P.std(0)                                                       # disagreement across seeds
err = ((pe > 0.5).astype(int) != y).astype(int)
print(f"\nUNCERTAINTY  (ensemble accuracy {1 - err.mean():.3f})")
print(f"  AUROC of entropy for detecting errors:      {roc_auc_score(err, ent):.3f}")
print(f"  AUROC of seed disagreement for errors:      {roc_auc_score(err, dis):.3f}")
print(f"  corr(entropy, disagreement):                {np.corrcoef(ent, dis)[0, 1]:.3f}")
q = pd.qcut(ent, 4, labels=["Q1 (sure)", "Q2", "Q3", "Q4 (unsure)"], duplicates="drop")
print("\n  error rate by entropy quartile:")
print(pd.DataFrame({"q": q, "err": err}).groupby("q", observed=True).err.agg(["mean", "size"]).round(3).to_string())

json.dump(dict(seeds=len(files), per_seed=per_seed.mean().to_dict(), ensemble=ens),
          open(f"{a.runs}/ensemble_{a.eval_split}.json", "w"), indent=1)
np.save(f"{a.runs}/ensemble_{a.eval_split}_probs.npy", pe)
import argparse, glob, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

p = argparse.ArgumentParser()
p.add_argument("--runs", default="runs_final")
p.add_argument("--arch", default="resnet18")
p.add_argument("--split_col", default="dev_split")
p.add_argument("--eval_split", default="dev_val")
p.add_argument("--out", default="results/figs")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

meta = pd.read_csv("data/processed/meta.csv")
m = meta[meta[a.split_col] == a.eval_split]
y, race = m.glaucoma.values, m.race.values
files = sorted(glob.glob(f"{a.runs}/{a.arch}_seed*_probs.npy"))
P = np.stack([np.load(f) for f in files]); pe = P.mean(0)
assert P.shape[1] == len(y)

short = {"Asian": "Asian", "Black or African American": "Black", "White or Caucasian": "White"}
col = {"Asian": "#1f77b4", "Black or African American": "#d62728", "White or Caucasian": "#2ca02c"}
races = [r for r in short if r in set(race)]
tag = f"{a.eval_split}, {len(P)} seeds, n={len(y)}"

def ece(yy, pp, bins=15):
    e = 0.0; ed = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(ed[:-1], ed[1:]):
        k = (pp > lo) & (pp <= hi)
        if k.any(): e += k.mean() * abs(yy[k].mean() - pp[k].mean())
    return e

def boot_ci(yy, pp, n=1000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[i])) == 2: v.append(roc_auc_score(yy[i], pp[i]))
    return np.percentile(v, [2.5, 97.5])

# ---------- panels ----------
def roc_panel(ax):
    f, t, _ = roc_curve(y, pe)
    ax.plot(f, t, "k", lw=2.5, label=f"All  (AUROC {roc_auc_score(y, pe):.3f})")
    for r in races:
        k = race == r; f, t, _ = roc_curve(y[k], pe[k])
        ax.plot(f, t, color=col[r], lw=1.6, label=f"{short[r]}  ({roc_auc_score(y[k], pe[k]):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (ensemble)")
    ax.legend(loc="lower right", fontsize=9)

def calib_panel(ax):
    ed = np.linspace(0, 1, 11); xs, ys, ns = [], [], []
    for lo, hi in zip(ed[:-1], ed[1:]):
        k = (pe > lo) & (pe <= hi)
        if k.sum() >= 5: xs.append(pe[k].mean()); ys.append(y[k].mean()); ns.append(k.sum())
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(xs, ys, "-", color="k", lw=1)
    ax.scatter(xs, ys, s=np.array(ns) * 1.5, c="#ff7f0e", edgecolor="k", zorder=3, label="bins (size = n)")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed glaucoma rate",
           title=f"Calibration  (ECE {ece(y, pe):.3f})", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=9)

def race_panel(ax):
    aucs, errs, dots = [], [], []
    for r in races:
        k = race == r; a_ = roc_auc_score(y[k], pe[k]); lo, hi = boot_ci(y[k], pe[k])
        aucs.append(a_); errs.append([a_ - lo, hi - a_])
        dots.append([roc_auc_score(y[k], P[i][k]) for i in range(len(P))])
    x = np.arange(len(races))
    ax.bar(x, aucs, color=[col[r] for r in races], alpha=0.75, yerr=np.array(errs).T, capsize=6)
    for i, d in enumerate(dots): ax.scatter([i] * len(d), d, color="k", s=14, zorder=3,
                                          label="individual seeds" if i == 0 else None)
    for i, v in enumerate(aucs): ax.text(i, 0.52, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set(xticks=x, xticklabels=[short[r] for r in races], ylim=(0.5, 1.0), ylabel="AUROC",
           title="AUROC by race (bars = ensemble, 95% CI)")
    ax.legend(loc="upper right", fontsize=9)

def seed_panel(ax):
    v = [roc_auc_score(y, P[i]) for i in range(len(P))]
    ax.bar(range(len(v)), v, color="#7f7f7f", alpha=0.8)
    ax.axhline(roc_auc_score(y, pe), color="k", ls="--", label=f"ensemble {roc_auc_score(y, pe):.3f}")
    for i, s in enumerate(v): ax.text(i, s + 0.002, f"{s:.3f}", ha="center", fontsize=9)
    ax.set(xticks=range(len(v)), xticklabels=[f"seed {i}" for i in range(len(v))],
           ylim=(min(v) - 0.03, max(max(v), roc_auc_score(y, pe)) + 0.02), ylabel="AUROC",
           title="Per-seed vs ensemble AUROC")
    ax.legend(loc="lower right", fontsize=9)

def unc_panel(ax):
    ent = -(pe * np.log(pe + 1e-9) + (1 - pe) * np.log(1 - pe + 1e-9))
    err = ((pe > 0.5).astype(int) != y).astype(int)
    q = pd.qcut(ent, 4, labels=False, duplicates="drop")
    rate = [err[q == i].mean() for i in range(q.max() + 1)]
    ax.bar(range(len(rate)), rate, color=plt.cm.OrRd(np.linspace(0.3, 0.9, len(rate))), edgecolor="k")
    for i, r in enumerate(rate): ax.text(i, r + 0.01, f"{r:.1%}", ha="center", fontweight="bold")
    ax.set(xticks=range(len(rate)), xticklabels=["Q1\n(sure)"] + [f"Q{i+1}" for i in range(1, len(rate) - 1)] + [f"Q{len(rate)}\n(unsure)"],
           ylabel="Error rate", title=f"Error rate by uncertainty quartile\n(entropy→error AUROC {roc_auc_score(err, ent):.3f})",
           ylim=(0, max(rate) + 0.08))

def hist_panel(ax):
    bins = np.linspace(0, 1, 26)
    ax.hist(pe[y == 0], bins, alpha=0.6, color="#1f77b4", label="no glaucoma")
    ax.hist(pe[y == 1], bins, alpha=0.6, color="#d62728", label="glaucoma")
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set(xlabel="Ensemble predicted probability", ylabel="Patients", title="Prediction separation by true class")
    ax.legend(fontsize=9)

panels = [("roc", roc_panel), ("calibration", calib_panel), ("auroc_by_race", race_panel),
          ("per_seed", seed_panel), ("uncertainty_vs_error", unc_panel), ("prob_hist", hist_panel)]

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
for ax, (_, fn) in zip(axs.flat, panels): fn(ax)
fig.suptitle(f"Glaucoma 2D RNFLT CNN: {tag}", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{a.out}/evaluation_dashboard_{a.eval_split}.png", dpi=150); plt.close(fig)

for name, fn in panels:
    f, ax = plt.subplots(figsize=(6, 5)); fn(ax); f.tight_layout()
    f.savefig(f"{a.out}/{name}_{a.eval_split}.png", dpi=150); plt.close(f)
print("saved to", a.out, os.listdir(a.out))
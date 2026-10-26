import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from src.data.dataset import GFDataset
from src.eval.metrics import ece

p = argparse.ArgumentParser()
p.add_argument("--split", default="val", choices=["val", "test"])
p.add_argument("--final", action="store_true", help="required to evaluate on test")
a = p.parse_args()
assert a.split != "test" or a.final, "test is touched once at the end: pass --final"
SPLIT = a.split
OUT = os.environ.get("FIG_OUT", "results/figs")
os.makedirs(OUT, exist_ok=True)

# fixed color per model
COLOR = {"Rule: vCDR": "#2a78d6", "Rule: ring_mean": "#eb6834", "Rule: ring_inf": "#1baf7a",
         "LogReg (12 features)": "#eda100", "LogReg (no age)": "#e87ba4"}
RACE_COLOR = {"asian": "#2a78d6", "black": "#eb6834", "white": "#1baf7a"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
                     "axes.edgecolor": "#8a8984", "font.size": 10})


def get(split):
    ds = GFDataset(split, modalities=("tab",))
    return ds.tab_matrix[ds.idx], ds.meta.iloc[ds.idx].reset_index(drop=True)


x_tr, m_tr = get("train")
x_ev, m_ev = get(SPLIT)
y = m_ev["label"].to_numpy(dtype=int)
race = m_ev["race"].to_numpy(dtype=str)

# scores for every model
scores: dict[str, np.ndarray] = {}
for name, col, sign in [("Rule: vCDR", "vcdr", 1), ("Rule: ring_mean", "ring_mean", -1),
                        ("Rule: ring_inf", "ring_inf", -1)]:
    scores[name] = sign * m_ev[col].fillna(m_tr[col].median()).to_numpy(dtype=float)
for name, cols in [("LogReg (12 features)", slice(None)), ("LogReg (no age)", slice(0, 11))]:
    clf = LogisticRegression(max_iter=2000).fit(x_tr[:, cols], m_tr["label"])
    scores[name] = clf.predict_proba(x_ev[:, cols])[:, 1]

rng = np.random.default_rng(0)
def auroc_ci(y, s, n=1000):
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


# ROC curves
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0, 1], [0, 1], color="#b5b4ae", lw=1, ls="--")
for name in ["Rule: vCDR", "Rule: ring_mean", "Rule: ring_inf", "LogReg (12 features)"]:
    fpr, tpr, _ = roc_curve(y, scores[name])
    ax.plot(fpr, tpr, color=COLOR[name], lw=2,
            label=f"{name}  ({roc_auc_score(y, scores[name]):.3f})")
ax.set(xlabel="False positive rate", ylabel="True positive rate",
       title=f"ROC curves ({SPLIT}, n={len(y)})", xlim=(0, 1), ylim=(0, 1.01))
ax.legend(title="Model (AUROC)", loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/baseline_roc_{SPLIT}.png", dpi=150)

# AUROC with 95% bootstrap CI
names = list(scores)
auc = np.array([roc_auc_score(y, scores[n]) for n in names])
ci = np.array([auroc_ci(y, scores[n]) for n in names])
fig, ax = plt.subplots(figsize=(6.5, 3.6))
pos = np.arange(len(names))[::-1]
ax.barh(pos, auc, height=0.6, color=[COLOR[n] for n in names])
ax.errorbar(auc, pos, xerr=[auc - ci[:, 0], ci[:, 1] - auc],
            fmt="none", ecolor="#3d3c39", capsize=3, lw=1)
for p, a, hi in zip(pos, auc, ci[:, 1]):
    ax.text(hi + 0.008, p, f"{a:.3f}", va="center", color="#3d3c39")
ax.set_yticks(pos, names)
ax.set(xlim=(0.5, 1.0), xlabel="AUROC (95% bootstrap CI), 0.5 = chance", title=f"Baseline AUROC ({SPLIT})")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/baseline_auroc_ci_{SPLIT}.png", dpi=150)

# Calibration (only the models that output probabilities)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0, 1], [0, 1], color="#b5b4ae", lw=1, ls="--", label="Perfect calibration")
for name in ["LogReg (12 features)", "LogReg (no age)"]:
    frac, mean_p = calibration_curve(y, scores[name], n_bins=8, strategy="quantile")
    ax.plot(mean_p, frac, color=COLOR[name], lw=2, marker="o", ms=6,
            label=f"{name}  (ECE {ece(y, scores[name]):.3f})")
ax.set(xlabel="Predicted probability of glaucoma", ylabel="Observed glaucoma rate",
       title=f"Calibration ({SPLIT})", xlim=(0, 1), ylim=(0, 1))
ax.legend(loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/baseline_calibration_{SPLIT}.png", dpi=150)

# AUROC per race
races = sorted(RACE_COLOR)
fig, ax = plt.subplots(figsize=(8, 4))
w = 0.26
for k, r in enumerate(races):
    m = np.asarray(race == r, dtype=bool)
    vals = np.array([roc_auc_score(y[m], scores[n][m]) for n in names])
    ax.bar(np.arange(len(names)) + (k - 1) * w, vals, width=w * 0.92,
           color=RACE_COLOR[r], label=f"{r} (n={m.sum()})")
ax.set_xticks(np.arange(len(names)), [n.replace(" (", "\n(") for n in names])
ax.set(ylim=(0.5, 1.0), ylabel="AUROC (0.5 = chance)", title=f"AUROC by race ({SPLIT})")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=3, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/baseline_auroc_by_race_{SPLIT}.png", dpi=150)

print("saved to", OUT)
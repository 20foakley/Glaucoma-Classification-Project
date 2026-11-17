import numpy as np
from sklearn.metrics import roc_auc_score

def _boot(y, scores, n, seed):
    rng, idx = np.random.default_rng(seed), np.arange(len(y))
    out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if y[s].min() == y[s].max():
            continue
        out.append(scores(s))
    return np.array(out)

def auc_ci(df, n=2000, seed=0):
    y, p = df["label"].values, df["prob"].values
    b = _boot(y, lambda s: roc_auc_score(y[s], p[s]), n, seed)
    return roc_auc_score(y, p), *np.percentile(b, [2.5, 97.5])

def paired_diff(a, b, n=2000, seed=0):
    d = a.merge(b, on="filename", suffixes=("_a", "_b"))
    y, pa, pb = d["label_a"].values, d["prob_a"].values, d["prob_b"].values
    diffs = _boot(y, lambda s: roc_auc_score(y[s], pa[s]) - roc_auc_score(y[s], pb[s]), n, seed)
    obs = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return obs, *np.percentile(diffs, [2.5, 97.5]), p

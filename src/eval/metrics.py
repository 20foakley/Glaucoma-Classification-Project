import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def ece(y, p, bins=10):
    y = np.array(y, float)
    p = np.array(p, float)
    b = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum((b == k).mean() * abs((y[b == k]).mean() - (p[b == k]).mean()) for k in range(bins) if (b == k).any()))

def report(df):
    out = {"auroc": roc_auc_score(df["label"], df["prob"]), 
           "auprc": average_precision_score(df["label"], df["prob"]),
           "ece": ece(df["label"], df["prob"])}
    
    for r in sorted(df["race"].unique()):
        d = df[df["race"] == r]
        out[f"auroc_{r}"] = roc_auc_score(d["label"], d["prob"])

    return out

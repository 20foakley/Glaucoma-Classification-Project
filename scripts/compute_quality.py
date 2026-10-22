import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

m = (pd.read_csv("data/processed/meta.csv").merge(pd.read_csv("data/processed/rnflt_features.csv"), on="filename"))
r = np.load("data/processed/rnflt.npy", mmap_mode="r")

m["zero_frac"] = [(np.array(r[i]) == 0).mean() for i in range(len(r))]
m[["filename", "zero_frac"]].to_csv("data/processed/quality.csv", index=False)

print("zero_frac alone AUROC:", round(roc_auc_score(m["label"], m["zero_frac"]), 3))
a = m.dropna(subset=["ring_inf"]); c = a[a["zero_frac"] < 0.05]
print("clean scans:", len(c), "of", len(a))
print("ring_inf AUROC all  :", round(roc_auc_score(a["label"], -a["ring_inf"]), 3))
print("ring_inf AUROC clean:", round(roc_auc_score(c["label"], -c["ring_inf"]), 3))
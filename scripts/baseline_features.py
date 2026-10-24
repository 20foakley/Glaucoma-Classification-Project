import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.data.dataset import GFDataset
from src.eval.metrics import report

def get(split):
    ds = GFDataset(split, modalities=("tab",))
    return ds.tab_matrix[ds.idx], ds.meta.iloc[ds.idx]

x_train, m_train = get("train")
os.makedirs("results/preds", exist_ok=True)
SPLITS = ["val"]

#  vCDR rule (one feauture), and two RNFL rules (two features)
for split in SPLITS:
    _, m = get(split)
    for col, sign in [("vcdr", 1), ("ring_inf", -1), ("ring_mean", -1)]:
        s = sign * m[col].fillna(m_train[col].median())
        print(f"{split} rule  {col:10s} AUROC:", round(roc_auc_score(m["label"], s), 3))

# Logistic regression, with and without age
for name, cols in [("all12", slice(None)), ("no_age", slice(0, 11))]:
    clf = LogisticRegression(max_iter=2000).fit(x_train[:, cols], m_train["label"])

    for split in SPLITS:
        x, m = get(split)
        p = clf.predict_proba(x[:, cols])[:, 1]
        df = pd.DataFrame({"filename": m["filename"].values, "prob": p,
                           "label": m["label"].values, "race": m["race"].values})
        if name == "all12":
            df.to_csv(f"results/preds/logreg_seed0_{split}.csv", index=False)
        print(name, split, {k: round(v, 3) for k, v in report(df).items()})
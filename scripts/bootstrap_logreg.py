import pandas as pd

from src.eval.bootstrap import auc_ci, paired_diff

for split in ["val", "test"]:
    df = pd.read_csv(f"results/preds/logreg_all12_seed0_{split}.csv")
    a, lo, hi = auc_ci(df)
    print(f"{split:4s} all   {a:.3f} ({lo:.3f}-{hi:.3f})")
    for r in ["asian", "black", "white"]:
        a, lo, hi = auc_ci(df[df["race"] == r])
        print(f"{split:4s} {r:5s} {a:.3f} ({lo:.3f}-{hi:.3f})")

# Paired test: all 12 features vs no age, on the same patients
for split in ["val", "test"]:
    a = pd.read_csv(f"results/preds/logreg_all12_seed0_{split}.csv")
    b = pd.read_csv(f"results/preds/logreg_no_age_seed0_{split}.csv")
    d, lo, hi, p = paired_diff(a, b)
    print(f"{split:4s} all12 - no_age  {d:+.3f} ({lo:+.3f} to {hi:+.3f})  p={p:.3f}")

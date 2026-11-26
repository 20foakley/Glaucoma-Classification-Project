import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.eval.bootstrap import auc_ci, paired_diff
from src.eval.metrics import ece


def load_runs(preds_dir):
    runs = defaultdict(dict)
    for path in sorted(glob.glob(f"{preds_dir}/*_seed*_*.csv")):
        base = os.path.basename(path)[:-4]
        model, rest = base.rsplit("_seed", 1)
        _, split = rest.split("_", 1)
        runs[model].setdefault(split, []).append(pd.read_csv(path))
    return runs


def seed_ensemble(frames):
    all_frames = pd.concat(frames, ignore_index=True)
    return all_frames.groupby("filename", as_index=False).agg(
        prob=("prob", "mean"), label=("label", "first"), race=("race", "first")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/preds")
    ap.add_argument("--ref", default="single_rnflt", help="model every other model is compared with")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default="results/summary_test.csv")
    args = ap.parse_args()

    runs = load_runs(args.preds)

    if not runs:
        raise SystemExit(f"no prediction files found in {args.preds}")
    ens = {m: seed_ensemble(r["test"]) for m, r in runs.items() if "test" in r}
    ref = ens.get(args.ref)

    rows = []
    for model, r in sorted(runs.items()):
        if "test" not in r:
            continue
        row = {"model": model, "n_seeds": len(r["test"])}
        for split in ["val", "test"]:
            aucs = [float(roc_auc_score(f["label"], f["prob"])) for f in r.get(split, [])]
            if aucs:
                row[f"{split}_auc_mean"] = np.mean(aucs)
                row[f"{split}_auc_std"] = np.std(aucs)
        e = ens[model]
        row["ens_test_auc"], row["ci_lo"], row["ci_hi"] = auc_ci(e, n=args.n_boot)
        row["ens_test_ece"] = ece(e["label"], e["prob"])
        for race in sorted(e["race"].unique()):
            auc, lo, hi = auc_ci(e[e["race"] == race], n=args.n_boot)
            row[f"auc_{race}"], row[f"{race}_lo"], row[f"{race}_hi"] = auc, lo, hi
        if ref is not None and model != args.ref:
            d, lo, hi, p = paired_diff(e, ref, n=args.n_boot)
            row["diff_vs_ref"], row["diff_lo"], row["diff_hi"], row["p_vs_ref"] = d, lo, hi, p
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("ens_test_auc", ascending=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"reference for paired tests: {args.ref}\n")
    for _, t in table.iterrows():
        line = (f"{t['model']:22s} seeds {t['n_seeds']}  "
                f"val {t.get('val_auc_mean', np.nan):.3f}±{t.get('val_auc_std', np.nan):.3f}  "
                f"test {t['test_auc_mean']:.3f}±{t['test_auc_std']:.3f}  "
                f"ensemble {t['ens_test_auc']:.3f} [{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]  "
                f"ECE {t['ens_test_ece']:.3f}")
        if not pd.isna(t.get("diff_vs_ref", np.nan)):
            line += (f"  vs ref {t['diff_vs_ref']:+.3f} "
                     f"[{t['diff_lo']:+.3f}, {t['diff_hi']:+.3f}] p={t['p_vs_ref']:.3f}")
        print(line)
    print(f"\nfull table (with per-race AUROC and CIs): {args.out}")


if __name__ == "__main__":
    main()

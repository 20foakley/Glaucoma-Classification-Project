import argparse
import os

import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="model names, e.g. single_rnflt single_tab")
    ap.add_argument("--name", required=True, help="output name, e.g. late_rnflt+tab")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--preds", default="results/preds")
    args = ap.parse_args()

    keep = ["filename", "label", "race", "md", "zero_frac"]
    for seed in args.seeds:
        for split in ["val", "test"]:
            paths = [f"{args.preds}/{m}_seed{seed}_{split}.csv" for m in args.inputs]
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                print("skipping, missing:", *missing)
                continue
            frames = [pd.read_csv(p) for p in paths]
            base = frames[0][keep]
            probs = []
            for f in frames:
                m = base[["filename", "label"]].merge(f[["filename", "label", "prob"]], on="filename",
                                                      suffixes=("", "_other"))
                assert len(m) == len(base), "the input files have different patients"
                assert (m["label"] == m["label_other"]).all(), "labels differ between inputs"
                probs.append(m["prob"].values)
            out = base.assign(prob=sum(probs) / len(probs))
            out.to_csv(f"{args.preds}/{args.name}_seed{seed}_{split}.csv", index=False)
            print("wrote", f"{args.preds}/{args.name}_seed{seed}_{split}.csv")


if __name__ == "__main__":
    main()
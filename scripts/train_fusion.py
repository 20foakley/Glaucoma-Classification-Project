import argparse
import os
import random
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.data.dataset import GFDataset
from src.models.fusion import FusionModel

IMAGE_BRANCHES = ["rnflt", "oct"]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_shift(x, max_px=8):
    B, C, H, W = x.shape
    xp = nn.functional.pad(x, (max_px, max_px, max_px, max_px))
    dy = torch.randint(0, 2 * max_px + 1, (B,)).tolist()
    dx = torch.randint(0, 2 * max_px + 1, (B,)).tolist()
    return torch.stack([xp[i, :, dy[i]:dy[i] + H, dx[i]:dx[i] + W] for i in range(B)])

def to_device(batch, branches, dev, train):
    out = {}
    for b in branches:
        x = batch[b].to(dev)
        if train and b in IMAGE_BRANCHES:
            x = random_shift(x)
        out[b] = x
    return out

@torch.no_grad()
def predict(model, loader, branches, dev, gate_prefix, leave_one_out=False):
    model.eval()
    parts = []
    for batch in loader:
        x = to_device(batch, branches, dev, train=False)
        logit, gates = model(x)
        part = pd.DataFrame({
            "filename": list(batch["filename"]),
            "prob": torch.sigmoid(logit).cpu().numpy(),
            "label": batch["label"].numpy().astype(int),
            "race": list(batch["race"]),
            "md": batch["md"].numpy(),
            "zero_frac": batch["zero_frac"].numpy(),
        })
        if gates is not None:
            g = gates.cpu().numpy()
            for j, b in enumerate(branches):
                part[f"{gate_prefix}_{b}"] = g[:, j]
        if leave_one_out:
            for b in branches:
                lo, _ = model(x, drop=b)
                part[f"prob_no_{b}"] = torch.sigmoid(lo).cpu().numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", nargs="+", required=True, choices=["rnflt", "oct", "tab"])
    ap.add_argument("--fusion", required=True, choices=["single", "concat", "gmu", "xattn"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--early_stop", action="store_true",
                    help="keep the best-val-AUROC epoch and stop after --patience bad epochs; "
                         "default keeps the last epoch (Veda's recipe: val never picks the model)")
    ap.add_argument("--patience", type=int, default=5, help="only used with --early_stop")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--p_drop", type=float, default=0.0,
                    help="modality dropout rate (concat / gmu only), e.g. 0.15")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    n = len(args.branches)
    if args.fusion == "single" and n != 1:
        ap.error("single need exactly one branch")
    if args.fusion == "xattn" and n != 2:
        ap.error("xattn needs exactly two branches")
    if args.fusion in ("concat", "gmu") and n < 2:
        ap.error(f"{args.fusion} needs at least two branches")
    if args.p_drop > 0 and args.fusion not in ("concat", "gmu"):
        ap.error("--p_drop works with concat or gmu only")

    set_seed(args.seed)
    dev = pick_device()
    name = f"{args.fusion}_{'+'.join(args.branches)}"
    if args.p_drop > 0:
        name += f"_pd{round(args.p_drop * 100)}"
    gate_prefix = "xgate" if args.fusion == "xattn" else "gate"
    print(f"== {name}  seed {args.seed}  device {dev}")

    mods = tuple(args.branches)
    g = torch.Generator().manual_seed(args.seed)
    train_ds = GFDataset("train", modalities=mods)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=g, num_workers=0)
    val_dl = DataLoader(GFDataset("val", modalities=mods), batch_size=64, num_workers=0)
    test_dl = DataLoader(GFDataset("test", modalities=mods), batch_size=64, num_workers=0)

    model = FusionModel(args.branches, args.fusion, p_drop=args.p_drop).to(dev)
    loo = args.p_drop > 0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_dl:
            x = to_device(batch, args.branches, dev, train=True)
            y = batch["label"].float().to(dev)
            logit, _ = model(x)
            loss = loss_fn(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        val = predict(model, val_dl, args.branches, dev, gate_prefix)
        val_auc = roc_auc_score(val["label"], val["prob"])
        print(f"epoch {epoch:2d}  train loss {total / len(train_ds):.4f}  val AUROC {val_auc:.4f}"
              f"  lr {opt.param_groups[0]['lr']:.2e}")
        if not args.early_stop:                # default: keep the latest epoch, val only monitors
            best_auc, best_state, best_epoch = val_auc, copy.deepcopy(model.state_dict()), epoch
        elif val_auc > best_auc:               # optional: early stopping on validation AUROC
            best_auc, best_state, best_epoch, bad = val_auc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    os.makedirs(f"{args.out}/preds", exist_ok=True)
    os.makedirs(f"{args.out}/checkpoints", exist_ok=True)
    for split, dl in [("val", val_dl), ("test", test_dl)]:
        predict(model, dl, args.branches, dev, gate_prefix, leave_one_out=loo).to_csv(
            f"{args.out}/preds/{name}_seed{args.seed}_{split}.csv", index=False)
    torch.save(best_state, f"{args.out}/checkpoints/{name}_seed{args.seed}.pt")
    kept = "best" if args.early_stop else "last"
    print(f"kept {kept} epoch {best_epoch} (val AUROC {best_auc:.4f})"
          f"  -> {args.out}/preds/{name}_seed{args.seed}_*.csv")


if __name__ == "__main__":
    main()
import argparse, json, os, numpy as np, torch, torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from dataset3d import OCTVolDataset

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--wd", type=float, default=1e-4)
p.add_argument("--arch", default="r3d_18", choices=["r3d_18", "mc3_18"])
p.add_argument("--size", type=int, default=64)
p.add_argument("--pretrained", action="store_true")
p.add_argument("--aug", action="store_true")
p.add_argument("--save_last", action="store_true")
p.add_argument("--split_col", default="dev_split")
p.add_argument("--train_split", default="dev_train")
p.add_argument("--eval_split", default="dev_val")
p.add_argument("--final", action="store_true", help="required to evaluate on the test split")
p.add_argument("--out", default="results/runs3d")
a = p.parse_args()

assert a.eval_split != "test" or a.final, "test split is touched once at the end: pass --final"
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

tr = OCTVolDataset(a.split_col, a.train_split, aug=a.aug, size=a.size)
ev = OCTVolDataset(a.split_col, a.eval_split, aug=False, size=a.size)
tl = DataLoader(tr, a.bs, shuffle=True, num_workers=2, drop_last=True)
el = DataLoader(ev, 16, shuffle=False, num_workers=2)

net = getattr(torchvision.models.video, a.arch)(weights="DEFAULT" if a.pretrained else None)
old = net.stem[0]
new = nn.Conv3d(1, 64, old.kernel_size, old.stride, old.padding, bias=False)
if a.pretrained:                                    # average the RGB filters into one channel
    new.weight.data = old.weight.data.mean(1, keepdim=True)
net.stem[0] = new
net.fc = nn.Linear(net.fc.in_features, 1)
net = net.to(dev)

opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
crit = nn.BCEWithLogitsLoss()
amp = dev == "cuda"
scaler = torch.amp.GradScaler(enabled=amp)

def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

@torch.no_grad()
def predict(loader):
    net.eval(); ps = []
    for x, _ in loader:
        with torch.autocast(dev, enabled=amp):
            lo = net(x.to(dev)).squeeze(1)
        ps.append(torch.sigmoid(lo.float()).cpu().numpy())
    return np.concatenate(ps)

def metrics(y, p, race):
    out = dict(auroc=float(roc_auc_score(y, p)), ece=ece(y, p))
    for r in np.unique(race):
        m = race == r
        out[f"auroc_{r}"] = float(roc_auc_score(y[m], p[m]))
    return out

name = f"{a.arch}_seed{a.seed}"
os.makedirs(a.out, exist_ok=True)
best = -1
for ep in range(a.epochs):
    net.train(); tot = 0
    for x, y in tl:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(dev, enabled=amp):
            loss = crit(net(x).squeeze(1).float(), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
    sched.step()
    pr = predict(el); m = metrics(ev.y, pr, ev.race)
    print(f"ep{ep:02d} loss={tot/len(tl):.4f} val_auroc={m['auroc']:.4f} ece={m['ece']:.4f}", flush=True)
    if a.save_last:
        best_ep, best_m, best_p = ep, m, pr
    elif m["auroc"] > best:
        best, best_ep, best_m, best_p = m["auroc"], ep, m, pr
        torch.save(net.state_dict(), f"{a.out}/{name}.pt")
if a.save_last:
    torch.save(net.state_dict(), f"{a.out}/{name}.pt")

print(f"\n{'final' if a.save_last else 'best'} epoch {best_ep}:", json.dumps(best_m, indent=1))
np.save(f"{a.out}/{name}_probs.npy", best_p)
json.dump(dict(args=vars(a), best_epoch=best_ep, metrics=best_m), open(f"{a.out}/{name}.json", "w"), indent=1)

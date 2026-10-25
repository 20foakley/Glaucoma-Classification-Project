import argparse, json, os, numpy as np, torch, torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from dataset import RNFLTDataset

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--wd", type=float, default=1e-4)
p.add_argument("--arch", default="resnet18")
p.add_argument("--pretrained", action="store_true")
p.add_argument("--aug", action="store_true")
p.add_argument("--use_mask", action="store_true")
p.add_argument("--split_col", default="dev_split")
p.add_argument("--train_split", default="dev_train")
p.add_argument("--eval_split", default="dev_val")
p.add_argument("--save_last", action="store_true", help="keep final-epoch weights/probs instead of best-epoch")
p.add_argument("--final", action="store_true", help="required to evaluate on the test split")
p.add_argument("--out", default="runs")
a = p.parse_args()

assert a.eval_split != "test" or a.final, "test split is touched once at the end: pass --final"
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

tr = RNFLTDataset(a.split_col, a.train_split, aug=a.aug, use_mask=a.use_mask)
ev = RNFLTDataset(a.split_col, a.eval_split, aug=False, use_mask=a.use_mask)
tl = DataLoader(tr, a.bs, shuffle=True, num_workers=2, drop_last=True)
el = DataLoader(ev, 64, shuffle=False, num_workers=2)

in_ch = 2 if a.use_mask else 1
net = getattr(torchvision.models, a.arch)(weights="DEFAULT" if a.pretrained else None)
w = net.conv1.weight.data
net.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
if a.pretrained:                                    # reuse ImageNet filters (averaged over RGB)
    net.conv1.weight.data = w.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) / in_ch
net.fc = nn.Linear(net.fc.in_features, 1)
net = net.to(dev)

opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
crit = nn.BCEWithLogitsLoss()

def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

@torch.no_grad()
def predict(loader):
    net.eval(); ps = []
    for x, _ in loader:
        ps.append(torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().numpy())
    return np.concatenate(ps)

def metrics(y, p, race):
    out = dict(auroc=float(roc_auc_score(y, p)), ece=ece(y, p))
    for r in np.unique(race):
        m = race == r
        out[f"auroc_{r}"] = float(roc_auc_score(y[m], p[m]))
    return out

name = f"{a.arch}_seed{a.seed}"
os.makedirs(a.out, exist_ok=True)
best, best_ep = -1, -1
for ep in range(a.epochs):
    net.train(); tot = 0
    for x, y in tl:
        x, y = x.to(dev), y.to(dev)
        loss = crit(net(x).squeeze(1), y)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    sched.step()
    pr = predict(el); m = metrics(ev.y, pr, ev.race)
    print(f"ep{ep:02d} loss={tot/len(tl):.4f} val_auroc={m['auroc']:.4f} ece={m['ece']:.4f}")
    if a.save_last:
        best_ep, best_m, best_p = ep, m, pr          # always the latest epoch
    elif m["auroc"] > best:
        best, best_ep, best_m, best_p = m["auroc"], ep, m, pr
        torch.save(net.state_dict(), f"{a.out}/{name}.pt")

if a.save_last:
    torch.save(net.state_dict(), f"{a.out}/{name}.pt")

print(f"\n{'final' if a.save_last else 'best'} epoch {best_ep}:", json.dumps(best_m, indent=1))
np.save(f"{a.out}/{name}_probs.npy", best_p)        # saved for the 5-seed ensemble
json.dump(dict(args=vars(a), best_epoch=best_ep, metrics=best_m), open(f"{a.out}/{name}.json", "w"), indent=1)
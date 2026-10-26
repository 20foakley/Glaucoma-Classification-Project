import argparse, glob, os, numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import DataLoader
from dataset import RNFLTDataset

p = argparse.ArgumentParser()
p.add_argument("--runs", default="results/runs_full", help="folder with the saved .pt weights")
p.add_argument("--arch", default="resnet18")
p.add_argument("--split_col", default="split")
p.add_argument("--eval_split", default="test")
p.add_argument("--use_mask", action="store_true")
p.add_argument("--out", default=None, help="default: <runs>_<eval_split>")
p.add_argument("--final", action="store_true", help="required to score the test split")
a = p.parse_args()

assert a.eval_split != "test" or a.final, "test split is touched once at the end: pass --final"
out = a.out or f"{a.runs}_{a.eval_split}"
os.makedirs(out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

ds = RNFLTDataset(a.split_col, a.eval_split, aug=False, use_mask=a.use_mask)
dl = DataLoader(ds, 64, shuffle=False, num_workers=2)

def build():
    net = getattr(torchvision.models, a.arch)(weights=None)
    net.conv1 = nn.Conv2d(2 if a.use_mask else 1, 64, 7, 2, 3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net.to(dev)

files = sorted(glob.glob(f"{a.runs}/{a.arch}_seed*.pt"))
assert files, f"no weights found in {a.runs}"
for f in files:
    net = build(); net.load_state_dict(torch.load(f, map_location=dev)); net.eval()
    ps = []
    with torch.no_grad():
        for x, _ in dl:
            ps.append(torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().numpy())
    name = os.path.basename(f)[:-3]
    np.save(f"{out}/{name}_probs.npy", np.concatenate(ps))
    print(f"{name}: {len(ds)} patients scored")
print("saved to", out)
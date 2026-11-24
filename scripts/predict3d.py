import argparse, glob, os, numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import DataLoader
from dataset3d import OCTVolDataset

p = argparse.ArgumentParser()
p.add_argument("--runs", default="results/runs3d_full")
p.add_argument("--arch", default="r3d_18", choices=["r3d_18", "mc3_18"])
p.add_argument("--size", type=int, default=64)
p.add_argument("--split_col", default="split")
p.add_argument("--eval_split", default="test")
p.add_argument("--out", default=None, help="default: <runs>_<eval_split>")
p.add_argument("--final", action="store_true", help="required to score the test split")
a = p.parse_args()

assert a.eval_split != "test" or a.final, "test split is touched once at the end: pass --final"
out = a.out or f"{a.runs}_{a.eval_split}"
os.makedirs(out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = dev == "cuda"

ds = OCTVolDataset(a.split_col, a.eval_split, aug=False, size=a.size)
dl = DataLoader(ds, 16, shuffle=False, num_workers=2)

def build():
    net = getattr(torchvision.models.video, a.arch)(weights=None)
    old = net.stem[0]
    net.stem[0] = nn.Conv3d(1, 64, old.kernel_size, old.stride, old.padding, bias=False)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net.to(dev)

files = sorted(glob.glob(f"{a.runs}/{a.arch}_seed*.pt"))
assert files, f"no weights found in {a.runs}"
for f in files:
    net = build(); net.load_state_dict(torch.load(f, map_location=dev)); net.eval()
    ps = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast(dev, enabled=amp):
                lo = net(x.to(dev)).squeeze(1)
            ps.append(torch.sigmoid(lo.float()).cpu().numpy())
    name = os.path.basename(f)[:-3]
    np.save(f"{out}/{name}_probs.npy", np.concatenate(ps))
    print(f"{name}: {len(ds)} patients scored")
print("saved to", out)

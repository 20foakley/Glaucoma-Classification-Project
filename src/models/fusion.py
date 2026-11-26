import torch
import torch.nn as nn

from src.models.branch_encoders import make_encoder


class SingleHead(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 1
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(dims[0], 1))

    def forward(self, feats):
        return self.fc(feats[0]).squeeze(1), None


HEADS = {"single": SingleHead}

class FusionModel(nn.Module):
    def __init__(self, branches, fusion, p_drop=0.0):
        super().__init__()
        self.p_drop = p_drop
        self.branches = list(branches)
        self.encoders = nn.ModuleDict({b: make_encoder(b) for b in self.branches})
        dims = [self.encoders[b].out_dim for b in self.branches]
        self.fuse = HEADS[fusion](dims)

    def forward(self, branch, drop=None):
        feats = [self.encoders[b](branch[b]) for b in self.branches]
        return self.fuse(feats)

if __name__ == "__main__":
    m = FusionModel(["rnflt"], "single")
    logit, gates = m({"rnflt": torch.randn(4, 1, 200, 200)})
    print(logit.shape, gates)

    m = FusionModel(["tab"], "single")
    logit, gates = m({"tab": torch.randn(4, 12)})
    print(logit.shape, gates)

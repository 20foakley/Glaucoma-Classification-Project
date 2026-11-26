import torch
import torch.nn as nn
from torchvision.models import resnet18

class ImageEncoder(nn.Module):
    out_dim = 512
    token_dim = 512

    def __init__(self, in_ch):
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Identity() # type: ignore[assignment]
        self.net = net

    def forward(self, x):
        return self.net(x)

    def tokens(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        return x.flatten(2).transpose(1, 2)

class TabularEncoder(nn.Module):
    out_dim = 32
    token_dim = 64

    def __init__(self, n_feat = 12):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.feat_w = nn.Parameter(torch.randn(n_feat, self.token_dim) * 0.02)
        self.feat_b = nn.Parameter(torch.zeros(n_feat, self.token_dim))

    def forward(self, x):
        return self.mlp(x)

    def tokens(self, x):
        return x.unsqueeze(-1) * self.feat_w + self.feat_b

def make_encoder(branch):
    if branch =="rnflt":
        return ImageEncoder(in_ch=1)
    if branch == "oct":
        return ImageEncoder(in_ch=16)
    if branch == "tab":
        return TabularEncoder(n_feat=12)
    raise ValueError(f"Unknown branch: {branch}")
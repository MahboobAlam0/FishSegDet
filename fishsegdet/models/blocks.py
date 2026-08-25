import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, ic, oc, k=3, s=1, g=1, d=1, act=True):
        super().__init__()
        p = d * (k - 1) // 2
        self.conv = nn.Conv2d(ic, oc, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(oc, momentum=0.03, eps=1e-3)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, ic, oc, k=3, s=1):
        super().__init__()
        self.dw = ConvBN(ic, ic, k, s, g=ic)
        self.pw = ConvBN(ic, oc, 1)

    def forward(self, x):
        return self.pw(self.dw(x))

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBN, DWConv


class BiFPNNode(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.w    = nn.Parameter(torch.ones(2))
        self.conv = DWConv(ch, ch)
        self.act  = nn.ReLU()

    def forward(self, x1, x2):
        w = self.act(self.w) / (self.act(self.w).sum() + 1e-4)
        return self.conv(w[0] * x1 + w[1] * x2)


class BiFPNNode3(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.w    = nn.Parameter(torch.ones(3))
        self.conv = DWConv(ch, ch)
        self.act  = nn.ReLU()

    def forward(self, x1, x2, x3):
        w = self.act(self.w) / (self.act(self.w).sum() + 1e-4)
        return self.conv(w[0] * x1 + w[1] * x2 + w[2] * x3)


class BiFPNLayer(nn.Module):
    def __init__(self, ch, num_levels=4):
        super().__init__()
        self.n      = num_levels
        self.td     = nn.ModuleList([BiFPNNode(ch) for _ in range(num_levels - 1)])
        self.bu_mid = nn.ModuleList([BiFPNNode3(ch) for _ in range(max(0, num_levels - 2))])
        self.bu_top = BiFPNNode(ch)

    def forward(self, feats):
        n  = self.n
        td = list(feats)
        for i in range(n - 2, -1, -1):
            td[i] = self.td[i](
                feats[i], F.interpolate(td[i + 1], size=feats[i].shape[2:], mode="nearest"))
        out = [td[0]]
        for i in range(1, n - 1):
            out.append(self.bu_mid[i - 1](feats[i], td[i], F.max_pool2d(out[-1], 2, 2)))
        out.append(self.bu_top(td[n - 1], F.max_pool2d(out[-1], 2, 2)))
        return out


class BiFPN(nn.Module):
    def __init__(self, in_channels, neck_ch=256, repeats=3, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        c3, c4, c5 = in_channels
        self.lat3 = ConvBN(c3, neck_ch, 1)
        self.lat4 = ConvBN(c4, neck_ch, 1)
        self.lat5 = ConvBN(c5, neck_ch, 1)
        if num_levels >= 4:
            self.p6_down = ConvBN(neck_ch, neck_ch, 3, 2)
        self.layers = nn.ModuleList(
            [BiFPNLayer(neck_ch, num_levels) for _ in range(repeats)])
        self.out_channels = [neck_ch] * num_levels

    def forward(self, feats):
        p3  = self.lat3(feats[0])
        p4  = self.lat4(feats[1])
        p5  = self.lat5(feats[2])
        out = [p3, p4, p5]
        if self.num_levels >= 4:
            out.append(self.p6_down(p5))
        for layer in self.layers:
            out = layer(out)
        return out

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBN, DWConv


class DetectHead(nn.Module):
    """
    YOLOv8-style DFL head.
    cls branch  → nc logits per anchor
    reg branch  → 4 × (reg_max+1) DFL logits per anchor
    Score at inference = sigmoid(cls) * iou  (no centerness)
    """
    def __init__(self, ic, nc, reg_max=16, num_convs=4, num_levels=4):
        super().__init__()
        self.nc      = nc
        self.reg_max = reg_max
        self.proj    = nn.Parameter(
            torch.arange(reg_max + 1, dtype=torch.float32), requires_grad=False)

        ct, rt = [], []
        for _ in range(num_convs):
            ct.append(DWConv(ic, ic))
            rt.append(DWConv(ic, ic))
        self.cls_tower = nn.Sequential(*ct)
        self.reg_tower = nn.Sequential(*rt)
        self.cls_pred  = nn.Conv2d(ic, nc, 1)
        self.reg_pred  = nn.Conv2d(ic, 4 * (reg_max + 1), 1)

        nn.init.constant_(self.cls_pred.bias, -math.log(99.0))
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.zeros_(self.reg_pred.bias)

    def forward(self, feats):
        out = []
        for f in feats:
            cf = self.cls_tower(f)
            rf = self.reg_tower(f)
            out.append({"cls": self.cls_pred(cf), "reg": self.reg_pred(rf)})
        return out

    def decode_reg(self, reg_logits, anchor_pts, stride):
        """DFL decode: reg_logits [N, 4*(reg_max+1)] → xyxy boxes in pixel coords."""
        rm1  = self.reg_max + 1
        N    = reg_logits.shape[0]
        dist = reg_logits.view(N, 4, rm1)
        dist = F.softmax(dist, dim=2)
        dist = (dist * self.proj.to(dist.device)).sum(2) * stride
        x1   = anchor_pts[:, 0] - dist[:, 0]
        y1   = anchor_pts[:, 1] - dist[:, 1]
        x2   = anchor_pts[:, 0] + dist[:, 2]
        y2   = anchor_pts[:, 1] + dist[:, 3]
        return torch.stack([x1, y1, x2, y2], 1)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBN(in_ch, out_ch, k=1 if d == 1 else 3, d=d) for d in dilations])
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBN(in_ch, out_ch, 1))
        self.project = ConvBN((len(dilations) + 1) * out_ch, out_ch, 1)

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        outs.append(F.interpolate(self.gap(x), size=x.shape[2:],
                                  mode="bilinear", align_corners=False))
        return self.project(torch.cat(outs, 1))


class FPNSegHead(nn.Module):
    def __init__(self, ic, mask_dim=128, nc=7, hidden=256):
        super().__init__()
        self.fuse_p4 = ConvBN(ic, ic, 1)
        self.fuse_p5 = ConvBN(ic, ic, 1)
        self.fuse_p6 = ConvBN(ic, ic, 1)
        self.fuse_cv = ConvBN(ic * 4, hidden, 1)
        self.aspp    = ASPP(hidden, hidden)
        self.proto_net = nn.Sequential(
            ConvBN(hidden, hidden, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBN(hidden, hidden, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBN(hidden, mask_dim, 3))
        self.sem_net = nn.Sequential(
            ConvBN(hidden, hidden, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBN(hidden, hidden // 2, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBN(hidden // 2, hidden // 2, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden // 2, nc, 1))

    def forward(self, feats):
        p3, p4, p5, p6 = feats[:4]
        p4u = F.interpolate(self.fuse_p4(p4), size=p3.shape[2:],
                            mode="bilinear", align_corners=False)
        p5u = F.interpolate(self.fuse_p5(p5), size=p3.shape[2:],
                            mode="bilinear", align_corners=False)
        p6u = F.interpolate(self.fuse_p6(p6), size=p3.shape[2:],
                            mode="bilinear", align_corners=False)
        fused = self.aspp(self.fuse_cv(torch.cat([p3, p4u, p5u, p6u], 1)))
        return {"prototypes": self.proto_net(fused), "semantic": self.sem_net(fused)}

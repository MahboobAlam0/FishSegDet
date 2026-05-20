import math

import torch.nn as nn

from .backbone import make_backbone
from .neck import BiFPN
from .heads import DetectHead, FPNSegHead


class FishSegDet(nn.Module):
    def __init__(self, num_classes=6, pretrained=True, neck_ch=256,
                 bifpn_repeats=3, mask_dim=128, num_fpn_levels=4,
                 backbone_variant="convnextv2_large.fcmae_ft_in22k_in1k_384",
                 reg_max=16):
        super().__init__()
        self.num_classes    = num_classes
        self.num_fpn_levels = num_fpn_levels
        self.reg_max        = reg_max
        self.backbone = make_backbone(backbone_variant, pretrained)
        self.neck     = BiFPN(self.backbone.out_channels, neck_ch, bifpn_repeats, num_fpn_levels)
        self.det_head = DetectHead(neck_ch, num_classes, reg_max,
                                   num_convs=4, num_levels=num_fpn_levels)
        self.seg_head = FPNSegHead(neck_ch, mask_dim, num_classes + 1, hidden=256)

        for mod in [self.neck, self.det_head, self.seg_head]:
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.det_head.cls_pred.bias, -math.log(99.0))

    def forward(self, x):
        nf = self.neck(self.backbone(x))
        so = self.seg_head(nf)
        return {"det_outputs": self.det_head(nf),
                "semantic":    so["semantic"],
                "prototypes":  so["prototypes"]}

    def count_parameters(self):
        parts  = {"backbone": self.backbone, "neck": self.neck,
                  "det_head": self.det_head, "seg_head": self.seg_head}
        counts = {k: sum(x.numel() for x in v.parameters()) for k, v in parts.items()}
        counts["total"] = sum(counts.values())
        return counts


def build_model(num_classes=6, pretrained=True, neck_ch=256, bifpn_repeats=3,
                mask_dim=128, num_fpn_levels=4,
                backbone_variant="convnextv2_large.fcmae_ft_in22k_in1k_384",
                reg_max=16) -> FishSegDet:
    return FishSegDet(num_classes, pretrained, neck_ch, bifpn_repeats,
                      mask_dim, num_fpn_levels, backbone_variant, reg_max)

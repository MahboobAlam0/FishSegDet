import torch.nn as nn

_SWIN_CFGS = {
    "swin_tiny":  ("swin_t", "Swin_T_Weights", [192, 384, 768]),
    "swin_small": ("swin_s", "Swin_S_Weights", [192, 384, 768]),
    "swin_base":  ("swin_b", "Swin_B_Weights", [256, 512, 1024]),
}


class SwinBackbone(nn.Module):
    def __init__(self, variant="swin_tiny", pretrained=True):
        super().__init__()
        fn_name, wcls, out_ch = _SWIN_CFGS[variant]
        import torchvision.models as tvm
        fn = getattr(tvm, fn_name)
        w = getattr(tvm, wcls).IMAGENET1K_V1 if pretrained else None
        feats = fn(weights=w).features
        self.patch_embed = feats[0]
        self.stage1 = feats[1]
        self.merge2  = feats[2]
        self.stage2  = feats[3]
        self.merge3  = feats[4]
        self.stage3  = feats[5]
        self.merge4  = feats[6]
        self.stage4  = feats[7]
        self.norm_p3 = nn.LayerNorm(out_ch[0])
        self.norm_p4 = nn.LayerNorm(out_ch[1])
        self.norm_p5 = nn.LayerNorm(out_ch[2])
        self.out_channels = out_ch

    def forward(self, x):
        x  = self.patch_embed(x)
        x  = self.stage1(x)
        x  = self.merge2(x)
        x  = self.stage2(x)
        p3 = self.norm_p3(x).permute(0, 3, 1, 2).contiguous()
        x  = self.merge3(x)
        x  = self.stage3(x)
        p4 = self.norm_p4(x).permute(0, 3, 1, 2).contiguous()
        x  = self.merge4(x)
        x  = self.stage4(x)
        p5 = self.norm_p5(x).permute(0, 3, 1, 2).contiguous()
        return [p3, p4, p5]


class TimmBackbone(nn.Module):
    def __init__(self, model_name="convnextv2_large.fcmae_ft_in22k_in1k_384",
                 pretrained=True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("pip install timm")
        self.model = timm.create_model(
            model_name, pretrained=pretrained,
            features_only=True, out_indices=(1, 2, 3))
        all_ch = self.model.feature_info.channels()
        n = len(all_ch)
        self.out_channels = [all_ch[n - 3], all_ch[n - 2], all_ch[n - 1]]

    def forward(self, x):
        return list(self.model(x))


def make_backbone(variant: str, pretrained: bool):
    if variant in _SWIN_CFGS:
        return SwinBackbone(variant, pretrained)
    return TimmBackbone(variant, pretrained)

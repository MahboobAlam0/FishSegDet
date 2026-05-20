import torch.nn as nn


class TimmBackbone(nn.Module):
    """
    Feature extractor built on any timm model.
    Default: ConvNeXtV2-Large pre-trained on IN-22K → IN-1K @ 384px.
    Returns the last three feature maps [C3, C4, C5].
    """

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


def make_backbone(model_name: str, pretrained: bool) -> TimmBackbone:
    return TimmBackbone(model_name, pretrained)

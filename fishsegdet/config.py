from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Config:
    dataset_root: str = "./USISDataset"
    classes: List[str] = field(default_factory=lambda: ["object"])  # overwritten by configure_dataset()
    num_classes: int = 1
    img_size: int = 640
    # Backbone
    backbone: str = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    neck_channels: int = 256
    bifpn_repeats: int = 3
    mask_dim: int = 128
    num_fpn_levels: int = 4
    pretrained: bool = True
    # DFL
    reg_max: int = 16
    # Training
    batch_size: int = 16
    num_workers: int = 4
    phase1_epochs: int = 60
    phase2_epochs: int = 40
    phase3_epochs: int = 40
    lr_backbone: float = 2e-5
    lr_head: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    # TAL
    fcos_strides: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    tal_topk: int = 13
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
    # Loss weights
    loss_cls: float = 1.0
    loss_box: float = 3.0
    loss_dfl: float = 0.5
    loss_seg: float = 1.5
    # Inference
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    # Preprocessing
    use_clahe: bool = True
    use_white_balance: bool = True
    use_gamma: bool = True
    gamma_value: float = 1.2
    clahe_clip: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./runs"


CFG = Config()

USIS_CLASSES: Dict[int, str] = {i: n for i, n in enumerate(CFG.classes)}
CLASS_TO_IDX: Dict[str, int] = {n: i for i, n in USIS_CLASSES.items()}

np.random.seed(42)
COLORS: List[Tuple[int, int, int]] = [
    tuple(int(x) for x in np.random.randint(50, 230, 3))
    for _ in range(max(CFG.num_classes, 20))
]

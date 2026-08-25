"""Augmentation pipelines: mild (Phase 1), heavy (Phase 2/3), and val (deterministic).

Each returned Compose is tagged `.is_heavy` -- MosaicDataset checks this flag
to decide whether to build a 4-image mosaic for a given sample, tying mosaic
to Phase 2/3 without needing changes to train.py's phase-switch logic.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_BBOX_PARAMS = A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_visibility=0.1,
    clip=True,
)


def get_train_transforms(img_size: int = 640) -> A.Compose:
    t = A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0, fill=(114, 114, 114)),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10),
                     border_mode=0, fill=(114, 114, 114), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=_BBOX_PARAMS,
    )
    t.is_heavy = False
    return t


def get_heavy_transforms(img_size: int = 640) -> A.Compose:
    t = A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0, fill=(114, 114, 114)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-15, 15),
                     border_mode=0, fill=(114, 114, 114), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.5),
            A.RandomFog(fog_coef_range=(0.1, 0.35), alpha_coef=0.08, p=0.25),
            A.MotionBlur(blur_limit=(3, 9), p=0.25),
            A.CoarseDropout(num_holes_range=(1, 8),
                             hole_height_range=(0.02, 0.08), hole_width_range=(0.02, 0.08),
                             fill=114, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=_BBOX_PARAMS,
    )
    t.is_heavy = True
    return t


def get_val_transforms(img_size: int = 640) -> A.Compose:
    t = A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0, fill=(114, 114, 114)),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=_BBOX_PARAMS,
    )
    t.is_heavy = False
    return t

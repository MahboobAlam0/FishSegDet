"""configure_dataset(): auto-discovers classes from the COCO annotation file.
create_dataloaders(): builds train/val/test DataLoaders with class-balanced sampling.
"""

from __future__ import annotations

import json
import os

from torch.utils.data import DataLoader, WeightedRandomSampler

from ..config import CFG, USIS_CLASSES, CLASS_TO_IDX
from .augmentations import get_train_transforms, get_val_transforms
from .datasets import USISDataset, MosaicDataset

_configured_roots = set()


def configure_dataset(root: str, split: str = "train") -> None:
    """Reads annotations/instances_{split}.json, updates CFG/USIS_CLASSES/
    CLASS_TO_IDX in place (not reassigned, so other modules' already-imported
    references stay in sync). Idempotent per root."""
    if root in _configured_roots:
        return

    ann_path = os.path.join(root, "annotations", f"instances_{split}.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"Annotation file not found: {ann_path}\n"
            f"Expected layout: {root}/annotations/instances_{{train,val,test}}.json"
        )

    with open(ann_path, "r") as f:
        coco_json = json.load(f)

    categories = sorted(coco_json.get("categories", []), key=lambda c: c["id"])
    class_names = [c["name"] for c in categories]
    if not class_names:
        raise ValueError(f"No categories found in {ann_path}")

    CFG.classes = class_names
    CFG.num_classes = len(class_names)

    USIS_CLASSES.clear()
    USIS_CLASSES.update({i: n for i, n in enumerate(class_names)})
    CLASS_TO_IDX.clear()
    CLASS_TO_IDX.update({n: i for i, n in USIS_CLASSES.items()})

    _configured_roots.add(root)
    print(f"[configure_dataset] {len(class_names)} classes discovered from {ann_path}")


def create_dataloaders(root: str, img_size: int = 640, batch_size: int = 16,
                        num_workers: int = 4, mosaic_prob: float = 0.5):
    """Returns (train_loader, val_loader, test_loader); test_loader is None
    if instances_test.json isn't present."""
    configure_dataset(root, split="train")

    train_ds = MosaicDataset(root, "train", img_size,
                              transform=get_train_transforms(img_size), mosaic_prob=mosaic_prob)
    val_ds = USISDataset(root, "val", img_size, transform=get_val_transforms(img_size))

    test_dir_exists = os.path.exists(os.path.join(root, "annotations", "instances_test.json"))
    test_ds = USISDataset(root, "test", img_size, transform=get_val_transforms(img_size)) \
        if test_dir_exists else None

    # class-balanced sampling: weight = 1/sqrt(freq) of the rarest class in each image
    sample_weights = train_ds.compute_sample_weights()
    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler, drop_last=True,
        num_workers=num_workers, pin_memory=True, collate_fn=USISDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True, collate_fn=USISDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True, collate_fn=USISDataset.collate_fn,
    ) if test_ds is not None else None

    return train_loader, val_loader, test_loader

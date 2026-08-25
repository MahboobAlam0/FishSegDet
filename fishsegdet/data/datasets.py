"""USISDataset / MosaicDataset -- COCO-format dataset for USIS16K.

ds[i]          -> {"image": FloatTensor[3,H,W], "bboxes": FloatTensor[M,4] (yolo,
                   normalised cx,cy,w,h), "labels": LongTensor[M], "mask": LongTensor[H,W]}
ds.get_raw(i)  -> (img_bgr uint8 HxWx3 UNPROCESSED, img_path, bboxes, labels, mask HxW int)

get_raw() returns the untouched image so FishSegDetInference (which applies
UnderwaterPreprocessor itself) doesn't double-apply WB/CLAHE/gamma.
"""

from __future__ import annotations

import os
import random
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import UnderwaterPreprocessor


class USISDataset(Dataset):
    def __init__(self, root: str, split: str, img_size: int = 640,
                 transform=None, use_preprocessing: bool = True):
        from pycocotools.coco import COCO

        self.root = root
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.preprocessor = UnderwaterPreprocessor() if use_preprocessing else None

        ann_path = os.path.join(root, "annotations", f"instances_{split}.json")
        self.coco = COCO(ann_path)
        self.img_dir = os.path.join(root, split)

        cat_ids = sorted(self.coco.getCatIds())
        self.catid_to_label = {cid: i for i, cid in enumerate(cat_ids)}
        self.label_to_catid = {i: cid for cid, i in self.catid_to_label.items()}
        self.img_ids = sorted(self.coco.getImgIds())

    def __len__(self) -> int:
        return len(self.img_ids)

    def _load_raw(self, idx: int):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        h, w = img.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        # largest instances painted first so small objects aren't erased
        anns = sorted(anns, key=lambda a: a.get("area", 0), reverse=True)

        mask = np.zeros((h, w), dtype=np.uint8)
        bboxes, labels = [], []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in self.catid_to_label:
                continue
            label = self.catid_to_label[cat_id]

            x, y, bw, bh = ann["bbox"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nbw, nbh = bw / w, bh / h
            if nbw <= 0 or nbh <= 0:
                continue
            bboxes.append([cx, cy, nbw, nbh])
            labels.append(label)

            if ann.get("segmentation"):
                try:
                    inst_mask = self.coco.annToMask(ann)
                    if inst_mask.shape != (h, w):
                        # annotation file's recorded width/height didn't match
                        # the actual decoded image -- resize to correct it
                        inst_mask = cv2.resize(inst_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask[inst_mask > 0] = label + 1  # 0 reserved for background
                except Exception:
                    pass  # malformed polygon -- fall back to bbox-only supervision

        return img, img_path, bboxes, labels, mask

    def get_raw(self, idx: int):
        return self._load_raw(idx)

    def compute_class_freqs(self) -> dict:
        """Instance count per class, over the whole split's annotations."""
        freqs = {label: 0 for label in self.catid_to_label.values()}
        for ann in self.coco.dataset.get("annotations", []):
            label = self.catid_to_label.get(ann["category_id"])
            if label is not None:
                freqs[label] += 1
        return freqs

    def compute_sample_weights(self) -> List[float]:
        """Per-sample weight = 1/sqrt(freq) of the rarest class in the image."""
        freqs = self.compute_class_freqs()
        weights = []
        for img_id in self.img_ids:
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            labels_in_img = [self.catid_to_label[a["category_id"]]
                              for a in anns if a["category_id"] in self.catid_to_label]
            if not labels_in_img:
                weights.append(1.0)
                continue
            rarest_freq = min(freqs[l] for l in labels_in_img)
            weights.append(1.0 / (rarest_freq ** 0.5) if rarest_freq > 0 else 1.0)
        return weights

    def _to_item(self, img_rgb, mask, bboxes, labels):
        if self.transform is not None:
            out = self.transform(image=img_rgb, mask=mask, bboxes=bboxes, class_labels=labels)
            img_t, mask_t = out["image"], out["mask"]
            bboxes, labels = out["bboxes"], out["class_labels"]
        else:
            img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
            mask_t = torch.from_numpy(mask)

        if not torch.is_tensor(mask_t):
            mask_t = torch.as_tensor(np.array(mask_t))
        mask_t = mask_t.long()

        if len(bboxes) > 0:
            bboxes_t = torch.tensor(bboxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            bboxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        return {"image": img_t, "bboxes": bboxes_t, "labels": labels_t, "mask": mask_t}

    def __getitem__(self, idx: int):
        img, _, bboxes, labels, mask = self._load_raw(idx)
        if self.preprocessor is not None:
            img = self.preprocessor(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._to_item(img_rgb, mask, bboxes, labels)

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """Pads bboxes/labels to the batch's max object count; produces num_objects."""
        images = torch.stack([b["image"] for b in batch], 0)
        masks = torch.stack([b["mask"] for b in batch], 0)
        num_objects = torch.tensor([b["bboxes"].shape[0] for b in batch], dtype=torch.long)
        max_m = max(int(num_objects.max().item()), 1)

        B = len(batch)
        bboxes = torch.zeros((B, max_m, 4), dtype=torch.float32)
        labels = torch.zeros((B, max_m), dtype=torch.long)
        for i, b in enumerate(batch):
            m = b["bboxes"].shape[0]
            if m > 0:
                bboxes[i, :m] = b["bboxes"]
                labels[i, :m] = b["labels"]

        return {"image": images, "bboxes": bboxes, "labels": labels,
                "num_objects": num_objects, "mask": masks}


class MosaicDataset(USISDataset):
    """USISDataset + 4-image mosaic. Mosaic activates only when the current
    transform is heavy (`get_heavy_transforms`, `.is_heavy = True`), tying
    mosaic to Phase 2/3 without changes to train.py."""

    def __init__(self, *args, mosaic_prob: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.mosaic_prob = mosaic_prob

    def _mosaic_raw(self, idx: int):
        """2x2 mosaic at 2*img_size, then random-cropped back to img_size."""
        s = self.img_size
        idxs = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        canvas = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
        mask_canvas = np.zeros((2 * s, 2 * s), dtype=np.uint8)
        all_boxes, all_labels = [], []
        offsets = [(0, 0), (s, 0), (0, s), (s, s)]

        for (ox, oy), i in zip(offsets, idxs):
            img, _, boxes, labels, mask = self._load_raw(i)
            h, w = img.shape[:2]
            scale = s / max(h, w)
            nw, nh = int(w * scale), int(h * scale)
            img_r = cv2.resize(img, (nw, nh))
            mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            canvas[oy:oy + nh, ox:ox + nw] = img_r
            mask_canvas[oy:oy + nh, ox:ox + nw] = mask_r

            for (cx, cy, bw, bh), lb in zip(boxes, labels):
                all_boxes.append([
                    (cx * nw + ox) / (2 * s), (cy * nh + oy) / (2 * s),
                    (bw * nw) / (2 * s), (bh * nh) / (2 * s),
                ])
                all_labels.append(lb)

        cx0 = random.randint(0, s)
        cy0 = random.randint(0, s)
        img_crop = canvas[cy0:cy0 + s, cx0:cx0 + s]
        mask_crop = mask_canvas[cy0:cy0 + s, cx0:cx0 + s]

        final_boxes, final_labels = [], []
        for (cx, cy, bw, bh), lb in zip(all_boxes, all_labels):
            acx, acy = cx * 2 * s - cx0, cy * 2 * s - cy0
            abw, abh = bw * 2 * s, bh * 2 * s
            x1, y1 = max(0, acx - abw / 2), max(0, acy - abh / 2)
            x2, y2 = min(s, acx + abw / 2), min(s, acy + abh / 2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            final_boxes.append([(x1 + x2) / 2 / s, (y1 + y2) / 2 / s, (x2 - x1) / s, (y2 - y1) / s])
            final_labels.append(lb)

        return img_crop, mask_crop, final_boxes, final_labels

    def __getitem__(self, idx: int):
        use_mosaic = getattr(self.transform, "is_heavy", False) and random.random() < self.mosaic_prob
        if not use_mosaic:
            return super().__getitem__(idx)

        img, mask, bboxes, labels = self._mosaic_raw(idx)
        if self.preprocessor is not None:
            img = self.preprocessor(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._to_item(img_rgb, mask, bboxes, labels)

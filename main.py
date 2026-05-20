"""
FishSegDet v4 — main.py
========================
Changes vs v3:
  DETECT :  TAL + DFL + IoU-aware cls (no centerness)
  LOSS   :  TAL-VFL + DFL + IoU-aware cls
  DATA   : class-balanced sampler, small/degenerate box filter
  AUG    : lighter Phase-1 augmentations — only geometry + mild colour
  MISC   : anchor cache keyed on device string, cleaner forward

train.py imports from this file. This file NEVER imports from itself.
"""

import time
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.ops import nms

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    A = None  # type: ignore
    ToTensorV2 = None  # type: ignore
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed.")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    dataset_root: str = "./USISDataset"
    classes: List[str] = field(default_factory=lambda: ["fish"])
    num_classes: int = 1
    img_size: int = 640
    # Backbone
    backbone: str = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    neck_channels: int = 256
    bifpn_repeats: int = 3
    mask_dim: int = 128
    num_fpn_levels: int = 4
    pretrained: bool = True
    # DFL bins
    reg_max: int = 16          # DFL: 0..reg_max → (reg_max+1) bins per side
    # Training schedule — 2 meaningful phases
    batch_size: int = 16
    num_workers: int = 4
    phase1_epochs: int = 60   # full model, differential LR, light aug
    phase2_epochs: int = 40    # hard augmentation
    phase3_epochs: int = 40     # reserved
    lr_backbone: float = 2e-5
    lr_head: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    # TAL parameters
    fcos_strides: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    tal_topk: int = 13         # max positive anchors per GT
    tal_alpha: float = 0.5     # cls weight in alignment metric
    tal_beta: float = 6.0      # iou weight in alignment metric
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
BRACKISH_CLASSES: Dict[int, str] = {i: n for i, n in enumerate(CFG.classes)}
CLASS_TO_IDX: Dict[str, int]     = {n: i for i, n in BRACKISH_CLASSES.items()}
np.random.seed(42)
COLORS = [tuple(int(x) for x in np.random.randint(50, 230, 3))
          for _ in range(max(CFG.num_classes, 20))]


# ============================================================================
# PREPROCESSING
# ============================================================================

class UnderwaterPreprocessor:
    def __init__(self, use_white_balance=True, use_clahe=True,
                 use_gamma=True, clahe_clip=2.0, clahe_grid=8, gamma=1.2):
        self.uwb = use_white_balance
        self.ucl = use_clahe
        self.ugm = use_gamma
        if use_clahe:
            self.clahe=cv2.createCLAHE(clipLimit=clahe_clip,
                                        tileGridSize=(clahe_grid,clahe_grid))
        if use_gamma:
            self.lut=np.array([((i/255.)**(1./gamma))*255
                                for i in range(256)],dtype=np.uint8)

    def __call__(self, img):
        if self.uwb:
            img = self._wb(img)
        if self.ucl:
            img = self._clahe(img)
        if self.ugm:
            img = cv2.LUT(img, self.lut)
        return img

    def _wb(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        aa, ab = lab[:, :, 1].mean(), lab[:, :, 2].mean()
        ln = lab[:, :, 0] / 255.
        lab[:, :, 1] = np.clip(lab[:, :, 1] - (aa - 128) * ln * 1.1, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] - (ab - 128) * ln * 1.1, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================================
# AUGMENTATIONS
# ============================================================================

def get_train_transforms(img_size=640):
    """
    Phase-1 augmentations — LIGHT.
    Only mild geometry + mild colour. No fog, no coarse dropout, no heavy RGB shift.
    Aggressive augmentation in Phase 1 destroys the weak signal from rare classes.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT,
                      fill=(114, 114, 114)),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1),
                 rotate=0, fill=(114, 114, 114), p=0.4),
        A.RandomBrightnessContrast((-0.2, 0.1), (-0.1, 0.2), p=0.5),
        A.HueSaturationValue(10, 15, 15, p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize((.485, .456, .406), (.229, .224, .225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams("yolo", label_fields=["class_labels"],
                                min_visibility=0.3))


def get_val_transforms(img_size=640):
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT,
                      fill=(114, 114, 114)),
        A.Normalize((.485, .456, .406), (.229, .224, .225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams("yolo", label_fields=["class_labels"],
                                min_visibility=0.3))


def get_heavy_transforms(img_size=640):
    """
    Phase-2 augmentations — full heavy set.
    Only applied AFTER the model has converged on Phase 1.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT,
                      fill=(114, 114, 114)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(.8, 1.2),
                 rotate=(-15, 15), fill=(114, 114, 114), p=0.6),
        A.OneOf([
            A.RGBShift(r_shift_limit=(-30, -5), g_shift_limit=(-10, 5),
                       b_shift_limit=(5, 20), p=1.),
            A.ColorJitter(.2, .2, .2, .05, p=1.),
        ], p=0.6),
        A.RandomBrightnessContrast((-0.4, .1), (-.2, .3), p=0.6),
        A.HueSaturationValue(15, 25, 25, p=0.5),
        A.OneOf([A.RandomFog(p=1.), A.GaussNoise(p=1.)], p=0.3),
        A.OneOf([A.MotionBlur((3, 7), p=1.), A.GaussianBlur((3, 5), p=1.)], p=0.2),
        A.CoarseDropout(num_holes_range=(4, 8),
                        hole_height_range=(img_size//32, img_size//16),
                        hole_width_range=(img_size//32, img_size//16),
                        fill=114, p=0.2),
        A.Normalize((.485, .456, .406), (.229, .224, .225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams("yolo", label_fields=["class_labels"],
                                min_visibility=0.3))


# ============================================================================
# DATASETS
# ============================================================================

def _filter_boxes(bboxes, labels, img_size, min_px=4, max_ar=20.):
    """
    Remove degenerate boxes:
      - absolute size < min_px pixels after resize
      - aspect ratio > max_ar
    Returns filtered (bboxes, labels).
    """
    if len(bboxes) == 0:
        return bboxes, labels
    keep = []
    for i,(b,l) in enumerate(zip(bboxes,labels)):
        cx,cy,bw,bh = b
        pw = bw * img_size
        ph = bh * img_size
        if pw < min_px or ph < min_px:
            continue
        ar = max(pw,ph) / (min(pw,ph) + 1e-6)
        if ar > max_ar:
            continue
        keep.append(i)
    if not keep:
        return np.zeros((0,4),np.float32), np.zeros((0,),np.int64)
    return bboxes[keep], labels[keep]


def _pad_boxes(bboxes, labels, max_boxes=100):
    pb = np.zeros((max_boxes,4), np.float32)
    pl = np.zeros((max_boxes,),  np.int64)
    n  = min(len(bboxes), max_boxes)
    if n > 0:
        pb[:n] = bboxes[:n]
        pl[:n] = labels[:n]
    return pb, pl, n


class BrackishDataset(Dataset):
    def __init__(self, root, split="train", img_size=640, transform=None):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.img_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.mask_dir = self.root / "masks" / split
        self.images = sorted(list(self.img_dir.glob("*.jpg")) +
                           list(self.img_dir.glob("*.png")))
        print(f"[{split.upper()}] {len(self.images)} images")

    def __len__(self): return len(self.images)

    def _load_labels(self, ip, w, h):
        lp = self.label_dir / f"{ip.stem}.txt"
        if not lp.exists():
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        bboxes, labels = [], []
        for line in lp.read_text().splitlines():
            p = line.strip().split()
            if len(p) < 5:
                continue
            labels.append(int(p[0]))
            b = [float(x) for x in p[1:5]]
            for i, dim in enumerate([w, h, w, h]):
                b[i] = b[i] / dim if b[i] > 1 else b[i]
            b[2] = min(b[2], 1.)
            b[3] = min(b[3], 1.)
            if b[0] - b[2] / 2 < 0:
                b[0] = b[2] / 2
            if b[0] + b[2] / 2 > 1:
                b[0] = 1 - b[2] / 2
            if b[1] - b[3] / 2 < 0:
                b[1] = b[3] / 2
            if b[1] + b[3] / 2 > 1:
                b[1] = 1 - b[3] / 2
            bboxes.append(b)
        if bboxes:
            return np.array(bboxes, np.float32), np.array(labels, np.int64)
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

    def _load_mask(self, ip, bboxes, labels, shape):
        mp = self.mask_dir / f"{ip.stem}.png"
        if mp.exists():
            return cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        h, w = shape
        mask = np.zeros((h, w), np.uint8)
        for bbox, lbl in zip(bboxes, labels):
            cx, cy, bw, bh = bbox
            x1 = max(0, int((cx - bw / 2) * w))
            y1 = max(0, int((cy - bh / 2) * h))
            x2 = min(w, int((cx + bw / 2) * w))
            y2 = min(h, int((cy + bh / 2) * h))
            mask[y1:y2, x1:x2] = lbl + 1
        return mask

    def get_raw(self, idx):
        ip = self.images[idx]
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        bboxes, labels = self._load_labels(ip, w, h)
        bboxes, labels = _filter_boxes(bboxes, labels, self.img_size)
        return img, str(ip), bboxes, labels, self._load_mask(ip, bboxes, labels, (h, w))

    def __getitem__(self, idx):
        img,ip,bboxes,labels,mask=self.get_raw(idx)
        if self.transform:
            t=self.transform(image=img, mask=mask,
                             bboxes=bboxes.tolist() if len(bboxes) else [],
                             class_labels=labels.tolist() if len(labels) else [])
            img=t["image"]; mask=t["mask"]
            bboxes=np.array(t["bboxes"],np.float32) if t["bboxes"] else np.zeros((0,4),np.float32)
            labels=np.array(t["class_labels"],np.int64) if t["class_labels"] else np.zeros((0,),np.int64)
        else:
            img=torch.from_numpy(img.transpose(2,0,1)).float()/255.
            mask=torch.from_numpy(mask).long()
        if not isinstance(mask,torch.Tensor): mask=torch.from_numpy(mask).long()
        pb,pl,n=_pad_boxes(bboxes,labels)
        return {"image":img,"mask":mask,"bboxes":torch.from_numpy(pb),
                "labels":torch.from_numpy(pl),"num_objects":torch.tensor(n),
                "img_path":ip}


class USISDataset(Dataset):
    def __init__(self, root, split="train", img_size=640, transform=None):
        from pycocotools.coco import COCO
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.img_dir = self.root / split
        ann = self.root / "annotations" / f"instances_{split}.json"
        self.coco = COCO(str(ann))
        self.img_ids = sorted(self.coco.getImgIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = [c["name"] for c in cats]
        self.cat2idx = {c["id"]: i for i, c in enumerate(cats)}
        print(f"[{split.upper()}] {len(self.img_ids)} images | {len(self.classes)} classes")

    def __len__(self): return len(self.img_ids)

    def get_raw(self, idx):
        info=self.coco.loadImgs(self.img_ids[idx])[0]
        ip=self.img_dir/info["file_name"]
        img=cv2.cvtColor(cv2.imread(str(ip)),cv2.COLOR_BGR2RGB)
        h,w=img.shape[:2]
        anns=self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.img_ids[idx]))
        bboxes,labels=[],[]
        mask=np.zeros((h,w),np.uint8)
        for ann in anns:
        if "bbox" not in ann or "category_id" not in ann:
            continue
            ci=self.cat2idx[ann["category_id"]]
            x,y,bw,bh=ann["bbox"]
            x1 = max(0., min(float(w), x))
            y1 = max(0., min(float(h), y))
            x2 = max(0., min(float(w), x + bw))
            y2 = max(0., min(float(h), y + bh))
            cbw = x2 - x1
            cbh = y2 - y1
            if cbw <= 0 or cbh <= 0:
                continue
            cx = (x1 + cbw / 2) / w
            cy = (y1 + cbh / 2) / h
            nw = cbw / w
            nh = cbh / h
            cx = max(1e-6, min(1 - 1e-6, cx))
            cy = max(1e-6, min(1 - 1e-6, cy))
            nw = max(1e-6, min(1 - 1e-6, nw))
            nh = max(1e-6, min(1 - 1e-6, nh))
            if cx + nw / 2 > 1:
                nw = (1 - cx) * 2 - 1e-6
            if cx - nw / 2 < 0:
                nw = cx * 2 - 1e-6
            if cy + nh / 2 > 1:
                nh = (1 - cy) * 2 - 1e-6
            if cy - nh / 2 < 0:
                nh = cy * 2 - 1e-6
            bboxes.append([cx, cy, nw, nh])
            labels.append(ci)
            if ann.get("segmentation"):
                m = self.coco.annToMask(ann)
                mask[m > 0] = ci + 1
        ba=np.array(bboxes,np.float32) if bboxes else np.zeros((0,4),np.float32)
        la=np.array(labels,np.int64)   if labels else np.zeros((0,), np.int64)
        ba,la=_filter_boxes(ba,la,self.img_size)
        return img,str(ip),ba,la,mask

    def __getitem__(self, idx):
        img,ip,bboxes,labels,mask=self.get_raw(idx)
        if self.transform:
            t=self.transform(image=img, mask=mask,
                             bboxes=bboxes.tolist() if len(bboxes) else [],
                             class_labels=labels.tolist() if len(labels) else [])
            img=t["image"]; mask=t["mask"]
            bboxes=np.array(t["bboxes"],np.float32) if t["bboxes"] else np.zeros((0,4),np.float32)
            labels=np.array(t["class_labels"],np.int64) if t["class_labels"] else np.zeros((0,),np.int64)
        else:
            img=torch.from_numpy(img.transpose(2,0,1)).float()/255.
            mask=torch.from_numpy(mask).long()
        if not isinstance(mask,torch.Tensor): mask=torch.from_numpy(mask).long()
        pb,pl,n=_pad_boxes(bboxes,labels)
        return {"image":img,"mask":mask,"bboxes":torch.from_numpy(pb),
                "labels":torch.from_numpy(pl),"num_objects":torch.tensor(n),
                "img_path":ip}


class MosaicDataset(Dataset):
    def __init__(self, base_ds, mosaic_prob=0.5):
        self.ds = base_ds
        self.mosaic_prob = mosaic_prob
        self.img_size=base_ds.img_size

    @property
    def transform(self): return self.ds.transform
    @transform.setter
    def transform(self, t): self.ds.transform=t
    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        if random.random() < self.mosaic_prob:
            return self._mosaic(idx)
        return self.ds[idx]

    def _mosaic(self, idx):
        s = self.img_size
        h = s // 2
        idxs=[idx]+[random.randint(0,len(self.ds)-1) for _ in range(3)]
        slots=[(0,0,h,h),(h,0,s,h),(0,h,h,s),(h,h,s,s)]
        mimg=np.full((s,s,3),114,np.uint8)
        mmask=np.zeros((s,s),np.uint8)
        all_bb = []
        all_lb = []
        for i,ind in enumerate(idxs):
            img,_,bb,lb,mask=self.ds.get_raw(ind)
            oh,ow=img.shape[:2]
            x1, y1, x2, y2 = slots[i]
            pw, ph = x2 - x1, y2 - y1
            mimg[y1:y2,x1:x2]=cv2.resize(img,(pw,ph))
            mmask[y1:y2,x1:x2]=cv2.resize(mask,(pw,ph),
                                            interpolation=cv2.INTER_NEAREST)
            sx,sy=pw/ow,ph/oh
            for b,l in zip(bb,lb):
                cx,cy,bw,bh=b
                ax1=max(0.,x1+cx*ow*sx-bw*ow*sx/2)
                ay1=max(0.,y1+cy*oh*sy-bh*oh*sy/2)
                ax2=min(float(s),x1+cx*ow*sx+bw*ow*sx/2)
                ay2=min(float(s),y1+cy*oh*sy+bh*oh*sy/2)
                if ax2 - ax1 < 4 or ay2 - ay1 < 4:
                    continue
                all_bb.append([(ax1+ax2)/2/s,(ay1+ay2)/2/s,
                                (ax2-ax1)/s,(ay2-ay1)/s])
                all_lb.append(l)
        bboxes=np.array(all_bb,np.float32) if all_bb else np.zeros((0,4),np.float32)
        labels=np.array(all_lb,np.int64)   if all_lb else np.zeros((0,),np.int64)
        tf=self.ds.transform
        if tf:
            t=tf(image=mimg, mask=mmask,
                 bboxes=bboxes.tolist() if len(bboxes) else [],
                 class_labels=labels.tolist() if len(labels) else [])
            img_t=t["image"]; mask_t=t["mask"]
            bboxes=np.array(t["bboxes"],np.float32) if t["bboxes"] else np.zeros((0,4),np.float32)
            labels=np.array(t["class_labels"],np.int64) if t["class_labels"] else np.zeros((0,),np.int64)
        else:
            img_t=torch.from_numpy(mimg.transpose(2,0,1)).float()/255.
            mask_t=torch.from_numpy(mmask).long()
        if not isinstance(mask_t, torch.Tensor):
            mask_t = torch.from_numpy(mask_t).long()
        pb,pl,n=_pad_boxes(bboxes,labels)
        return {"image":img_t,"mask":mask_t,"bboxes":torch.from_numpy(pb),
                "labels":torch.from_numpy(pl),"num_objects":torch.tensor(n),
                "img_path":"mosaic"}


# ── class-balanced sampler ────────────────────────────────────────────────────

def _build_class_weights(dataset, num_classes):
    """
    Compute per-sample weight = 1/sqrt(class_freq) for the rarest class in
    each image.  This over-samples images containing rare classes.
    """
    print("[Sampler] Computing class frequencies for balanced sampling...")
    freq = np.zeros(num_classes, dtype=np.float64)
    for idx in range(len(dataset)):
        _,_,_,labels,_ = dataset.ds.get_raw(idx)
        for l in labels:
            freq[int(l)] += 1
    freq = np.maximum(freq, 1.)
    w_cls = 1. / np.sqrt(freq)
    sample_weights = []
    for idx in range(len(dataset)):
        _,_,_,labels,_ = dataset.ds.get_raw(idx)
        if len(labels) == 0:
            sample_weights.append(w_cls.min())
        else:
            sample_weights.append(max(w_cls[int(l)] for l in labels))
    return torch.tensor(sample_weights, dtype=torch.double)


def configure_dataset(root, batch_size=16):
    import json
    root_path=Path(root)
    usis_ann=root_path/"annotations"/"instances_train.json"
    if usis_ann.exists():
        with open(usis_ann) as f:
            ann_data = json.load(f)
        classes=[c["name"] for c in ann_data.get("categories",[])]
        if not classes:
            raise ValueError(f"No categories in {usis_ann}")
        CFG.classes = classes
        CFG.num_classes = len(classes)
        global BRACKISH_CLASSES, CLASS_TO_IDX, COLORS
        BRACKISH_CLASSES.clear()
        BRACKISH_CLASSES.update({i:n for i,n in enumerate(classes)})
        CLASS_TO_IDX.clear()
        CLASS_TO_IDX.update({n:i for i,n in BRACKISH_CLASSES.items()})
        np.random.seed(42)
        COLORS=[tuple(int(x) for x in np.random.randint(50,230,3))
                for _ in range(CFG.num_classes)]
        DatasetClass=USISDataset
        print(f"[Dataset] USIS — {CFG.num_classes} classes")
        safe_bs=min(batch_size,32)
        if safe_bs != batch_size:
            print(f"[Memory] batch {batch_size}→{safe_bs}")
        batch_size = safe_bs
        CFG.batch_size = batch_size
    else:
        DatasetClass=BrackishDataset
        print(f"[Dataset] Brackish — {CFG.num_classes} classes")
    return DatasetClass, batch_size


def create_dataloaders(root, img_size=640, batch_size=16, num_workers=4,
                       balanced_sampling=True):
    DatasetClass,batch_size=configure_dataset(root,batch_size)
    pin = torch.cuda.is_available()
    pf = 2 if num_workers > 0 else None
    kw=dict(num_workers=num_workers,pin_memory=pin,
            persistent_workers=num_workers>0,prefetch_factor=pf)

    train_base=DatasetClass(root,"train",img_size,get_train_transforms(img_size))
    train_ds=MosaicDataset(train_base,mosaic_prob=0.5)

    if balanced_sampling:
        print("[Sampler] Building class-balanced sampler (1/√freq)...")
        weights = _build_class_weights(train_ds, CFG.num_classes)
        sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(weights),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, drop_last=True, **kw)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, drop_last=True, **kw)

    val_loader = DataLoader(DatasetClass(root, "val", img_size, get_val_transforms(img_size)),
                            batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(DatasetClass(root, "test", img_size, get_val_transforms(img_size)),
                            batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ============================================================================
# MODEL BLOCKS
# ============================================================================

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


# ============================================================================
# BACKBONES
# ============================================================================

_SWIN_CFGS={
    "swin_tiny": ("swin_t","Swin_T_Weights",[192,384, 768]),
    "swin_small":("swin_s","Swin_S_Weights",[192,384, 768]),
    "swin_base": ("swin_b","Swin_B_Weights",[256,512,1024]),
}


class SwinBackbone(nn.Module):
    def __init__(self, variant="swin_tiny", pretrained=True):
        super().__init__()
        fn_name,wcls,out_ch=_SWIN_CFGS[variant]
        import torchvision.models as tvm
        fn=getattr(tvm,fn_name)
        w=getattr(tvm,wcls).IMAGENET1K_V1 if pretrained else None
        feats=fn(weights=w).features
        self.patch_embed = feats[0]
        self.stage1 = feats[1]
        self.merge2 = feats[2]
        self.stage2 = feats[3]
        self.merge3 = feats[4]
        self.stage3 = feats[5]
        self.merge4 = feats[6]
        self.stage4 = feats[7]
        self.norm_p3=nn.LayerNorm(out_ch[0])
        self.norm_p4=nn.LayerNorm(out_ch[1])
        self.norm_p5=nn.LayerNorm(out_ch[2])
        self.out_channels=out_ch
        print(f"[Backbone] Swin ch={out_ch} "
              f"params={sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def forward(self,x):
        x = self.patch_embed(x)
        x = self.stage1(x)
        x = self.merge2(x)
        x = self.stage2(x)
        p3=self.norm_p3(x).permute(0,3,1,2).contiguous()
        x = self.merge3(x)
        x = self.stage3(x)
        p4=self.norm_p4(x).permute(0,3,1,2).contiguous()
        x = self.merge4(x)
        x = self.stage4(x)
        p5=self.norm_p5(x).permute(0,3,1,2).contiguous()
        return [p3,p4,p5]


class TimmBackbone(nn.Module):
    def __init__(self, model_name="convnextv2_large.fcmae_ft_in22k_in1k_384",
                 pretrained=True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("pip install timm")
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       features_only=True, out_indices=(1, 2, 3))
        all_ch = self.model.feature_info.channels()
        n = len(all_ch)
        self.out_channels = [all_ch[n-3], all_ch[n-2], all_ch[n-1]]
        print(f"[Backbone] {model_name}  ch={self.out_channels}  "
              f"params={sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def forward(self, x):
        return list(self.model(x))


def _make_backbone(variant, pretrained):
    if variant in _SWIN_CFGS:
        return SwinBackbone(variant, pretrained)
    return TimmBackbone(variant,pretrained)


# ============================================================================
# NECK — BiFPN
# ============================================================================

class BiFPNNode(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.conv = DWConv(ch, ch)
        self.act = nn.ReLU()

    def forward(self, x1, x2):
        w = self.act(self.w) / (self.act(self.w).sum() + 1e-4)
        return self.conv(w[0] * x1 + w[1] * x2)


class BiFPNNode3(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3))
        self.conv = DWConv(ch, ch)
        self.act = nn.ReLU()

    def forward(self, x1, x2, x3):
        w = self.act(self.w) / (self.act(self.w).sum() + 1e-4)
        return self.conv(w[0] * x1 + w[1] * x2 + w[2] * x3)


class BiFPNLayer(nn.Module):
    def __init__(self, ch, num_levels=4):
        super().__init__()
        self.n = num_levels
        self.td = nn.ModuleList([BiFPNNode(ch) for _ in range(num_levels - 1)])
        self.bu_mid = nn.ModuleList([BiFPNNode3(ch) for _ in range(max(0, num_levels - 2))])
        self.bu_top = BiFPNNode(ch)

    def forward(self, feats):
        n = self.n
        td = list(feats)
        for i in range(n - 2, -1, -1):
            td[i] = self.td[i](feats[i],
                F.interpolate(td[i+1], size=feats[i].shape[2:], mode="nearest"))
        out = [td[0]]
        for i in range(1, n - 1):
            out.append(self.bu_mid[i-1](feats[i], td[i], F.max_pool2d(out[-1], 2, 2)))
        out.append(self.bu_top(td[n-1], F.max_pool2d(out[-1], 2, 2)))
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
        self.layers = nn.ModuleList([BiFPNLayer(neck_ch, num_levels)
                                     for _ in range(repeats)])
        self.out_channels = [neck_ch] * num_levels

    def forward(self, feats):
        p3 = self.lat3(feats[0])
        p4 = self.lat4(feats[1])
        p5 = self.lat5(feats[2])
        out = [p3, p4, p5]
        if self.num_levels >= 4:
            out.append(self.p6_down(p5))
        for layer in self.layers:
            out = layer(out)
        return out


# ============================================================================
# DETECTION HEAD — DFL + IoU-aware (NO centerness)
# ============================================================================

class DetectHead(nn.Module):
    """
    YOLOv8-style head:
      - cls branch  → nc logits per anchor
      - reg branch  → 4 × (reg_max+1) DFL logits per anchor
      - NO centerness branch
    Final score at inference = sigmoid(cls) * iou  (iou estimated from DFL box)
    """
    def __init__(self, ic, nc, reg_max=16, num_convs=4, num_levels=4):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.proj = nn.Parameter(
            torch.arange(reg_max + 1, dtype=torch.float32), requires_grad=False)

        ct = []
        rt = []
        for _ in range(num_convs):
            ct += [DWConv(ic, ic)]
            rt += [DWConv(ic, ic)]
        self.cls_tower = nn.Sequential(*ct)
        self.reg_tower = nn.Sequential(*rt)

        self.cls_pred = nn.Conv2d(ic, nc, 1)
        self.reg_pred = nn.Conv2d(ic, 4 * (reg_max + 1), 1)

        nn.init.constant_(self.cls_pred.bias, -math.log(99.))
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.zeros_(self.reg_pred.bias)

    def forward(self, feats):
        """Returns list of dicts: {"cls": [B,nc,H,W], "reg": [B,4*(rm+1),H,W]}"""
        out = []
        for f in feats:
            cf = self.cls_tower(f)
            rf = self.reg_tower(f)
            out.append({"cls": self.cls_pred(cf),
                        "reg": self.reg_pred(rf)})
        return out

    def decode_reg(self, reg_logits, anchor_pts, stride):
        """
        DFL decode: reg_logits [N, 4*(reg_max+1)] → xyxy boxes in pixel coords.
        anchor_pts: [N, 2] centre points.
        Returns [N, 4] xyxy.
        """
        rm1 = self.reg_max + 1
        N = reg_logits.shape[0]
        # [N, 4, rm1]
        dist = reg_logits.view(N, 4, rm1)
        dist = F.softmax(dist, dim=2)  # prob over bins
        dist = (dist * self.proj.to(dist.device)).sum(2)  # [N,4] mean
        dist = dist * stride  # scale to pixels
        x1 = anchor_pts[:, 0] - dist[:, 0]
        y1 = anchor_pts[:, 1] - dist[:, 1]
        x2 = anchor_pts[:, 0] + dist[:, 2]
        y2 = anchor_pts[:, 1] + dist[:, 3]
        return torch.stack([x1, y1, x2, y2], 1)  # [N,4]


# ============================================================================
# SEGMENTATION HEAD — FPN + ASPP
# ============================================================================

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
        self.aspp = ASPP(hidden, hidden)
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
        return {"prototypes": self.proto_net(fused),
                "semantic": self.sem_net(fused)}


# ============================================================================
# FULL MODEL
# ============================================================================

class FishSegDet(nn.Module):
    def __init__(self, num_classes=6, pretrained=True, neck_ch=256,
                 bifpn_repeats=3, mask_dim=128, num_fpn_levels=4,
                 backbone_variant="convnextv2_large.fcmae_ft_in22k_in1k_384",
                 reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.num_fpn_levels = num_fpn_levels
        self.reg_max = reg_max
        self.backbone = _make_backbone(backbone_variant, pretrained)
        self.neck = BiFPN(self.backbone.out_channels, neck_ch, bifpn_repeats, num_fpn_levels)
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
        nn.init.constant_(self.det_head.cls_pred.bias, -math.log(99.))

    def forward(self, x):
        nf = self.neck(self.backbone(x))
        so = self.seg_head(nf)
        return {"det_outputs": self.det_head(nf),
                "semantic": so["semantic"],
                "prototypes": so["prototypes"]}

    def count_parameters(self):
        p = {"backbone": self.backbone, "neck": self.neck,
             "det_head": self.det_head, "seg_head": self.seg_head}
        c = {k: sum(x.numel() for x in v.parameters()) for k, v in p.items()}
        c["total"] = sum(c.values())
        return c


def build_model(num_classes=6,pretrained=True,neck_ch=256,bifpn_repeats=3,
                mask_dim=128,num_fpn_levels=4,
                backbone_variant="convnextv2_large.fcmae_ft_in22k_in1k_384",
                reg_max=16):
    return FishSegDet(num_classes,pretrained,neck_ch,bifpn_repeats,
                       mask_dim,num_fpn_levels,backbone_variant,reg_max)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def _box_iou(b1, b2, eps=1e-7):
    """[N,4] × [M,4] → [N,M] IoU, xyxy."""
    a1=(b1[:,2]-b1[:,0]).clamp(0)*(b1[:,3]-b1[:,1]).clamp(0)
    a2=(b2[:,2]-b2[:,0]).clamp(0)*(b2[:,3]-b2[:,1]).clamp(0)
    ix1=torch.max(b1[:,None,0],b2[None,:,0])
    iy1=torch.max(b1[:,None,1],b2[None,:,1])
    ix2=torch.min(b1[:,None,2],b2[None,:,2])
    iy2=torch.min(b1[:,None,3],b2[None,:,3])
    inter=(ix2-ix1).clamp(0)*(iy2-iy1).clamp(0)
    return inter/(a1[:,None]+a2[None,:]-inter+eps)


def _ciou_loss(pred, gt, eps=1e-7):
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = gt.unbind(-1)
    inter=((torch.min(px2,gx2)-torch.max(px1,gx1)).clamp(0)*
           (torch.min(py2,gy2)-torch.max(py1,gy1)).clamp(0))
    ap=(px2-px1).clamp(0)*(py2-py1).clamp(0)
    ag=(gx2-gx1).clamp(0)*(gy2-gy1).clamp(0)
    u = ap + ag - inter + eps
    iou = inter / u
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    gcx = (gx1 + gx2) / 2
    gcy = (gy1 + gy2) / 2
    rho2=(pcx-gcx)**2+(pcy-gcy)**2
    cw=(torch.max(px2,gx2)-torch.min(px1,gx1)).clamp(0)
    ch=(torch.max(py2,gy2)-torch.min(py1,gy1)).clamp(0)
    c2=cw**2+ch**2+eps
    pw = (px2 - px1).clamp(0)\n    ph_ = (py2 - py1).clamp(0)
    gw=(gx2-gx1).clamp(0); gh_=(gy2-gy1).clamp(0)
    v=(4/math.pi**2)*(torch.atan(gw/(gh_+eps))-torch.atan(pw/(ph_+eps)))**2
    with torch.no_grad(): alpha=v/(1-iou+v+eps)
    return 1-iou+rho2/c2+alpha*v


def _dfl_loss(reg_logits, gt_ltrb, reg_max):
    """
    Distribution Focal Loss.
    reg_logits: [N, 4*(reg_max+1)]
    gt_ltrb:    [N, 4]  ground-truth distances in units of stride
    """
    rm1 = reg_max + 1
    N   = reg_logits.shape[0]
    if N == 0: return reg_logits.sum() * 0.
    pred = reg_logits.view(N*4, rm1)                     # [N*4, rm1]
    # soft label: clamp distances to [0, reg_max]
    tgt  = gt_ltrb.clamp(0., float(reg_max)).reshape(-1) # [N*4]
    tl   = tgt.long().clamp(0, reg_max-1)                # floor bin
    tr   = (tl + 1).clamp(0, reg_max)                   # ceil bin
    wl   = tr.float() - tgt                              # weight for floor
    wr   = tgt - tl.float()                              # weight for ceil
    loss = (F.cross_entropy(pred, tl, reduction="none") * wl +
            F.cross_entropy(pred, tr, reduction="none") * wr)
    return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.s = smooth

    def forward(self, pred, target):
        if pred.shape[-1] > 160:
            pred = F.adaptive_avg_pool2d(pred.float(), 160)
            target = F.interpolate(target.unsqueeze(1).float(),
                                   size=160, mode="nearest").squeeze(1).long()
        ps = F.softmax(pred[:, 1:].float(), dim=1)
        B, C, H, W = ps.shape
        fg = (target > 0).float()
        tgt = (target - 1).clamp(min=0)
        oh = F.one_hot(tgt, C).permute(0, 3, 1, 2).float() * fg.unsqueeze(1)
        inter = (ps * oh).sum((2, 3))
        dice = (2 * inter + self.s) / ((ps + oh).sum((2, 3)) + self.s)
        return 1 - dice.mean()


class FishSegDetLoss(nn.Module):
    """
    TAL assignment + DFL + IoU-aware classification (no centerness).

    TAL alignment metric:
        align = cls_score^alpha * iou^beta
    Top-k anchors per GT by alignment metric are positives.

    Classification target:
        t_cls[pos, gt_label] = iou_with_gt   (IoU-weighted soft label)
        t_cls[neg, :]        = 0

    Box loss: CIoU on decoded boxes.
    DFL loss: Distribution Focal Loss on raw reg logits vs stride-normalised LTRB.
    Seg loss: weighted CE + Dice.
    """

    def __init__(self, num_classes, strides=None, reg_max=16,
                 cls_w=1., box_w=3., dfl_w=0.5, seg_w=1.5,
                 topk=13, tal_alpha=0.5, tal_beta=6.0):
        super().__init__()
        self.nc = num_classes
        self.strides = strides or [8, 16, 32, 64]
        self.reg_max = reg_max
        self.cls_w = cls_w
        self.box_w = box_w
        self.dfl_w = dfl_w
        self.seg_w = seg_w
        self.topk = topk
        self.alpha = tal_alpha
        self.beta = tal_beta
        self.seg_ce = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.05)
        self.seg_dice = DiceLoss()
        self._proj = None  # will be created on first forward
        self._anchor_cache = {}

    # ── anchor helpers ────────────────────────────────────────────────────────

    def _get_anchors(self, det_outs, device):
        key=tuple(o["cls"].shape[2:4] for o in det_outs)+(str(device),)
        if key not in self._anchor_cache:
            pts=[]; strd=[]
            for lvl,o in enumerate(det_outs):
                s=self.strides[lvl]; _,_,H,W=o["cls"].shape
                yc=(torch.arange(H,device=device).float()+.5)*s
                xc=(torch.arange(W,device=device).float()+.5)*s
                gy,gx=torch.meshgrid(yc,xc,indexing="ij")
                pts.append(torch.stack([gx.reshape(-1),gy.reshape(-1)],1))
                strd.append(torch.full((H*W,),float(s),device=device))
            self._anchor_cache[key]=(torch.cat(pts,0),torch.cat(strd,0))
        return self._anchor_cache[key]

    def _decode(self, reg_logits_flat, ap, strides_flat):
        """
        reg_logits_flat: [N, 4*(reg_max+1)]
        Returns xyxy decoded boxes [N, 4].
        """
        rm1 = self.reg_max + 1
        if self._proj is None or self._proj.device != ap.device:
            self._proj = torch.arange(rm1, dtype=torch.float32, device=ap.device)
        N = reg_logits_flat.shape[0]
        dist = reg_logits_flat.view(N, 4, rm1)
        dist = F.softmax(dist, dim=2)
        dist = (dist * self._proj).sum(2) * strides_flat.unsqueeze(1)  # [N,4] pixels
        x1 = ap[:, 0] - dist[:, 0]
        y1 = ap[:, 1] - dist[:, 1]
        x2 = ap[:, 0] + dist[:, 2]
        y2 = ap[:, 1] + dist[:, 3]
        return torch.stack([x1, y1, x2, y2], 1)

    # ── TAL assignment (one image) ────────────────────────────────────────────

    def _tal(self, cls_logits, decoded_boxes, gt_xyxy, gt_labels, ap):
        """
        Task-Aligned Label Assignment.
        Returns matched_gt [N] (-1=neg), is_pos [N bool], alignment_metric [N,M].
        """
        N=cls_logits.shape[0]; M=gt_xyxy.shape[0]; dev=cls_logits.device

        # 1) anchor must lie inside GT box (hard constraint)
        ax,ay=ap[:,0],ap[:,1]
        inside=((ax.unsqueeze(1)>gt_xyxy[:,0])&
                (ax.unsqueeze(1)<gt_xyxy[:,2])&
                (ay.unsqueeze(1)>gt_xyxy[:,1])&
                (ay.unsqueeze(1)<gt_xyxy[:,3]))  # [N,M]

        if not inside.any():
            return (torch.full((N,),-1,dtype=torch.long,device=dev),
                    torch.zeros(N,dtype=torch.bool,device=dev),
                    torch.zeros(N,M,device=dev))

        with torch.no_grad():
            # IoU of every anchor's decoded box with every GT
            iou=_box_iou(decoded_boxes,gt_xyxy)  # [N,M]

            # Classification score for GT class at each anchor
            cls_s=cls_logits.sigmoid()
            gt_cls_score=cls_s[:,gt_labels.long()]  # [N,M]

            # TAL alignment metric
            align=gt_cls_score.pow(self.alpha)*iou.pow(self.beta)  # [N,M]
            align=align*inside.float()

            # Top-k per GT by alignment
            topk_k=min(self.topk, N)
            topk_val,_=align.topk(topk_k,dim=0,largest=True)   # [topk,M]
            thresh=topk_val[-1:,:]                               # [1,M]
            assign=(align>=thresh)&inside                        # [N,M]

            # Conflict resolution: anchor → GT with highest alignment
            align_a=align*assign.float()
            matched_gt=align_a.argmax(1)                         # [N]
            is_pos=assign.any(1)                                  # [N]
            matched_gt[~is_pos]=-1

        return matched_gt, is_pos, align

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, outputs, bboxes, labels, num_objects, seg_masks):
        det_outs=outputs["det_outputs"]; semantic=outputs["semantic"]
        dev=semantic.device; BS=bboxes.shape[0]; IS=float(CFG.img_size)

        ap, sf = self._get_anchors(det_outs, dev)  # [N,2], [N]
        N_total = ap.shape[0]

        # Flatten cls and reg across FPN levels
        all_cls = torch.cat([o["cls"].view(BS,self.nc,-1).permute(0,2,1)
                             for o in det_outs],1)           # [B,N,nc]
        rm1=self.reg_max+1
        all_reg = torch.cat([o["reg"].view(BS,4*rm1,-1).permute(0,2,1)
                             for o in det_outs],1)           # [B,N,4*rm1]

        total_cls=all_cls.sum()*0.
        total_box=all_cls.sum()*0.
        total_dfl=all_cls.sum()*0.
        n_pos=0

        for b in range(BS):
            M=int(num_objects[b].item())

            # Decode this image's boxes (detached for assignment)
            dec=self._decode(all_reg[b],ap,sf)               # [N,4]

            if M==0:
                # All negatives: cls target = 0, standard BCE
                neg_tgt=torch.zeros_like(all_cls[b])
                neg_loss=F.binary_cross_entropy_with_logits(
                    all_cls[b],neg_tgt,reduction="sum")/N_total
                total_cls=total_cls+neg_loss/BS
                continue

            gt_n=bboxes[b,:M]; gt_lb=labels[b,:M]
            gt_xy=torch.stack([(gt_n[:,0]-gt_n[:,2]/2)*IS,
                                (gt_n[:,1]-gt_n[:,3]/2)*IS,
                                (gt_n[:,0]+gt_n[:,2]/2)*IS,
                                (gt_n[:,1]+gt_n[:,3]/2)*IS],1)

            mg,ip,align=self._tal(all_cls[b].detach(),dec.detach(),
                                   gt_xy,gt_lb,ap)

            if not ip.any():
                neg_tgt=torch.zeros_like(all_cls[b])
                neg_loss=F.binary_cross_entropy_with_logits(
                    all_cls[b],neg_tgt,reduction="sum")/N_total
                total_cls=total_cls+neg_loss/BS
                continue

            pos_cls=all_cls[b][ip]       # [K,nc]
            pos_reg=all_reg[b][ip]       # [K,4*rm1]
            pos_ap=ap[ip]; pos_sf=sf[ip]
            pos_gt_idx=mg[ip]
            pos_gt_box=gt_xy[pos_gt_idx]
            pos_lb=gt_lb[pos_gt_idx]

            # Decoded boxes for positives (with grad)
            pos_dec=self._decode(pos_reg,pos_ap,pos_sf)  # [K,4]
            pos_iou=_box_iou(pos_dec.detach(),pos_gt_box).diag().clamp(0,1)

            # ── Classification: IoU-aware soft target ────────────────────────
            # Positive: t[k, gt_label] = iou_k  (all other classes = 0)
            # Negative: t = 0
            # Full image target → positives overwrite negatives
            cls_tgt=torch.zeros(N_total,self.nc,device=dev)
            cls_tgt[ip.nonzero(as_tuple=True)[0], pos_lb]=pos_iou

            # Normalise by alignment metric (TAL recommendation)
            align_norm=align[ip,pos_gt_idx]             # [K]
            align_norm=align_norm/(align_norm.sum()+1e-8)

            # Focal-weighted BCE on full anchor set (positives get IoU targets)
            ps=all_cls[b].sigmoid()
            w=torch.where(cls_tgt>0, cls_tgt,
                          0.75*ps.detach().pow(2.))
            bce=F.binary_cross_entropy_with_logits(
                all_cls[b],cls_tgt,reduction="none")
            cls_loss=(w*bce).sum()/N_total
            total_cls=total_cls+cls_loss

            # ── Box regression: CIoU ─────────────────────────────────────────
            box_loss=(_ciou_loss(pos_dec,pos_gt_box)*align_norm).sum()
            total_box=total_box+box_loss

            # ── DFL: stride-normalised LTRB ──────────────────────────────────
            gt_ltrb=torch.stack([
                pos_ap[:,0]-pos_gt_box[:,0],  # l
                pos_ap[:,1]-pos_gt_box[:,1],  # t
                pos_gt_box[:,2]-pos_ap[:,0],  # r
                pos_gt_box[:,3]-pos_ap[:,1],  # b
            ],1)/pos_sf.unsqueeze(1)          # normalise by stride
            dfl_loss=_dfl_loss(pos_reg,gt_ltrb.clamp(0.,float(self.reg_max)),
                               self.reg_max)
            total_dfl=total_dfl+dfl_loss
            n_pos+=int(ip.sum())

        norm=float(max(n_pos,1))
        cls_l=self.cls_w*total_cls
        box_l=self.box_w*total_box/norm
        dfl_l=self.dfl_w*total_dfl/norm

        mr=F.interpolate(seg_masks.unsqueeze(1).float(),
                         size=semantic.shape[2:],mode="nearest").squeeze(1).long()
        seg_l=self.seg_w*(self.seg_ce(semantic,mr)+self.seg_dice(semantic,mr))

        total=cls_l+box_l+dfl_l+seg_l
        return {"total":total,"cls":cls_l,"box":box_l,"dfl":dfl_l,"seg":seg_l}


# ============================================================================
# INFERENCE
# ============================================================================

class FishSegDetInference:
    def __init__(self,weights_path,device="cuda",img_size=640,
                 conf=0.25,iou=0.45,preprocess=True):
        self.device=device; self.img_size=img_size
        self.conf=conf; self.iou=iou; self.strides=CFG.fcos_strides
        ck=torch.load(weights_path,map_location=device,weights_only=False)
        nc=ck["model_state_dict"]["det_head.cls_pred.weight"].shape[0]
        mc=ck.get("config",{})
        self.model=build_model(
            nc,False,
            mc.get("neck_ch",        CFG.neck_channels),
            mc.get("bifpn_repeats",  CFG.bifpn_repeats),
            mc.get("mask_dim",       CFG.mask_dim),
            mc.get("num_fpn_levels", CFG.num_fpn_levels),
            mc.get("backbone_variant",CFG.backbone),
            mc.get("reg_max",        CFG.reg_max))
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(device).eval()
        self.pre=UnderwaterPreprocessor() if preprocess else None
        print(f"Loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M params")

    def preprocess(self,img):
        if self.pre: img=self.pre(img)
        h,w=img.shape[:2]; sc=self.img_size/max(h,w)
        nw,nh=int(w*sc),int(h*sc)
        pw,ph=(self.img_size-nw)//2,(self.img_size-nh)//2
        cv=np.full((self.img_size,self.img_size,3),114,np.uint8)
        cv[ph:ph+nh,pw:pw+nw]=cv2.resize(img,(nw,nh))
        t=((cv.astype(np.float32)/255.-[.485,.456,.406])/[.229,.224,.225])
        return (torch.from_numpy(t.transpose(2,0,1)).unsqueeze(0)
                     .float().to(self.device),
                {"sc":sc,"pw":pw,"ph":ph,"shape":(h,w)})

    def decode(self,det_outputs):
        """DFL decode + score = cls * iou (no centerness)."""
        ab=[]; as_=[]; al=[]
        head=self.model.det_head
        rm1=head.reg_max+1
        proj=head.proj.to(self.device)
        for lvl,o in enumerate(det_outputs):
            s=self.strides[lvl]; cl=o["cls"][0]; rg=o["reg"][0]
            _,H,W=cl.shape
            yc=(torch.arange(H,device=cl.device).float()+.5)*s
            xc=(torch.arange(W,device=cl.device).float()+.5)*s
            gy,gx=torch.meshgrid(yc,xc,indexing="ij")
            ap=torch.stack([gx.reshape(-1),gy.reshape(-1)],1)  # [HW,2]
            # DFL decode
            N=H*W
            dist=rg.view(4,rm1,-1).permute(2,0,1)              # [HW,4,rm1]
            dist=F.softmax(dist,dim=2)
            dist=(dist*proj).sum(2)*s                           # [HW,4]
            x1=ap[:,0]-dist[:,0]; y1=ap[:,1]-dist[:,1]
            x2=ap[:,0]+dist[:,2]; y2=ap[:,1]+dist[:,3]
            boxes=torch.stack([x1.clamp(0,self.img_size),
                                y1.clamp(0,self.img_size),
                                x2.clamp(0,self.img_size),
                                y2.clamp(0,self.img_size)],1)   # [HW,4]
            # scores = cls_sigmoid (IoU-weighted at training; use cls directly)
            scores_all=cl.reshape(cl.shape[0],-1).t().sigmoid() # [HW,nc]
            max_score,max_cls=scores_all.max(1)
            mk=max_score>self.conf
            if not mk.any(): continue
            ab.append(boxes[mk]); as_.append(max_score[mk]); al.append(max_cls[mk])
        if not ab: return None,None,None
        boxes=torch.cat(ab); scores=torch.cat(as_); lbs=torch.cat(al)
        k=nms(boxes,scores,self.iou)
        return boxes[k],scores[k],lbs[k]

    @torch.no_grad()
    def predict(self,img):
        img_t,meta=self.preprocess(img); t0=time.perf_counter()
        out=self.model(img_t); ms=(time.perf_counter()-t0)*1000
        boxes,scores,lbs=self.decode(out["det_outputs"])
        sem=out["semantic"][0].argmax(0).cpu().numpy()
        h,w=meta["shape"]
        if boxes is not None:
            boxes=boxes.cpu().numpy()
            boxes[:,[0,2]]=(boxes[:,[0,2]]-meta["pw"])/meta["sc"]
            boxes[:,[1,3]]=(boxes[:,[1,3]]-meta["ph"])/meta["sc"]
            boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w)
            boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h)
            scores=scores.cpu().numpy(); lbs=lbs.cpu().numpy()
        mc=sem[meta["ph"]:meta["ph"]+int(h*meta["sc"]),
               meta["pw"]:meta["pw"]+int(w*meta["sc"])]
        mask=cv2.resize(mc.astype(np.uint8),(w,h),
                        interpolation=cv2.INTER_NEAREST)
        return {"boxes":boxes,"scores":scores,"labels":lbs,
                "mask":mask,"time_ms":ms}

    def visualize(self,img,res,alpha=0.45):
        vis=img.copy()
        if res["mask"] is not None:
            ov=np.zeros_like(img)
            for ci in range(len(BRACKISH_CLASSES)):
                ov[res["mask"]==ci+1]=COLORS[ci]
            vis=cv2.addWeighted(vis,1-alpha,ov,alpha,0)
        if res["boxes"] is not None:
            for box,sc,lb in zip(res["boxes"],res["scores"],res["labels"]):
                x1,y1,x2,y2=box.astype(int); col=COLORS[int(lb)%len(COLORS)]
                name=BRACKISH_CLASSES.get(int(lb),str(lb))
                cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
                txt=f"{name}:{sc:.2f}"
                (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,.5,1)
                cv2.rectangle(vis,(x1,y1-th-8),(x1+tw+4,y1),col,-1)
                cv2.putText(vis,txt,(x1+2,y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.putText(vis,f"{res['time_ms']:.1f}ms",
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
        return vis


# ============================================================================
# UTILITIES
# ============================================================================

def generate_masks_from_bboxes(data_root,split="train",method="ellipse"):
    from tqdm import tqdm
    root=Path(data_root)
    img_dir=root/"images"/split; lbl_dir=root/"labels"/split
    msk_dir=root/"masks"/split; msk_dir.mkdir(parents=True,exist_ok=True)
    files=list(img_dir.glob("*.jpg"))+list(img_dir.glob("*.png"))
    for ip in tqdm(files,desc=f"Masks [{split}]"):
        img=cv2.imread(str(ip)); h,w=img.shape[:2]
        mask=np.zeros((h,w),np.uint8); lp=lbl_dir/f"{ip.stem}.txt"
        if lp.exists():
            for line in lp.read_text().splitlines():
                p=line.strip().split()
                if len(p)<5: continue
                ci=int(p[0]); cx_,cy_,bw_,bh_=map(float,p[1:5])
                if method=="ellipse":
                    cv2.ellipse(mask,(int(cx_*w),int(cy_*h)),
                                (int(bw_*w*.85/2),int(bh_*h*.85/2)),
                                0,0,360,ci+1,-1)
                else:
                    x1=max(0,int((cx_-bw_/2)*w)); y1=max(0,int((cy_-bh_/2)*h))
                    x2=min(w,int((cx_+bw_/2)*w)); y2=min(h,int((cy_+bh_/2)*h))
                    mask[y1:y2,x1:x2]=ci+1
        cv2.imwrite(str(msk_dir/f"{ip.stem}.png"),mask)
    print(f"Generated {len(files)} masks → {msk_dir}")
# FishSegDet

**Joint Instance Detection and Segmentation for Underwater Fish**

FishSegDet is a high-accuracy, single-stage model that simultaneously detects and segments fish in challenging underwater imagery. It combines a heavyweight ConvNeXtV2 backbone with a BiFPN feature pyramid, a Distribution Focal Loss (DFL) detection head, and an FPN+ASPP segmentation head — trained end-to-end with Task-Aligned Label Assignment.

---

## Architecture

```
Input Image (640×640)
       │
       ▼
┌─────────────────────────────┐
│  UnderwaterPreprocessor     │  White balance · CLAHE · Gamma correction
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Backbone: ConvNeXtV2-Large │  Pre-trained on IN-22K → IN-1K @ 384px
│  (timm: fcmae_ft_in22k_...)  │  Extracts C3, C4, C5 feature maps
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Neck: BiFPN  (3 repeats)   │  256-channel, 4 FPN levels P3–P6
│  Bidirectional feature      │  Fuses multi-scale context top-down
│  fusion with learned weights│  and bottom-up
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌────────────┐  ┌──────────────────┐
│  Detection │  │  Segmentation    │
│  Head      │  │  Head            │
│            │  │                  │
│ DFL: 4×17  │  │ FPN + ASPP       │
│   logits   │  │ (dilations 6,12, │
│ per anchor │  │  18, global avg) │
│            │  │                  │
│ IoU-aware  │  │ 128-d prototype  │
│ cls score  │  │ + per-instance   │
│ (no ctrns) │  │   mask coeff     │
└────────────┘  └──────────────────┘
```

| Component | Choice | Notes |
|-----------|--------|-------|
| Backbone | ConvNeXtV2-Large | `timm` pretrained on IN-22K+IN-1K @ 384px |
| Neck | BiFPN × 3 | 256 channels, 4 levels (P3–P6), learnable fusion weights |
| Detection | DFL + TAL | reg_max=16, topk=13, IoU-weighted cls (no centerness) |
| Segmentation | FPN + ASPP | ASPP dilations 6/12/18 + global pool, 128-d mask prototypes |
| Assignment | TAL | Alignment metric: cls^0.5 × IoU^6.0 |
| Box loss | CIoU | |
| Cls loss | IoU-weighted soft labels (VFL-style) | |
| Seg loss | CE + Dice (1.5×) | |

---

## Key Features

- **DFL detection** — models box regression as a probability distribution over discrete offsets, improving accuracy on small and partially occluded fish
- **TAL (Task-Aligned Label Assignment)** — unified metric aligns classification and regression objectives during training, replacing separate centerness
- **IoU-aware classification** — score at inference is `sigmoid(cls) × predicted_IoU`, matching the training target directly
- **Underwater-specific preprocessing** — gray-world white balance → CLAHE → gamma LUT applied per-image before augmentation
- **Class-balanced sampler** — inverse-sqrt-frequency weighting prevents majority-class dominance in multi-class datasets
- **Mosaic augmentation** — 4-image mosaic in Phase 2 for scale and context diversity
- **4-level FPN (P3–P6)** — covers strides 8, 16, 32, 64 for fish ranging from a few pixels to full-frame

---

## Training Schedule

| Phase | Epochs | Backbone | Augmentation | LR (backbone / head) |
|-------|--------|----------|--------------|----------------------|
| 1 | 60 | Frozen | Geometry + mild colour | 2e-5 / 3e-4 |
| 2 | 40 | Unfrozen | Heavy (mosaic, fog, motion blur) | 2e-5 / 3e-4 |
| 3 | 40 | Unfrozen | Reserved / custom | 2e-5 / 3e-4 |

Warmup: 10 epochs cosine warm-up then cosine-annealing decay.

---

## Dataset Support

| Format | Dataset | Notes |
|--------|---------|-------|
| COCO JSON | [USIS10K](https://github.com/LiamLian0852/USIS10K) | Underwater instance segmentation, 10 k images |
| YOLO TXT | [Brackish](https://public.roboflow.com/object-detection/brackish-underwater) | Multi-class fish + crab + shrimp |

Both datasets are auto-detected from `--data` path via `configure_dataset()`.

Expected directory layout:
```
USISDataset/
├── train/
│   ├── images/
│   └── annotations/instances_train.json   # COCO format
└── val/
    ├── images/
    └── annotations/instances_val.json
```

---

## Installation

```bash
git clone https://github.com/<your-username>/FishSegDet.git
cd FishSegDet
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0 with CUDA.

---

## Training

```bash
# Train with default config (ConvNeXtV2-Large, 640px, USIS dataset)
python train.py --data ./USISDataset

# Specify backbone and image size
python train.py --data ./USISDataset --backbone convnextv2_large.fcmae_ft_in22k_in1k_384 --img-size 640

# Resume from checkpoint
python train.py --data ./USISDataset --resume runs/exp1/last.pt

# Multi-GPU (DDP)
torchrun --nproc_per_node=2 train.py --data ./USISDataset
```

Checkpoints and training logs are saved to `runs/<timestamp>/`.

---

## Evaluation

```bash
# Full dataset evaluation — mAP + mIoU + Dice + per-class breakdown
python test.py --weights runs/exp1/best.pt --data ./USISDataset --eval

# Save per-image overlay visualizations alongside metrics
python test.py --weights runs/exp1/best.pt --data ./USISDataset --eval --save-vis
```

Reported metrics:
- Detection: mAP@0.5, mAP@0.5:0.95, per-class AP50
- Segmentation: mIoU, Dice, pixel accuracy, per-class IoU (top 10)
- Summary dashboard plot saved to `runs/eval_summary.png`

---

## Inference

```bash
# Single image
python test.py --weights runs/exp1/best.pt --image fish.jpg

# Folder of images (overlays saved to --out-dir)
python test.py --weights runs/exp1/best.pt --folder ./images/ --out-dir ./results/
```

---

## Publication Figures

```bash
# Generate IEEE/Springer-quality result panels (one image per class)
python visualization_pub.py --weights runs/exp1/best.pt --data ./USISDataset

# Export as PDF (600 dpi, camera-ready)
python visualization_pub.py --weights runs/exp1/best.pt --data ./USISDataset --format pdf
```

Outputs colorblind-safe overlays (Paul Tol palette) with feathered mask edges on a black background.

---

## Project Structure

```
FishSegDet/
├── main.py               # Model, datasets, losses, inference engine
├── train.py              # Training loop, optimizer, scheduler, logging
├── test.py               # Evaluation and single-image inference
├── visualization_pub.py  # Publication-quality figure generation
├── requirements.txt
└── README.md
```

---

## Requirements

See [requirements.txt](requirements.txt).

Core dependencies: `torch`, `torchvision`, `timm`, `albumentations`, `opencv-python`, `pycocotools`, `tqdm`, `matplotlib`.

---

## License

MIT — see [LICENSE](LICENSE).

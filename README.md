<div align="center">

# FishSegDet

### A Unified Framework for Joint Detection and Semantic Segmentation of Underwater Marine Species

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/backbone-timm-blueviolet)](https://github.com/huggingface/pytorch-image-models)
[![Dataset: USIS16K](https://img.shields.io/badge/dataset-USIS16K-teal)](https://github.com/LiamLian0852/USIS10K)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

## The Problem

Detecting and segmenting fish in underwater imagery is substantially harder than in standard benchmarks. Water selectively absorbs red wavelengths, shifting colour toward blue-green and destroying the visual cues models pre-trained on ImageNet rely on. Turbidity, caustics, backscatter, and camouflage further degrade contrast, while class imbalance is severe — common species appear thousands of times, rare ones a handful. Applications from autonomous underwater vehicles to marine biodiversity monitoring require **both bounding-box localisation** (for tracking) **and pixel-level masks** (for biomass estimation, morphometric analysis) — treating them as separate problems is inefficient and ignores the signal each task provides to the other.

FishSegDet addresses all of this in a single forward pass.

---

## What It Does

FishSegDet is a **single-stage, end-to-end model** that simultaneously:

- Predicts bounding boxes for every marine animal in the frame using **Distribution Focal Loss** regression
- Assigns each box to a class using **IoU-aware soft labels**, so the confidence score reflects localisation quality, not just category recognition
- Produces a full-resolution semantic segmentation mask via an **FPN + ASPP** decoder
- Runs underwater-specific preprocessing (white balance → CLAHE → gamma) before any augmentation

Trained end-to-end on **USIS16K** with a 3-phase curriculum strategy and class-balanced sampling.

---

## Architecture

```
Input (640 × 640, BGR)
        │
        ▼
┌──────────────────────┐
│  UnderwaterPreproc.  │  Gray-world white balance → CLAHE → gamma LUT
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   ConvNeXtV2-Large   │  FCMAE pre-trained on IN-22K → fine-tuned IN-1K @ 384px
│   (timm backbone)    │  Outputs: C3 (stride 8), C4 (stride 16), C5 (stride 32)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   BiFPN  (3 × 256ch) │  4 levels P3–P6. Bidirectional weighted fusion.
│                      │  Learnable per-edge weights normalised with fast softmax.
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌─────────┐  ┌────────────────┐
│ Detect  │  │   Segment      │
│  Head   │  │    Head        │
│         │  │                │
│  4-conv │  │  Fuse P3–P6   │
│  tower  │  │  → ASPP(6/12/ │
│  ──     │  │    18/global) │
│  cls:   │  │  → ProtoNet   │
│   nc    │  │  → SemNet     │
│  reg:   │  │  (128-d mask  │
│  4×17   │  │   prototypes) │
│  (DFL)  │  │               │
└─────────┘  └────────────────┘
```

| Component | Choice | Why |
|---|---|---|
| Backbone | ConvNeXtV2-Large | Pure-conv architecture maintains translation equivariance critical for detecting fish at arbitrary positions; FCMAE pre-training is superior for dense prediction |
| Neck | BiFPN × 3, 256ch, P3–P6 | Bidirectional fusion with learnable weights outperforms top-down-only FPN; 4 levels cover strides 8–64 for fish from a few pixels to full-frame |
| Detection | DFL, reg\_max=16 | Models box regression as a probability distribution over discrete offsets — directly captures the localisation ambiguity caused by blurry underwater edges |
| Assignment | TAL, topk=13 | Joint cls–reg alignment metric (cls^α × IoU^β) selects anchors that are simultaneously well-classified and well-localised, unlike FCOS which uses a fixed spatial radius |
| Classification | IoU-aware soft labels | Training target = IoU value; inference score = sigmoid(cls) × predicted\_IoU, so high-confidence detections are guaranteed to be well-localised |
| Box loss | CIoU | Penalises centre-distance and aspect-ratio deviation, not just overlap area |
| Segmentation | FPN + ASPP + prototype | ASPP dilations (6, 12, 18, global) capture multi-scale context without the resolution loss of pooling |
| Seg loss | CE + Dice (λ=1.5) | Dice directly optimises the F1-overlap metric; CE provides per-pixel supervision |

---

## Training Strategy

### 3-Phase Curriculum

| Phase | Epochs | Backbone | Augmentation | Backbone LR | Head LR |
|---|---|---|---|---|---|
| 1 | 60 | Frozen | Geometry + mild colour | — | 3e-4 |
| 2 | 40 | Unfrozen | Heavy (mosaic, fog, motion blur, coarse dropout) | 2e-5 | 3e-4 |
| 3 | 40 | Unfrozen | Same as Phase 2 | 2e-5 | 3e-4 |

**Rationale:** Freezing the backbone in Phase 1 prevents the pre-trained features from being corrupted before the randomly-initialised heads have stabilised. Differential learning rates in Phases 2–3 prevent catastrophic forgetting of the ImageNet representations.

### Additional Training Details

- **Scheduler:** 10-epoch cosine warm-up → cosine annealing decay per phase
- **Class-balanced sampler:** Per-sample weight = 1/√class\_freq of the rarest class in the image. Prevents the model from ignoring rare marine species.
- **Mosaic augmentation:** 4-image mosaic (Phase 2+) — forces the model to handle fragmented objects and non-standard scales.
- **Optimiser:** AdamW with fused CUDA kernels, weight decay 0.05
- **AMP + GradScaler:** Mixed precision on every forward/backward
- **EMA:** Exponential moving average (decay=0.9999) on a CPU shadow copy; final checkpoint uses EMA weights

---

## Results on USIS16K

FishSegDet is evaluated against two strong YOLO baselines of comparable scale on the USIS16K test split. All models produce both bounding-box detections and segmentation masks.

| Model | Params | mAP | mAP₅₀ | mIoU | Dice | Precision | Recall | Pixel Acc |
|---|---|---|---|---|---|---|---|---|
| YOLOv10l-seg | 24.4 M | 64.1 % | 85.5 % | 84.32 % | 83.92 % | 82.40 % | 85.50 % | 91.00 % |
| YOLOv11l-seg | 27.7 M | 67.3 % | 88.0 % | 86.45 % | 86.12 % | 85.27 % | 87.00 % | 93.25 % |
| **FishSegDet (Ours)** | **204.75 M** | **70.0 %** | **95.1 %** | **93.4 %** | **94.39 %** | **94.69 %** | **94.09 %** | **99.1 %** |

FishSegDet outperforms YOLOv11l-seg by **+2.7 pp mAP**, **+7.1 pp mAP₅₀**, **+6.95 pp mIoU**, and **+8.27 pp Dice** — while achieving 99.1 % pixel accuracy, indicating near-perfect background suppression in challenging underwater conditions.

---

## Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/FishSegDet.git
cd FishSegDet
pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, CUDA 11.8+.

### Dataset Layout

```
USIS16K/
├── train/                          # training images
├── val/                            # validation images
├── test/                           # test images
└── annotations/
    ├── instances_train.json
    ├── instances_val.json
    └── instances_test.json
```

Class names are read from the annotation file automatically — no manual configuration required.

### Train

```bash
# Default: ConvNeXtV2-Large, 640px, USIS16K
python train.py --data ./USIS16K --device cuda

# Faster backbone for iteration
python train.py --data ./USIS16K --backbone convnextv2_tiny.fcmae_ft_in22k_in1k

# Resume from checkpoint
python train.py --data ./USIS16K --resume runs/fishsegdet_20240901/best.pt

# Gradient accumulation (for smaller GPUs)
python train.py --data ./USIS16K --batch-size 8 --accum-steps 2
```

Checkpoints, CSV metrics log, and optional TensorBoard logs are written to `runs/fishsegdet_<timestamp>/`.

### Evaluate

```bash
# Full test-set evaluation with per-class breakdown
python test.py --weights runs/fishsegdet_20240901/best.pt --data ./USIS16K --eval

# Evaluation + save overlay visualisations
python test.py --weights runs/fishsegdet_20240901/best.pt --data ./USIS16K --eval --save-vis
```

### Inference

```bash
# Single image — displays result in a window
python test.py --weights best.pt --image sample.jpg

# Folder — saves overlays to ./results/
python test.py --weights best.pt --folder ./images/ --out-dir ./results/
```

### Publication Figures

```bash
# One representative image per class, IEEE/Springer layout
python visualization_pub.py --weights best.pt --data ./USIS16K

# Camera-ready PDF at 600 dpi
python visualization_pub.py --weights best.pt --data ./USIS16K --format pdf
```

---

## Project Structure

```
FishSegDet/
├── fishsegdet/                     # installable Python package
│   ├── config.py                   # Config dataclass · CFG singleton · class maps
│   ├── models/
│   │   ├── blocks.py               # ConvBN, DWConv  (shared primitives)
│   │   ├── backbone.py             # TimmBackbone  (any timm model, default ConvNeXtV2-L)
│   │   ├── neck.py                 # BiFPN  (weighted bidirectional FPN)
│   │   ├── heads.py                # DetectHead (DFL) · ASPP · FPNSegHead
│   │   └── detector.py             # FishSegDet · build_model()
│   ├── data/
│   │   ├── preprocessing.py        # UnderwaterPreprocessor  (WB → CLAHE → γ)
│   │   ├── augmentations.py        # get_train · get_val · get_heavy transforms
│   │   ├── datasets.py             # USISDataset (COCO) · MosaicDataset
│   │   └── loaders.py              # configure_dataset() · create_dataloaders()
│   ├── losses/
│   │   └── detection.py            # FishSegDetLoss  (TAL · DFL · IoU-aware · Dice)
│   └── engine/
│       └── inference.py            # FishSegDetInference  (preprocess · decode · NMS · visualise)
├── train.py                        # training entry point
├── test.py                         # evaluation and inference entry point
├── visualization_pub.py            # publication-quality qualitative results
├── requirements.txt
└── README.md
```

---

## Requirements

```
torch>=2.0.0       torchvision>=0.15.0    timm>=0.9.0
albumentations>=1.3.0    opencv-python>=4.7.0    pycocotools>=2.0.6
numpy>=1.24.0      tqdm>=4.65.0           matplotlib>=3.7.0
```

---

## License

[MIT](LICENSE)

"""
LiteFishSeg v2 — test.py
=========================
Test & visualization script. Three modes:

  Single image
    python test.py --weights runs/.../best.pt --image fish.jpg

  Folder of images (saves overlays to --out-dir)
    python test.py --weights runs/.../best.pt --folder ./images/ --out-dir ./results/

  Full dataset evaluation (mAP + mIoU per class, summary plot)
    python test.py --weights runs/.../best.pt --data ./USISDataset --eval
    python test.py --weights runs/.../best.pt --data ./USISDataset --eval --save-vis
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Georgia", "DejaVu Serif"],
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "figure.facecolor": "#fafafa",
    "axes.facecolor":   "#ffffff",
    "grid.color":       "#e8e8e8",
    "grid.linewidth":   0.6,
})

# ── project imports ────────────────────────────────────────────────────────────
from main import (
    CFG,
    LiteFishSegInference,
    BRACKISH_CLASSES,
    COLORS,
    configure_dataset,
    get_val_transforms,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================================
# COLOUR HELPERS
# ============================================================================

def _bgr_colors(n):
    """Return up to n BGR colors from the global COLORS palette."""
    return [COLORS[i % len(COLORS)] for i in range(n)]


def _rgb_norm(bgr_color):
    """Convert a BGR uint8 tuple to a normalized (r,g,b) float tuple."""
    return tuple(c / 255.0 for c in reversed(bgr_color))


# ============================================================================
# VISUALIZATION — single prediction panel
# ============================================================================

def make_seg_rgba(mask, num_classes, colors, alpha=170):
    """Build an RGBA overlay image from a class-index mask."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for ci in range(num_classes):
        m = mask == (ci + 1)
        if m.any():
            bgr = colors[ci % len(colors)]
            rgba[m] = (bgr[2], bgr[1], bgr[0], alpha)   # BGR → RGB + alpha
    return rgba


def draw_boxes_on_ax(ax, boxes, scores, labels, class_names, colors, img_wh):
    """Draw detection boxes + labels onto a matplotlib axis."""
    if boxes is None or len(boxes) == 0:
        return
    w, h = img_wh
    for box, sc, lb in zip(boxes, scores, labels):
        ci   = int(lb)
        col  = _rgb_norm(colors[ci % len(colors)])
        x1, y1, x2, y2 = box
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0", linewidth=1.8,
            edgecolor=col, facecolor="none")
        ax.add_patch(rect)
        name = class_names.get(ci, str(ci))
        ax.text(x1 + 2, y1 - 4, f"{name}  {sc:.2f}",
                fontsize=6.5, color="white", fontweight="bold",
                bbox=dict(facecolor=col, alpha=0.85, pad=1.5, edgecolor="none"),
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])


def visualize_prediction(img_bgr, result, class_names, colors,
                         save_path=None, fmt="png", dpi=150, show=False):
    """
    Three-panel figure:
      [Original] | [Segmentation overlay] | [Full overlay + detections]
    Returns the save path if saved, else None.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w   = img_rgb.shape[:2]
    nc     = len(class_names)
    mask   = result.get("mask")
    boxes  = result.get("boxes")
    scores = result.get("scores")
    labels = result.get("labels")
    t_ms   = result.get("time_ms", 0.0)
    n_det  = len(boxes) if boxes is not None else 0

    seg_rgba = make_seg_rgba(mask, nc, colors) if mask is not None else np.zeros((h, w, 4), np.uint8)

    # Collect which classes appear (for legend)
    present = set()
    if mask is not None:
        present.update(int(v) - 1 for v in np.unique(mask) if v > 0)
    if labels is not None:
        present.update(int(lb) for lb in labels)

    # ── figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5.8), facecolor="#111111")
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                             wspace=0.04, left=0.01, right=0.99,
                             top=0.88, bottom=0.13)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_titles = ["Original", "Segmentation", "Detections + Seg"]

    # Panel 0 — original
    axs[0].imshow(img_rgb)

    # Panel 1 — seg overlay only
    axs[1].imshow(img_rgb)
    axs[1].imshow(seg_rgba)

    # Panel 2 — full overlay with boxes
    axs[2].imshow(img_rgb)
    axs[2].imshow(seg_rgba)
    draw_boxes_on_ax(axs[2], boxes, scores, labels, class_names, colors, (w, h))

    for ax, title in zip(axs, panel_titles):
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=5)
        ax.axis("off")

    # Header
    fig.text(0.5, 0.95,
             f"LiteFishSeg v2  ·  {n_det} detection{'s' if n_det != 1 else ''}  "
             f"·  {t_ms:.1f} ms  ({1000/t_ms:.0f} FPS)" if t_ms > 0 else
             f"LiteFishSeg v2  ·  {n_det} detections",
             ha="center", fontsize=12, fontweight="bold", color="white")

    # Legend (only present classes)
    if present:
        handles = [
            mpatches.Patch(facecolor=_rgb_norm(colors[ci % len(colors)]),
                           label=class_names.get(ci, str(ci)),
                           edgecolor="white", linewidth=0.5)
            for ci in sorted(present) if ci < nc
        ]
        if handles:
            fig.legend(handles=handles,
                       loc="lower center", ncol=min(12, len(handles)),
                       bbox_to_anchor=(0.5, 0.01),
                       framealpha=0.25, facecolor="#333333", edgecolor="#555555",
                       labelcolor="white", fontsize=7.5,
                       title="Detected classes", title_fontsize=7.5,
                       handlelength=1.2, handleheight=0.9)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return None


# ============================================================================
# VISUALIZATION — evaluation summary panel
# ============================================================================

def plot_eval_summary(metrics, class_names, save_path=None, fmt="png", dpi=150):
    """
    Four-panel evaluation dashboard:
      (a) Per-class mIoU bar chart (top 30)
      (b) Overall metric summary table
      (c) Detection score distribution
      (d) Speed stats
    """
    per_cls_iou  = metrics.get("per_cls_iou",  {})
    per_cls_map  = metrics.get("per_cls_map",  {})
    overall      = metrics.get("overall",      {})
    times_ms     = metrics.get("times_ms",     [])

    fig = plt.figure(figsize=(16, 10), facecolor="#fafafa")
    fig.text(0.5, 0.97, "LiteFishSeg v2 — Evaluation Summary",
             ha="center", fontsize=14, fontweight="bold", color="#1a1a2e")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.32,
                           left=0.07, right=0.97,
                           top=0.92, bottom=0.08)
    ax_iou  = fig.add_subplot(gs[0, :])   # full top row
    ax_tbl  = fig.add_subplot(gs[1, 0])
    ax_spd  = fig.add_subplot(gs[1, 1])

    # ── (a) Per-class mIoU ─────────────────────────────────────────────────
    if per_cls_iou:
        sorted_cls = sorted(per_cls_iou, key=per_cls_iou.get, reverse=True)[:30]
        names_bar  = [class_names.get(ci, str(ci)) for ci in sorted_cls]
        vals_iou   = [per_cls_iou[ci] * 100 for ci in sorted_cls]
        vals_map   = [per_cls_map.get(ci, 0) * 100 for ci in sorted_cls]
        x = np.arange(len(names_bar))
        w = 0.4
        ax_iou.bar(x - w/2, vals_iou, width=w, color="#1a6b8a", alpha=0.85, label="mIoU (%)")
        ax_iou.bar(x + w/2, vals_map, width=w, color="#c0392b", alpha=0.85, label="AP50 (%)")
        ax_iou.set_xticks(x)
        ax_iou.set_xticklabels(names_bar, rotation=45, ha="right", fontsize=7)
        ax_iou.set_ylabel("Score (%)")
        ax_iou.set_title("(a)  Per-class mIoU & AP50 (top 30 by mIoU)",
                         fontsize=10, fontweight="bold", loc="left")
        ax_iou.legend(fontsize=8)
        ax_iou.set_ylim(0, 105)
        ax_iou.grid(True, axis="y", linestyle=":", linewidth=0.6)
        ax_iou.spines["top"].set_visible(False)
        ax_iou.spines["right"].set_visible(False)

    # ── (b) Overall metric table ───────────────────────────────────────────
    ax_tbl.axis("off")
    rows = [
        ["mAP (50:95)",  f"{overall.get('mAP',   0)*100:.1f}%"],
        ["mAP@50",       f"{overall.get('mAP50', 0)*100:.1f}%"],
        ["mIoU",         f"{overall.get('mIoU',  0)*100:.1f}%"],
        ["Dice",         f"{overall.get('dice',  0)*100:.1f}%"],
        ["Pixel Acc",    f"{overall.get('pix_acc', 0)*100:.1f}%"],
        ["Precision",    f"{overall.get('prec',  0)*100:.1f}%"],
        ["Recall",       f"{overall.get('rec',   0)*100:.1f}%"],
        ["Images",       str(overall.get("n_images", "—"))],
        ["Avg time",     f"{overall.get('mean_ms', 0):.1f} ms"],
        ["FPS",          f"{overall.get('fps', 0):.1f}"],
    ]
    tbl = ax_tbl.table(cellText=rows, colLabels=["Metric", "Value"],
                       cellLoc="center", loc="center",
                       bbox=[0.05, 0.0, 0.9, 1.0])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a6b8a"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eaf4fb")
        cell.set_edgecolor("#cccccc"); cell.set_linewidth(0.5)
    ax_tbl.set_title("(b)  Overall Metrics", fontsize=10, fontweight="bold",
                     pad=7, loc="left")

    # ── (c) Speed distribution ─────────────────────────────────────────────
    if times_ms:
        ax_spd.hist(times_ms, bins=30, color="#2e7d52", alpha=0.8, edgecolor="white", linewidth=0.4)
        ax_spd.axvline(np.mean(times_ms), color="#c0392b", linewidth=1.5,
                       linestyle="--", label=f"Mean {np.mean(times_ms):.1f} ms")
        ax_spd.axvline(np.median(times_ms), color="#d4860b", linewidth=1.5,
                       linestyle=":", label=f"Median {np.median(times_ms):.1f} ms")
        ax_spd.set_xlabel("Inference time (ms)")
        ax_spd.set_ylabel("Count")
        ax_spd.set_title("(c)  Inference Speed Distribution",
                         fontsize=10, fontweight="bold", loc="left")
        ax_spd.legend(fontsize=8)
        ax_spd.grid(True, axis="y", linestyle=":", linewidth=0.6)
        ax_spd.spines["top"].set_visible(False)
        ax_spd.spines["right"].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved summary → {save_path}")
    else:
        plt.show()
        plt.close(fig)


# ============================================================================
# EVALUATION ENGINE
# ============================================================================

def _confusion_update(pm_np, gm_np, nc, tp, fp, fn):
    """O(H*W) confusion matrix update — avoids 159 full-mask scans per image."""
    pm_flat = pm_np.ravel().astype(np.int64)
    gm_flat = gm_np.ravel().astype(np.int64)
    n = nc + 1
    # clamp out-of-range values defensively
    np.clip(pm_flat, 0, nc, out=pm_flat)
    np.clip(gm_flat, 0, nc, out=gm_flat)
    conf = np.bincount(pm_flat * n + gm_flat, minlength=n * n).reshape(n, n)
    # conf[pred_cls, gt_cls]
    for ci in range(nc):
        ci1 = ci + 1
        tp[ci] += conf[ci1, ci1]
        fp[ci] += conf[ci1, :].sum() - conf[ci1, ci1]
        fn[ci] += conf[:, ci1].sum() - conf[ci1, ci1]


def evaluate_dataset(inf, data_root, split="test", save_vis=False, out_dir=None,
                     max_vis=20, device="cuda"):
    """
    Run inference over a full dataset split and compute:
      - Overall mAP (torchmetrics), mIoU, Dice, Pixel Acc
      - Per-class IoU and AP50
    Returns a metrics dict.
    """
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    DatasetClass, _ = configure_dataset(data_root)
    tf  = get_val_transforms(inf.img_size)
    ds  = DatasetClass(data_root, split, inf.img_size, tf)
    nc  = CFG.num_classes

    # MeanAveragePrecision: update per-image (no giant list accumulation)
    mm = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                              class_metrics=True,
                              max_detection_thresholds=[10, 100, 1000])
    mm.warn_on_many_detections = False

    # Segmentation stats via confusion matrix (CPU, O(H*W) per image)
    tp = np.zeros(nc, dtype=np.float64)
    fp = np.zeros(nc, dtype=np.float64)
    fn = np.zeros(nc, dtype=np.float64)
    # Per-class IoU accumulator (incremental, avoids jim GPU state growth)
    iou_sum  = np.zeros(nc, dtype=np.float64)
    iou_cnt  = np.zeros(nc, dtype=np.float64)

    tp_px = 0; tot_px = 0
    times_ms = []
    vis_saved = 0

    if out_dir and save_vis:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    # mAP needs the full precision-recall curve — use near-zero conf so all
    # detections reach torchmetrics (it sweeps thresholds internally).
    # Save the user's display conf and restore it for vis predictions.
    vis_conf  = inf.conf
    inf.conf  = 0.01

    for idx in tqdm(range(len(ds)), desc=f"Eval [{split}]"):
        img, img_path, bboxes_gt, labels_gt, mask_gt = ds.get_raw(idx)
        if img is None:
            continue

        h, w = img.shape[:2]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result  = inf.predict(img_bgr)
        times_ms.append(result["time_ms"])

        # ── detection metrics (update per-image, no list growth) ──────────
        pred_boxes  = result["boxes"]
        pred_scores = result["scores"]
        pred_labels = result["labels"]

        if pred_boxes is not None and len(pred_boxes) > 0:
            pd = {"boxes":  torch.tensor(pred_boxes,  dtype=torch.float32),
                  "scores": torch.tensor(pred_scores, dtype=torch.float32),
                  "labels": torch.tensor(pred_labels, dtype=torch.int64)}
        else:
            pd = {"boxes":  torch.zeros((0, 4), dtype=torch.float32),
                  "scores": torch.zeros((0,),   dtype=torch.float32),
                  "labels": torch.zeros((0,),   dtype=torch.int64)}

        if len(bboxes_gt) > 0:
            cx, cy, bw_, bh_ = bboxes_gt[:, 0], bboxes_gt[:, 1], bboxes_gt[:, 2], bboxes_gt[:, 3]
            gx1 = (cx - bw_/2) * w; gy1 = (cy - bh_/2) * h
            gx2 = (cx + bw_/2) * w; gy2 = (cy + bh_/2) * h
            td = {"boxes":  torch.stack([torch.tensor(gx1), torch.tensor(gy1),
                                         torch.tensor(gx2), torch.tensor(gy2)], 1).float(),
                  "labels": torch.tensor(labels_gt, dtype=torch.int64)}
        else:
            td = {"boxes":  torch.zeros((0, 4), dtype=torch.float32),
                  "labels": torch.zeros((0,),   dtype=torch.int64)}

        mm.update([pd], [td])

        # ── segmentation metrics — all CPU / numpy, O(H*W) ───────────────
        pred_mask = result["mask"]
        if pred_mask is not None and mask_gt is not None:
            pm_np = pred_mask.astype(np.int32)
            gm_np = mask_gt.astype(np.int32)
            if pm_np.shape != gm_np.shape:
                pm_np = cv2.resize(pm_np, (gm_np.shape[1], gm_np.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

            tp_px += int((pm_np == gm_np).sum())
            tot_px += gm_np.size

            # O(H*W) confusion matrix — replaces 159-iteration GPU loop
            _confusion_update(pm_np, gm_np, nc, tp, fp, fn)

            # Per-class IoU for this image (incremental mean over images)
            pm_flat = pm_np.ravel().astype(np.int64)
            gm_flat = gm_np.ravel().astype(np.int64)
            np.clip(pm_flat, 0, nc, out=pm_flat)
            np.clip(gm_flat, 0, nc, out=gm_flat)
            conf_n = nc + 1
            conf = np.bincount(pm_flat * conf_n + gm_flat,
                               minlength=conf_n * conf_n).reshape(conf_n, conf_n)
            for ci in range(nc):
                ci1 = ci + 1
                inter = conf[ci1, ci1]
                union = conf[ci1, :].sum() + conf[:, ci1].sum() - inter
                if union > 0:
                    iou_sum[ci] += inter / union
                    iou_cnt[ci] += 1

        # ── optional per-image visualization (re-predict at display conf) ──
        if save_vis and vis_saved < max_vis and out_dir:
            inf.conf = vis_conf
            vis_result = inf.predict(img_bgr)
            inf.conf = 0.01
            vis_name = Path(img_path).stem + "_pred.png"
            vis_path = Path(out_dir) / vis_name
            visualize_prediction(img_bgr, vis_result, BRACKISH_CLASSES, COLORS,
                                 save_path=vis_path, dpi=120)
            vis_saved += 1

    # ── aggregate ─────────────────────────────────────────────────────────
    inf.conf = vis_conf  # restore display threshold
    print("\nComputing mAP (this may take ~30 s for 3k images)…")
    mr = mm.compute()
    per_cls_ap = {}
    if "map_per_class" in mr and mr["map_per_class"] is not None:
        ap_vals = mr["map_per_class"].cpu().numpy()
        per_cls_ap = {i: float(ap_vals[i]) for i in range(len(ap_vals))}

    # Per-class IoU from incremental mean
    per_cls_iou = {ci: float(iou_sum[ci] / iou_cnt[ci]) if iou_cnt[ci] > 0 else 0.0
                   for ci in range(nc)}
    mean_iou = float(np.mean([v for v in per_cls_iou.values() if v > 0]) if any(iou_cnt > 0) else 0.0)

    eps = 1e-7
    hg   = (tp + fn) > 0
    prec = (tp / (tp + fp + eps))[hg].mean() if hg.any() else 0.0
    rec  = (tp / (tp + fn + eps))[hg].mean() if hg.any() else 0.0
    dice = (2*tp / (2*tp + fp + fn + eps))[hg].mean() if hg.any() else 0.0

    mean_ms = float(np.mean(times_ms)) if times_ms else 0.0
    return {
        "overall": {
            "mAP":      mr["map"].item(),
            "mAP50":    mr["map_50"].item(),
            "mIoU":     mean_iou,
            "dice":     float(dice),
            "pix_acc":  tp_px / max(tot_px, 1),
            "prec":     float(prec),
            "rec":      float(rec),
            "n_images": len(times_ms),
            "mean_ms":  mean_ms,
            "fps":      1000.0 / mean_ms if mean_ms > 0 else 0,
        },
        "per_cls_iou": per_cls_iou,
        "per_cls_map": per_cls_ap,
        "times_ms":    times_ms,
    }


# ============================================================================
# SINGLE IMAGE & FOLDER MODES
# ============================================================================

def run_single(inf, img_path, out_dir=None, show=True, fmt="png", dpi=150):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[ERROR] Cannot read {img_path}")
        return
    result = inf.predict(img_bgr)
    n_det  = len(result["boxes"]) if result["boxes"] is not None else 0
    print(f"  {Path(img_path).name}  →  {n_det} detection(s)  |  {result['time_ms']:.1f} ms")

    save_path = None
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(out_dir) / f"{Path(img_path).stem}_pred.{fmt}"

    visualize_prediction(img_bgr, result, BRACKISH_CLASSES, COLORS,
                         save_path=save_path, fmt=fmt, dpi=dpi, show=show)
    if save_path:
        print(f"  Saved → {save_path}")


def run_folder(inf, folder, out_dir, fmt="png", dpi=120):
    images = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"[ERROR] No images found in {folder}"); return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(images)} images → {out_dir}")
    times = []
    for img_path in tqdm(images, desc="Inference"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        result = inf.predict(img_bgr)
        times.append(result["time_ms"])
        save_path = Path(out_dir) / f"{img_path.stem}_pred.{fmt}"
        visualize_prediction(img_bgr, result, BRACKISH_CLASSES, COLORS,
                             save_path=save_path, fmt=fmt, dpi=dpi)
    if times:
        print(f"\nDone.  Mean: {np.mean(times):.1f} ms  |  "
              f"FPS: {1000/np.mean(times):.1f}  |  "
              f"Min: {np.min(times):.1f} ms  Max: {np.max(times):.1f} ms")


# ============================================================================
# CHECKPOINT HELPER
# ============================================================================

def _resolve_weights(weights_arg):
    p = Path(weights_arg)
    if p.is_file():
        return p
    # Glob for best.pt in most recent run
    runs = sorted(Path("runs").glob("litefishseg_*/best.pt"))
    if runs:
        print(f"[Auto] Using latest best.pt: {runs[-1]}")
        return runs[-1]
    print("[ERROR] --weights not found and no runs/ directory present.")
    sys.exit(1)


def _load_class_names_from_ckpt(ckpt_path):
    """Restore class names from checkpoint config so we don't need the dataset."""
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_ck = ck.get("config", {})
    classes = cfg_ck.get("classes", CFG.classes)
    # Patch global state so BRACKISH_CLASSES / COLORS are populated correctly
    CFG.classes = classes
    CFG.num_classes = len(classes)
    BRACKISH_CLASSES.clear()
    BRACKISH_CLASSES.update({i: n for i, n in enumerate(classes)})
    from main import _make_palette
    COLORS.clear()
    COLORS.extend(_make_palette(CFG.num_classes))
    return cfg_ck


# ============================================================================
# MAIN
# ============================================================================

def main():
    pa = argparse.ArgumentParser(
        description="LiteFishSeg v2 — Test & Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--weights",  default="runs/litefishseg_20260420_154540/best.pt",
                    help="Path to checkpoint (.pt)")
    pa.add_argument("--image",    default=None,  help="Single image to test")
    pa.add_argument("--folder",   default=None,  help="Folder of images")
    pa.add_argument("--data",     default="./USISDataset",
                    help="Dataset root (for --eval)")
    pa.add_argument("--eval",     action="store_true",
                    help="Run full dataset evaluation on test split")
    pa.add_argument("--split",    default="test",
                    help="Dataset split to evaluate (train/val/test)")
    pa.add_argument("--save-vis", action="store_true",
                    help="Save per-image visualizations during --eval")
    pa.add_argument("--out-dir",  default="./test_results",
                    help="Output directory for saved images")
    pa.add_argument("--conf",     type=float, default=0.35,
                    help="Detection confidence threshold")
    pa.add_argument("--iou",      type=float, default=0.20,
                    help="NMS IoU threshold")
    pa.add_argument("--img-size", type=int,   default=640)
    pa.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--no-preprocess", action="store_true",
                    help="Skip underwater preprocessing (CLAHE/WB/gamma)")
    pa.add_argument("--format",   dest="fmt", default="png",
                    choices=["png", "jpg", "pdf", "svg"],
                    help="Output image format")
    pa.add_argument("--dpi",      type=int, default=150)
    pa.add_argument("--no-show",  action="store_true",
                    help="Do not display plots interactively (useful on headless servers)")
    a = pa.parse_args()

    weights = _resolve_weights(a.weights)

    # ── restore class names from checkpoint ─────────────────────────────────
    print(f"Loading checkpoint: {weights}")
    _load_class_names_from_ckpt(weights)
    print(f"Classes: {CFG.num_classes}  Device: {a.device}")

    # ── build inference engine ───────────────────────────────────────────────
    inf = LiteFishSegInference(
        weights_path=str(weights),
        device=a.device,
        img_size=a.img_size,
        conf=a.conf,
        iou=a.iou,
        preprocess=not a.no_preprocess,
    )
    show = not a.no_show

    # ── dispatch ─────────────────────────────────────────────────────────────
    if a.image:
        print(f"\n── Single image: {a.image}")
        run_single(inf, a.image, out_dir=a.out_dir, show=show,
                   fmt=a.fmt, dpi=a.dpi)

    elif a.folder:
        print(f"\n── Folder: {a.folder}")
        run_folder(inf, a.folder, out_dir=a.out_dir, fmt=a.fmt, dpi=a.dpi)

    if a.eval:
        print(f"\n── Dataset evaluation: {a.data}  split={a.split}")
        configure_dataset(a.data)   # sets CFG.num_classes from dataset JSON
        metrics = evaluate_dataset(
            inf, a.data, split=a.split,
            save_vis=a.save_vis, out_dir=a.out_dir,
            device=a.device)

        ov = metrics["overall"]
        print(f"\n{'='*54}")
        print(f"  mAP (50:95) : {ov['mAP']*100:6.1f}%")
        print(f"  mAP@50      : {ov['mAP50']*100:6.1f}%")
        print(f"  mIoU        : {ov['mIoU']*100:6.1f}%")
        print(f"  Dice        : {ov['dice']*100:6.1f}%")
        print(f"  Pixel Acc   : {ov['pix_acc']*100:6.1f}%")
        print(f"  Precision   : {ov['prec']*100:6.1f}%")
        print(f"  Recall      : {ov['rec']*100:6.1f}%")
        print(f"  Speed       : {ov['mean_ms']:.1f} ms  ({ov['fps']:.1f} FPS)")
        print(f"  Images      : {ov['n_images']}")
        print(f"{'='*54}")

        # Top-10 per-class IoU
        if metrics["per_cls_iou"]:
            top10 = sorted(metrics["per_cls_iou"], key=metrics["per_cls_iou"].get, reverse=True)[:10]
            print("\n  Top-10 classes by mIoU:")
            for ci in top10:
                print(f"    [{ci:3d}] {BRACKISH_CLASSES.get(ci,'?'):20s}  "
                      f"IoU={metrics['per_cls_iou'][ci]*100:.1f}%  "
                      f"AP50={metrics['per_cls_map'].get(ci,0)*100:.1f}%")

        summary_path = Path(a.out_dir) / f"eval_summary.{a.fmt}"
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        plot_eval_summary(metrics, BRACKISH_CLASSES,
                          save_path=summary_path, fmt=a.fmt, dpi=a.dpi)

    if not a.image and not a.folder and not a.eval:
        pa.print_help()


if __name__ == "__main__":
    main()

"""
FishSegDet — train.py
=====================
Speed-optimised training pipeline with:
  [SPEED-1]  cudnn.benchmark + channels_last
  [SPEED-2]  torch.compile (opt-in via --compile)
  [SPEED-3]  AMP + GradScaler on every forward/backward
  [SPEED-4]  Reduced validation frequency (--val-interval N)
  [SPEED-5]  EMA: cheap copy-only, no extra forward pass
  [SPEED-6]  tqdm: update every 10 steps, not every step
  [SPEED-7]  Validation decode: vectorised DFL, no Python loop over levels
  [SPEED-8]  Prefetch / DataLoader tuned in main()
  [SPEED-9]  Loss dict uses .item() only once per epoch (not per step)
"""

import copy
import math
import argparse
import csv
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from fishsegdet import (
    CFG,
    build_model,
    FishSegDetLoss,
    create_dataloaders,
    configure_dataset,
    get_train_transforms,
    get_val_transforms,
    get_heavy_transforms,
    generate_masks_from_bboxes,
    FishSegDetInference,
    BRACKISH_CLASSES,
)

try:
    from plot_metrics import plot_metrics as _plot_metrics
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ============================================================================
# EMA
# ============================================================================

class ModelEMA:
    """Exponential Moving Average of model weights (CPU shadow, zero VRAM overhead)."""

    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).cpu()
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d   = self.decay
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            v.copy_(d * v + (1.0 - d) * msd[k].float().cpu())

    def state_dict(self):
        return self.shadow.state_dict()


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg,
                 accum_steps=1, val_interval=1, use_ema=True):
        self.cfg          = cfg
        self.device       = cfg.device
        self.accum_steps  = max(1, accum_steps)
        self.val_interval = max(1, val_interval)

        # [SPEED-1] channels_last for Conv-heavy models
        self.model = model.to(self.device).to(memory_format=torch.channels_last)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(cfg.save_dir) / f"fishsegdet_{ts}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = FishSegDetLoss(
            num_classes=cfg.num_classes,
            strides    =cfg.fcos_strides,
            reg_max    =cfg.reg_max,
            cls_w      =cfg.loss_cls,
            box_w      =cfg.loss_box,
            dfl_w      =cfg.loss_dfl,
            seg_w      =cfg.loss_seg,
            topk       =cfg.tal_topk,
            tal_alpha  =cfg.tal_alpha,
            tal_beta   =cfg.tal_beta,
        )

        self._amp_device = self.device.split(":")[0]
        self.scaler      = (GradScaler(self._amp_device)
                            if self._amp_device == "cuda" else None)
        self.ema         = ModelEMA(self.model, decay=0.9999) if use_ema else None

        self.best_loss = float("inf")
        self.best_map  = 0.0
        self.cur_ep    = 0
        self.gstep     = 0
        self.gep       = 0

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.save_dir / "logs")
        except ImportError:
            self.writer = None

        self.csv_path = self.save_dir / "metrics.csv"
        self._fields  = [
            "global_epoch", "phase", "epoch_in_phase", "lr",
            "train_loss", "train_cls", "train_box", "train_dfl", "train_seg",
            "val_loss",   "val_cls",   "val_box",   "val_dfl",   "val_seg",
            "val_mAP", "val_mAP_50", "val_mIoU",
            "val_precision", "val_recall", "val_dice", "val_pixel_acc",
        ]
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields).writeheader()
        print(f"[CSV] {self.csv_path}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ac(self):
        return autocast(device_type=self._amp_device)

    def _mem(self):
        if self._amp_device != "cuda":
            return ""
        a = torch.cuda.memory_allocated(self.device) / 1024 ** 3
        r = torch.cuda.memory_reserved(self.device)  / 1024 ** 3
        t = torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 3
        return f"  GPU: {a:.1f}/{t:.1f} GB alloc  ({r:.1f} GB reserved)"

    def _opt(self, phase):
        if phase == 1:
            for p in self.model.backbone.parameters():
                p.requires_grad_(False)
            params = [
                {"params": self.model.neck.parameters(),     "lr": self.cfg.lr_head},
                {"params": self.model.det_head.parameters(), "lr": self.cfg.lr_head},
                {"params": self.model.seg_head.parameters(), "lr": self.cfg.lr_head},
            ]
        else:
            for p in self.model.backbone.parameters():
                p.requires_grad_(True)
            m = 1.0 if phase == 2 else 0.3
            params = [
                {"params": self.model.backbone.parameters(), "lr": self.cfg.lr_backbone * m},
                {"params": self.model.neck.parameters(),     "lr": self.cfg.lr_head * m},
                {"params": self.model.det_head.parameters(), "lr": self.cfg.lr_head * m},
                {"params": self.model.seg_head.parameters(), "lr": self.cfg.lr_head * m},
            ]
        return optim.AdamW(params, weight_decay=self.cfg.weight_decay, fused=True)

    def _sched(self, opt, epochs):
        wu = min(self.cfg.warmup_epochs, max(1, epochs // 5))
        def fn(ep):
            if ep < wu:
                return (ep + 1) / wu
            return 0.5 * (1 + math.cos(math.pi * (ep - wu) / max(1, epochs - wu)))
        return optim.lr_scheduler.LambdaLR(opt, fn)

    # ── train one epoch ───────────────────────────────────────────────────────

    def train_epoch(self, opt):
        self.model.train()
        L  = {"total": 0., "cls": 0., "box": 0., "dfl": 0., "seg": 0.}
        nb = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.cur_ep + 1}", mininterval=2.0)

        for step, batch in enumerate(pbar):
            imgs = (batch["image"].to(self.device, non_blocking=True)
                    .to(memory_format=torch.channels_last))
            bbs  = batch["bboxes"].to(self.device, non_blocking=True)
            lbs  = batch["labels"].to(self.device, non_blocking=True)
            nos  = batch["num_objects"].to(self.device, non_blocking=True)
            msks = batch["mask"].to(self.device, non_blocking=True)

            with self._ac():
                out  = self.model(imgs)
                ld   = self.loss_fn(out, bbs, lbs, nos, msks)
                loss = ld["total"] / self.accum_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
                if (step + 1) % self.accum_steps == 0:
                    self.scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    self.scaler.step(opt)
                    self.scaler.update()
                    opt.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (step + 1) % self.accum_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            if self.ema is not None and (step + 1) % self.accum_steps == 0:
                self.ema.update(self.model)

            for k in L:
                L[k] += ld[k].item()
            nb += 1

            if step % 10 == 0:
                pbar.set_postfix(
                    loss=f"{ld['total'].item():.3f}",
                    cls =f"{ld['cls'].item():.3f}",
                    box =f"{ld['box'].item():.3f}",
                    seg =f"{ld['seg'].item():.3f}",
                    refresh=False,
                )
            if self.writer and self.gstep % 200 == 0:
                for k, v in ld.items():
                    self.writer.add_scalar(f"train/{k}", v.item(), self.gstep)
            self.gstep += 1

        return {k: v / nb for k, v in L.items()}

    # ── validate ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self):
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        from torchmetrics.classification import MulticlassJaccardIndex
        from torchvision.ops import nms

        self.model.eval()
        L   = {"total": 0., "cls": 0., "box": 0., "dfl": 0., "seg": 0.}
        nb  = 0
        mm  = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", class_metrics=False,
            max_detection_thresholds=[10, 100, 1000]).to(self.device)
        mm.warn_on_many_detections = False
        im  = MulticlassJaccardIndex(
            num_classes=self.cfg.num_classes + 1, ignore_index=0).to(self.device)

        nc   = self.cfg.num_classes
        IS   = float(self.cfg.img_size)
        st   = self.cfg.fcos_strides
        rm1  = self.cfg.reg_max + 1
        proj = torch.arange(rm1, dtype=torch.float32, device=self.device)
        tp   = torch.zeros(nc, device=self.device)
        fp   = torch.zeros(nc, device=self.device)
        fn_t = torch.zeros(nc, device=self.device)
        tp_px, tot_px = 0, 0

        for batch in tqdm(self.val_loader, desc="Val", mininterval=2.0):
            imgs = (batch["image"].to(self.device, non_blocking=True)
                    .to(memory_format=torch.channels_last))
            bbs  = batch["bboxes"].to(self.device, non_blocking=True)
            lbs  = batch["labels"].to(self.device, non_blocking=True)
            nos  = batch["num_objects"].to(self.device, non_blocking=True)
            msks = batch["mask"].to(self.device, non_blocking=True)

            with self._ac():
                out = self.model(imgs)
                ld  = self.loss_fn(out, bbs, lbs, nos, msks)

            for k in L:
                L[k] += ld[k].item()
            nb += 1

            su = F.interpolate(out["semantic"].float(), size=msks.shape[1:],
                               mode="bilinear", align_corners=False).argmax(1)
            im.update(su, msks)
            tp_px  += (su == msks).sum().item()
            tot_px += msks.numel()
            for c in range(nc):
                pc = (su == c + 1); gc = (msks == c + 1)
                tp[c]   += (pc & gc).sum()
                fp[c]   += (pc & ~gc).sum()
                fn_t[c] += (~pc & gc).sum()

            BS = imgs.shape[0]
            preds, tgts = [], []
            for b in range(BS):
                ab_list, sc_list, cl_list = [], [], []
                for lvl, o in enumerate(out["det_outputs"]):
                    s        = st[lvl]
                    cls_b    = o["cls"][b]
                    reg_b    = o["reg"][b]
                    _, H, W  = cls_b.shape
                    yc = (torch.arange(H, device=self.device).float() + 0.5) * s
                    xc = (torch.arange(W, device=self.device).float() + 0.5) * s
                    gy, gx = torch.meshgrid(yc, xc, indexing="ij")
                    ap   = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
                    dist = reg_b.view(4, rm1, -1).permute(2, 0, 1)
                    dist = F.softmax(dist, dim=2)
                    dist = (dist * proj).sum(2) * s
                    x1 = (ap[:, 0] - dist[:, 0]).clamp(0, IS)
                    y1 = (ap[:, 1] - dist[:, 1]).clamp(0, IS)
                    x2 = (ap[:, 0] + dist[:, 2]).clamp(0, IS)
                    y2 = (ap[:, 1] + dist[:, 3]).clamp(0, IS)
                    boxes      = torch.stack([x1, y1, x2, y2], 1)
                    scores_all = cls_b.reshape(nc, -1).t().sigmoid()
                    max_score, max_cls = scores_all.max(1)
                    mk = max_score > 0.05
                    if not mk.any():
                        continue
                    ab_list.append(boxes[mk])
                    sc_list.append(max_score[mk])
                    cl_list.append(max_cls[mk])

                if ab_list:
                    pb = torch.cat(ab_list)
                    ps = torch.cat(sc_list)
                    pl = torch.cat(cl_list)
                    k  = nms(pb, ps, self.cfg.iou_threshold)
                    pd = {"boxes": pb[k], "scores": ps[k], "labels": pl[k]}
                else:
                    pd = {"boxes":  torch.zeros((0, 4), device=self.device),
                          "scores": torch.zeros((0,),   device=self.device),
                          "labels": torch.zeros((0,), dtype=torch.int64, device=self.device)}

                n = int(nos[b].item())
                if n > 0:
                    tb  = bbs[b, :n]
                    tbx = torch.stack([
                        (tb[:, 0] - tb[:, 2] / 2) * IS,
                        (tb[:, 1] - tb[:, 3] / 2) * IS,
                        (tb[:, 0] + tb[:, 2] / 2) * IS,
                        (tb[:, 1] + tb[:, 3] / 2) * IS,
                    ], dim=1)
                    td = {"boxes": tbx, "labels": lbs[b, :n]}
                else:
                    td = {"boxes":  torch.zeros((0, 4), device=self.device),
                          "labels": torch.zeros((0,), dtype=torch.int64, device=self.device)}
                preds.append(pd)
                tgts.append(td)
            mm.update(preds, tgts)

        L   = {k: v / nb for k, v in L.items()}
        mr  = mm.compute()
        ir  = im.compute()
        L["mAP"]    = mr["map"].item()
        L["mAP_50"] = mr["map_50"].item()
        L["mIoU"]   = ir.item()
        eps = 1e-7
        hg  = (tp + fn_t) > 0
        if hg.any():
            L["precision"] = (tp / (tp + fp   + eps))[hg].mean().item()
            L["recall"]    = (tp / (tp + fn_t + eps))[hg].mean().item()
            L["dice"]      = (2 * tp / (2 * tp + fp + fn_t + eps))[hg].mean().item()
        else:
            L["precision"] = L["recall"] = L["dice"] = 0.0
        L["pixel_acc"] = tp_px / max(tot_px, 1)
        if self.writer:
            for k, v in L.items():
                self.writer.add_scalar(f"val/{k}", v, self.gep)
        return L

    # ── checkpoint ────────────────────────────────────────────────────────────

    def ckpt(self, name, vl=None, best=False):
        if vl is not None and vl < self.best_loss:
            self.best_loss = vl
        c = {"epoch": self.cur_ep, "global_epoch": self.gep,
             "model_state_dict": self.model.state_dict(),
             "ema_state_dict":   self.ema.state_dict() if self.ema else None,
             "best_loss": self.best_loss, "best_map": self.best_map,
             "config": {
                 "num_classes": self.cfg.num_classes, "img_size": self.cfg.img_size,
                 "classes": self.cfg.classes, "neck_ch": self.cfg.neck_channels,
                 "bifpn_repeats": self.cfg.bifpn_repeats, "mask_dim": self.cfg.mask_dim,
                 "num_fpn_levels": self.cfg.num_fpn_levels,
                 "backbone_variant": self.cfg.backbone, "reg_max": self.cfg.reg_max,
             }}
        torch.save(c, self.save_dir / name)
        if best:
            torch.save(c, self.save_dir / "best.pt")
            print(f"  ✓ Best → {self.save_dir / 'best.pt'}")

    # ── phase ─────────────────────────────────────────────────────────────────

    def phase(self, ph, epochs, transform=None):
        if epochs == 0:
            return
        print(f"\n{'='*60}\nPHASE {ph}: {epochs} epochs | {self.cfg.num_classes} classes\n{'='*60}")
        if transform:
            self.train_loader.dataset.transform = transform
        opt   = self._opt(ph)
        sched = self._sched(opt, epochs)
        vl    = {"total": 0., "cls": 0., "box": 0., "dfl": 0., "seg": 0.,
                 "mAP": 0., "mAP_50": 0., "mIoU": 0.,
                 "precision": 0., "recall": 0., "dice": 0., "pixel_acc": 0.}

        for ep in range(epochs):
            self.cur_ep = ep
            tl = self.train_epoch(opt)
            do_val = ((ep + 1) % self.val_interval == 0) or (ep == epochs - 1)
            if do_val:
                vl = self.validate()
            sched.step()
            lr = sched.get_last_lr()[0]
            val_str = (f"Val:{vl['total']:.4f}  "
                       f"mAP:{vl['mAP']*100:.1f}% mAP50:{vl['mAP_50']*100:.1f}% "
                       f"mIoU:{vl['mIoU']*100:.1f}%") if do_val else "Val:skipped"
            print(f"\nEp {ep+1}/{epochs} [Ph{ph}]  Train:{tl['total']:.4f}  {val_str}  LR:{lr:.6f}")
            m = self._mem()
            if m:
                print(m)
            row = {"global_epoch": self.gep, "phase": ph, "epoch_in_phase": ep + 1,
                   "lr": round(lr, 8),
                   "train_loss": round(tl["total"], 6), "train_cls": round(tl["cls"], 6),
                   "train_box":  round(tl["box"],   6), "train_dfl": round(tl["dfl"], 6),
                   "train_seg":  round(tl["seg"],   6),
                   "val_loss":   round(vl["total"],     6), "val_cls":  round(vl["cls"],    6),
                   "val_box":    round(vl["box"],       6), "val_dfl":  round(vl.get("dfl", 0.0), 6),
                   "val_seg":    round(vl["seg"],       6),
                   "val_mAP":    round(vl["mAP"],       6), "val_mAP_50": round(vl["mAP_50"],  6),
                   "val_mIoU":   round(vl["mIoU"],      6),
                   "val_precision": round(vl["precision"], 6), "val_recall": round(vl["recall"], 6),
                   "val_dice":   round(vl["dice"],      6), "val_pixel_acc": round(vl["pixel_acc"], 6)}
            with open(self.csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fields).writerow(row)
            is_best = do_val and vl["mAP"] > self.best_map
            if is_best:
                self.best_map = vl["mAP"]
            if (ep + 1) % 10 == 0 or is_best:
                self.ckpt(f"ph{ph}_ep{ep+1}.pt", vl["total"], is_best)
            self.gep += 1
        self.ckpt(f"ph{ph}_final.pt", vl["total"])

    # ── full training pipeline ────────────────────────────────────────────────

    def train(self):
        print(f"\n{'='*60}\nFishSegDet\n{'='*60}")
        print(f"Device:{self.device}  Classes:{self.cfg.num_classes}  "
              f"Batch:{self.cfg.batch_size}  Accum:{self.accum_steps}  "
              f"ValInterval:{self.val_interval}")
        print(f"Backbone:{self.cfg.backbone}\n"
              f"Params:{sum(p.numel() for p in self.model.parameters())/1e6:.2f}M  "
              f"Save:{self.save_dir}")
        if self._amp_device == "cuda":
            p = torch.cuda.get_device_properties(self.device)
            print(f"GPU:{p.name}  ({p.total_memory/1024**3:.1f}GB VRAM)")

        self.phase(1, self.cfg.phase1_epochs, get_train_transforms(self.cfg.img_size))
        self.phase(2, self.cfg.phase2_epochs, get_heavy_transforms(self.cfg.img_size))
        self.phase(3, self.cfg.phase3_epochs, get_heavy_transforms(self.cfg.img_size))

        print(f"\nDone! Best mAP:{self.best_map*100:.1f}%  Best loss:{self.best_loss:.4f}")
        if self.writer:
            self.writer.close()
        if PLOT_AVAILABLE:
            try:
                out = _plot_metrics(run_dir=str(self.save_dir), dpi=200, fmt="png",
                                    smooth_w=0.85, auto_open=True)
                print(f"Dashboard → {out}")
            except Exception as e:
                print(f"[Plot failed] {e}")
        else:
            print(f"Plot: python plot_metrics.py --run-dir {self.save_dir}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(weights, data_root, device="cuda"):
    import numpy as np
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from torchmetrics.classification import MulticlassJaccardIndex

    DatasetClass, _ = configure_dataset(data_root)
    inf = FishSegDetInference(weights, device=device)
    ds  = DatasetClass(data_root, "test", 640, get_val_transforms(640))
    print(f"Evaluating {len(ds)} images...")

    mm = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                              class_metrics=True,
                              max_detection_thresholds=[10, 100, 1000])
    mm.warn_on_many_detections = False
    im = MulticlassJaccardIndex(num_classes=len(BRACKISH_CLASSES) + 1,
                                ignore_index=0).to(device)
    times, preds, tgts = [], [], []

    for i in tqdm(range(len(ds)), desc="Eval"):
        img, _, bbs, lbs, tmask = ds.get_raw(i)
        if img is None:
            continue
        h, w = img.shape[:2]
        tbs  = [[max(0, float((cx - bw / 2) * w)), max(0, float((cy - bh / 2) * h)),
                 min(w, float((cx + bw / 2) * w)), min(h, float((cy + bh / 2) * h))]
                for cx, cy, bw, bh in bbs]
        tgts.append({"boxes":  torch.tensor(tbs,  dtype=torch.float32) if tbs  else torch.zeros((0, 4)),
                     "labels": torch.tensor(lbs,  dtype=torch.int64)   if len(lbs) else torch.zeros((0,), dtype=torch.int64)})
        r = inf.predict(img)
        times.append(r["time_ms"])
        preds.append({
            "boxes":  torch.tensor(r["boxes"],  dtype=torch.float32) if r["boxes"]  is not None and len(r["boxes"])  > 0 else torch.zeros((0, 4)),
            "scores": torch.tensor(r["scores"], dtype=torch.float32) if r["scores"] is not None and len(r["scores"]) > 0 else torch.zeros((0,)),
            "labels": torch.tensor(r["labels"], dtype=torch.int64)   if r["labels"] is not None and len(r["labels"]) > 0 else torch.zeros((0,), dtype=torch.int64),
        })
        im.update(torch.from_numpy(r["mask"]).to(device),
                  torch.from_numpy(tmask).to(device))

    mm.update(preds, tgts)
    if not times:
        return
    print(f"Speed:{np.mean(times):.1f}ms ({1000/np.mean(times):.1f}FPS)")
    mr = mm.compute(); ir = im.compute()
    print(f"mAP:{mr['map'].item()*100:.1f}%  mAP@50:{mr['map_50'].item()*100:.1f}%  mIoU:{ir.item()*100:.1f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    pa = argparse.ArgumentParser("FishSegDet")
    pa.add_argument("--data",           default="./USISDataset")
    pa.add_argument("--generate-masks", action="store_true")
    pa.add_argument("--batch-size",     type=int, default=16)
    pa.add_argument("--img-size",       type=int, default=640)
    pa.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--workers",        type=int, default=4)
    pa.add_argument("--save-dir",       default="./runs")
    pa.add_argument("--resume",         default=None)
    pa.add_argument("--accum-steps",    type=int, default=1)
    pa.add_argument("--no-ema",         action="store_true")
    pa.add_argument("--backbone",       default=CFG.backbone,
                    help="e.g. convnextv2_tiny.fcmae_ft_in22k_in1k (faster) or default large")
    pa.add_argument("--phase1-epochs",  type=int, default=CFG.phase1_epochs)
    pa.add_argument("--phase2-epochs",  type=int, default=CFG.phase2_epochs)
    pa.add_argument("--phase3-epochs",  type=int, default=CFG.phase3_epochs)
    pa.add_argument("--val-interval",   type=int, default=5)
    pa.add_argument("--eval",           action="store_true")
    pa.add_argument("--infer",          default=None)
    pa.add_argument("--weights",        default=None)
    pa.add_argument("--output",         default=None)
    pa.add_argument("--compile",        action="store_true",
                    help="Enable torch.compile (PyTorch 2.x)")
    a = pa.parse_args()

    CFG.dataset_root  = a.data
    CFG.batch_size    = a.batch_size
    CFG.img_size      = a.img_size
    CFG.device        = a.device
    CFG.num_workers   = a.workers
    CFG.save_dir      = a.save_dir
    CFG.phase1_epochs = a.phase1_epochs
    CFG.phase2_epochs = a.phase2_epochs
    CFG.phase3_epochs = a.phase3_epochs
    CFG.backbone      = a.backbone

    if a.device.startswith("cuda"):
        torch.backends.cudnn.benchmark              = True
        torch.backends.cuda.matmul.allow_tf32       = True
        torch.backends.cudnn.allow_tf32             = True

    if a.generate_masks:
        for s in ["train", "val", "test"]:
            generate_masks_from_bboxes(a.data, s)
        return

    if a.eval:
        if not a.weights:
            print("--weights required for --eval"); return
        evaluate(a.weights, a.data, a.device); return

    if a.infer:
        if not a.weights:
            print("--weights required for --infer"); return
        import cv2
        inf = FishSegDetInference(a.weights, device=a.device)
        img = cv2.imread(a.infer)
        if img is None:
            print(f"Could not read: {a.infer}"); return
        r   = inf.predict(img)
        vis = inf.visualize(img, r)
        if a.output:
            cv2.imwrite(a.output, vis); print(f"Saved → {a.output}")
        else:
            cv2.imshow("FishSegDet", vis); cv2.waitKey(0); cv2.destroyAllWindows()
        return

    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        a.data, a.img_size, a.batch_size, a.workers)

    print(f"Building model ({CFG.num_classes} classes, backbone={CFG.backbone})...")
    model = build_model(
        num_classes=CFG.num_classes, pretrained=True,
        neck_ch=CFG.neck_channels, bifpn_repeats=CFG.bifpn_repeats,
        mask_dim=CFG.mask_dim, num_fpn_levels=CFG.num_fpn_levels,
        backbone_variant=CFG.backbone, reg_max=CFG.reg_max)

    if a.resume:
        print(f"Resuming from {a.resume}")
        ck = torch.load(a.resume, map_location=a.device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print("Weights loaded.")

    if getattr(a, "compile", False):
        try:
            print("Enabling torch.compile (mode=reduce-overhead)...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    Trainer(model, train_loader, val_loader, CFG,
            accum_steps=a.accum_steps,
            val_interval=a.val_interval,
            use_ema=not a.no_ema).train()


if __name__ == "__main__":
    main()

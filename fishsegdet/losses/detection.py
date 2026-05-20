from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG


# ── IoU helpers ───────────────────────────────────────────────────────────────

def _box_iou(b1, b2, eps=1e-7):
    """[N,4] × [M,4] → [N,M] IoU (xyxy format)."""
    a1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    ix1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    iy1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    ix2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    iy2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    return inter / (a1[:, None] + a2[None, :] - inter + eps)


def _ciou_loss(pred, gt, eps=1e-7):
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = gt.unbind(-1)
    inter = ((torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(0) *
             (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(0))
    ap  = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ag  = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    u   = ap + ag - inter + eps
    iou = inter / u
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    gcx = (gx1 + gx2) / 2
    gcy = (gy1 + gy2) / 2
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
    cw   = (torch.max(px2, gx2) - torch.min(px1, gx1)).clamp(0)
    ch   = (torch.max(py2, gy2) - torch.min(py1, gy1)).clamp(0)
    c2   = cw ** 2 + ch ** 2 + eps
    pw   = (px2 - px1).clamp(0)
    ph_  = (py2 - py1).clamp(0)
    gw   = (gx2 - gx1).clamp(0)
    gh_  = (gy2 - gy1).clamp(0)
    v    = (4 / math.pi ** 2) * (torch.atan(gw / (gh_ + eps)) - torch.atan(pw / (ph_ + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return 1 - iou + rho2 / c2 + alpha * v


def _dfl_loss(reg_logits, gt_ltrb, reg_max):
    """Distribution Focal Loss on stride-normalised LTRB distances."""
    rm1 = reg_max + 1
    N   = reg_logits.shape[0]
    if N == 0:
        return reg_logits.sum() * 0.0
    pred = reg_logits.view(N * 4, rm1)
    tgt  = gt_ltrb.clamp(0.0, float(reg_max)).reshape(-1)
    tl   = tgt.long().clamp(0, reg_max - 1)
    tr   = (tl + 1).clamp(0, reg_max)
    wl   = tr.float() - tgt
    wr   = tgt - tl.float()
    return (F.cross_entropy(pred, tl, reduction="none") * wl +
            F.cross_entropy(pred, tr, reduction="none") * wr).mean()


# ── segmentation loss ─────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.s = smooth

    def forward(self, pred, target):
        if pred.shape[-1] > 160:
            pred   = F.adaptive_avg_pool2d(pred.float(), 160)
            target = F.interpolate(target.unsqueeze(1).float(),
                                   size=160, mode="nearest").squeeze(1).long()
        ps = F.softmax(pred[:, 1:].float(), dim=1)
        B, C, H, W = ps.shape
        fg  = (target > 0).float()
        tgt = (target - 1).clamp(min=0)
        oh  = F.one_hot(tgt, C).permute(0, 3, 1, 2).float() * fg.unsqueeze(1)
        inter = (ps * oh).sum((2, 3))
        dice  = (2 * inter + self.s) / ((ps + oh).sum((2, 3)) + self.s)
        return 1 - dice.mean()


# ── main loss ─────────────────────────────────────────────────────────────────

class FishSegDetLoss(nn.Module):
    """
    TAL assignment + DFL + IoU-aware classification (no centerness).

    Assignment metric: cls_score^alpha * iou^beta
    Top-k positives per GT; classification target = IoU (soft label).
    """

    def __init__(self, num_classes, strides=None, reg_max=16,
                 cls_w=1.0, box_w=3.0, dfl_w=0.5, seg_w=1.5,
                 topk=13, tal_alpha=0.5, tal_beta=6.0):
        super().__init__()
        self.nc       = num_classes
        self.strides  = strides or [8, 16, 32, 64]
        self.reg_max  = reg_max
        self.cls_w    = cls_w
        self.box_w    = box_w
        self.dfl_w    = dfl_w
        self.seg_w    = seg_w
        self.topk     = topk
        self.alpha    = tal_alpha
        self.beta     = tal_beta
        self.seg_ce   = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.05)
        self.seg_dice = DiceLoss()
        self._proj    = None
        self._anchor_cache: dict = {}

    def _get_anchors(self, det_outs, device):
        key = tuple(o["cls"].shape[2:4] for o in det_outs) + (str(device),)
        if key not in self._anchor_cache:
            pts, strd = [], []
            for lvl, o in enumerate(det_outs):
                s    = self.strides[lvl]
                _, _, H, W = o["cls"].shape
                yc   = (torch.arange(H, device=device).float() + 0.5) * s
                xc   = (torch.arange(W, device=device).float() + 0.5) * s
                gy, gx = torch.meshgrid(yc, xc, indexing="ij")
                pts.append(torch.stack([gx.reshape(-1), gy.reshape(-1)], 1))
                strd.append(torch.full((H * W,), float(s), device=device))
            self._anchor_cache[key] = (torch.cat(pts, 0), torch.cat(strd, 0))
        return self._anchor_cache[key]

    def _decode(self, reg_logits_flat, ap, strides_flat):
        rm1 = self.reg_max + 1
        if self._proj is None or self._proj.device != ap.device:
            self._proj = torch.arange(rm1, dtype=torch.float32, device=ap.device)
        N    = reg_logits_flat.shape[0]
        dist = reg_logits_flat.view(N, 4, rm1)
        dist = F.softmax(dist, dim=2)
        dist = (dist * self._proj).sum(2) * strides_flat.unsqueeze(1)
        x1 = ap[:, 0] - dist[:, 0]
        y1 = ap[:, 1] - dist[:, 1]
        x2 = ap[:, 0] + dist[:, 2]
        y2 = ap[:, 1] + dist[:, 3]
        return torch.stack([x1, y1, x2, y2], 1)

    def _tal(self, cls_logits, decoded_boxes, gt_xyxy, gt_labels, ap):
        N, M = cls_logits.shape[0], gt_xyxy.shape[0]
        dev  = cls_logits.device
        ax, ay = ap[:, 0], ap[:, 1]
        inside = ((ax.unsqueeze(1) > gt_xyxy[:, 0]) &
                  (ax.unsqueeze(1) < gt_xyxy[:, 2]) &
                  (ay.unsqueeze(1) > gt_xyxy[:, 1]) &
                  (ay.unsqueeze(1) < gt_xyxy[:, 3]))
        if not inside.any():
            return (torch.full((N,), -1, dtype=torch.long, device=dev),
                    torch.zeros(N, dtype=torch.bool, device=dev),
                    torch.zeros(N, M, device=dev))
        with torch.no_grad():
            iou          = _box_iou(decoded_boxes, gt_xyxy)
            gt_cls_score = cls_logits.sigmoid()[:, gt_labels.long()]
            align        = gt_cls_score.pow(self.alpha) * iou.pow(self.beta) * inside.float()
            topk_k       = min(self.topk, N)
            thresh       = align.topk(topk_k, dim=0, largest=True).values[-1:]
            assign       = (align >= thresh) & inside
            align_a      = align * assign.float()
            matched_gt   = align_a.argmax(1)
            is_pos       = assign.any(1)
            matched_gt[~is_pos] = -1
        return matched_gt, is_pos, align

    def forward(self, outputs, bboxes, labels, num_objects, seg_masks):
        det_outs = outputs["det_outputs"]
        semantic = outputs["semantic"]
        dev = semantic.device
        BS  = bboxes.shape[0]
        IS  = float(CFG.img_size)

        ap, sf    = self._get_anchors(det_outs, dev)
        N_total   = ap.shape[0]
        rm1       = self.reg_max + 1

        all_cls = torch.cat([o["cls"].view(BS, self.nc, -1).permute(0, 2, 1)
                             for o in det_outs], 1)
        all_reg = torch.cat([o["reg"].view(BS, 4 * rm1, -1).permute(0, 2, 1)
                             for o in det_outs], 1)

        total_cls = all_cls.sum() * 0.0
        total_box = all_cls.sum() * 0.0
        total_dfl = all_cls.sum() * 0.0
        n_pos     = 0

        for b in range(BS):
            M   = int(num_objects[b].item())
            dec = self._decode(all_reg[b], ap, sf)

            if M == 0:
                neg_tgt  = torch.zeros_like(all_cls[b])
                neg_loss = F.binary_cross_entropy_with_logits(
                    all_cls[b], neg_tgt, reduction="sum") / N_total
                total_cls = total_cls + neg_loss / BS
                continue

            gt_n  = bboxes[b, :M]
            gt_lb = labels[b, :M]
            gt_xy = torch.stack([
                (gt_n[:, 0] - gt_n[:, 2] / 2) * IS,
                (gt_n[:, 1] - gt_n[:, 3] / 2) * IS,
                (gt_n[:, 0] + gt_n[:, 2] / 2) * IS,
                (gt_n[:, 1] + gt_n[:, 3] / 2) * IS,
            ], 1)

            mg, ip, align = self._tal(
                all_cls[b].detach(), dec.detach(), gt_xy, gt_lb, ap)

            if not ip.any():
                neg_tgt  = torch.zeros_like(all_cls[b])
                neg_loss = F.binary_cross_entropy_with_logits(
                    all_cls[b], neg_tgt, reduction="sum") / N_total
                total_cls = total_cls + neg_loss / BS
                continue

            pos_cls     = all_cls[b][ip]
            pos_reg     = all_reg[b][ip]
            pos_ap      = ap[ip]
            pos_sf      = sf[ip]
            pos_gt_idx  = mg[ip]
            pos_gt_box  = gt_xy[pos_gt_idx]
            pos_lb      = gt_lb[pos_gt_idx]

            pos_dec = self._decode(pos_reg, pos_ap, pos_sf)
            pos_iou = _box_iou(pos_dec.detach(), pos_gt_box).diag().clamp(0, 1)

            cls_tgt = torch.zeros(N_total, self.nc, device=dev)
            cls_tgt[ip.nonzero(as_tuple=True)[0], pos_lb] = pos_iou
            align_norm = align[ip, pos_gt_idx]
            align_norm = align_norm / (align_norm.sum() + 1e-8)

            ps  = all_cls[b].sigmoid()
            w   = torch.where(cls_tgt > 0, cls_tgt, 0.75 * ps.detach().pow(2.0))
            bce = F.binary_cross_entropy_with_logits(all_cls[b], cls_tgt, reduction="none")
            total_cls = total_cls + (w * bce).sum() / N_total

            total_box = total_box + (_ciou_loss(pos_dec, pos_gt_box) * align_norm).sum()

            gt_ltrb = torch.stack([
                pos_ap[:, 0] - pos_gt_box[:, 0],
                pos_ap[:, 1] - pos_gt_box[:, 1],
                pos_gt_box[:, 2] - pos_ap[:, 0],
                pos_gt_box[:, 3] - pos_ap[:, 1],
            ], 1) / pos_sf.unsqueeze(1)
            total_dfl = total_dfl + _dfl_loss(pos_reg, gt_ltrb.clamp(0.0, float(self.reg_max)),
                                               self.reg_max)
            n_pos += int(ip.sum())

        norm  = float(max(n_pos, 1))
        cls_l = self.cls_w * total_cls
        box_l = self.box_w * total_box / norm
        dfl_l = self.dfl_w * total_dfl / norm

        mr    = F.interpolate(seg_masks.unsqueeze(1).float(),
                              size=semantic.shape[2:], mode="nearest").squeeze(1).long()
        seg_l = self.seg_w * (self.seg_ce(semantic, mr) + self.seg_dice(semantic, mr))

        total = cls_l + box_l + dfl_l + seg_l
        return {"total": total, "cls": cls_l, "box": box_l, "dfl": dfl_l, "seg": seg_l}

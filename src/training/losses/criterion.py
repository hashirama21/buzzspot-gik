"""
Losses for BuzzSpot detection.

- FocalLoss:         per-class gamma, handles class imbalance
- ciou_loss:         precise bounding-box regression, supports per-element reduction
- HungarianMatcher:  bipartite matching (mirrors DETR)
- BuzzSpotCriterion: combined loss with curriculum instance weighting
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Sigmoid focal loss with optional per-class gamma.

    Args:
        num_classes:      Number of detection classes.
        gamma:            Default focusing parameter.
        alpha:            Balance parameter for positive / negative.
        per_class_gamma:  Dict mapping class index → gamma override.
    """

    def __init__(
        self,
        num_classes: int = 4,
        gamma: float = 2.0,
        alpha: float = 0.25,
        per_class_gamma: Optional[Dict[int, float]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        gammas = torch.full((num_classes,), gamma)
        if per_class_gamma:
            for idx, g in per_class_gamma.items():
                gammas[idx] = g
        self.register_buffer("gammas", gammas)

    def forward(
        self,
        logits: torch.Tensor,          # (B, Q, C) raw logits
        targets: torch.Tensor,         # (B, Q) long class indices, -1 = ignore
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Q, C = logits.shape
        probs = logits.sigmoid()

        valid_mask = targets >= 0
        target_oh = torch.zeros_like(logits)
        target_oh[valid_mask] = F.one_hot(
            targets[valid_mask], num_classes=C
        ).float()

        pt = torch.where(target_oh == 1, probs, 1 - probs)
        gamma_t = self.gammas.view(1, 1, C)
        focal_w = (1 - pt) ** gamma_t

        alpha_t = torch.where(target_oh == 1, self.alpha, 1 - self.alpha)

        bce = F.binary_cross_entropy_with_logits(logits, target_oh, reduction="none")
        loss = alpha_t * focal_w * bce

        if class_weights is not None:
            loss = loss * class_weights.view(1, 1, C)

        loss = loss * valid_mask.float().unsqueeze(-1)
        return loss.sum() / (valid_mask.float().sum() + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# CIoU Loss
# ─────────────────────────────────────────────────────────────────────────────

def ciou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
    reduction: str = "mean",
) -> torch.Tensor:
    """Complete-IoU loss for tight bounding-box regression.

    Args:
        pred:      (N, 4) predicted boxes in normalised XYXY
        target:    (N, 4) ground-truth boxes in normalised XYXY
        reduction: "mean" | "sum" | "none"
    Returns:
        Scalar (mean/sum) or per-element (none) CIoU loss.
    """
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = target.unbind(-1)

    pw, ph = (px2 - px1).clamp(min=0), (py2 - py1).clamp(min=0)
    gw, gh = (gx2 - gx1).clamp(min=0), (gy2 - gy1).clamp(min=0)

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)
    inter_w  = (inter_x2 - inter_x1).clamp(min=0)
    inter_h  = (inter_y2 - inter_y1).clamp(min=0)
    inter    = inter_w * inter_h

    union = pw * ph + gw * gh - inter + eps
    iou   = inter / union

    enc_x1   = torch.min(px1, gx1)
    enc_y1   = torch.min(py1, gy1)
    enc_x2   = torch.max(px2, gx2)
    enc_y2   = torch.max(py2, gy2)
    enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    arctan = torch.atan(gw / (gh + eps)) - torch.atan(pw / (ph + eps))
    v = (4 / (math.pi ** 2)) * arctan ** 2
    with torch.no_grad():
        alpha_ciou = v / (1 - iou + v + eps)

    ciou = iou - rho2 / enc_diag - alpha_ciou * v
    loss = 1 - ciou

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Hungarian Matcher
# ─────────────────────────────────────────────────────────────────────────────

class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground-truth."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return list of (pred_idx, gt_idx) tuples per image."""
        B, Q, _ = outputs["pred_logits"].shape

        pred_probs = outputs["pred_logits"].flatten(0, 1).sigmoid()  # (B*Q, C)
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)              # (B*Q, 4)

        gt_labels = torch.cat([t["labels"] for t in targets])
        gt_boxes  = torch.cat([t["boxes"]  for t in targets])

        cost_cls  = -pred_probs[:, gt_labels]
        cost_l1   = torch.cdist(pred_boxes, gt_boxes, p=1)
        cost_giou = -_batch_giou(pred_boxes, gt_boxes)

        C = (
            self.cost_class * cost_cls
            + self.cost_bbox  * cost_l1
            + self.cost_giou  * cost_giou
        )
        C = C.view(B, Q, -1).cpu()

        sizes = [len(t["labels"]) for t in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, dim=-1)):
            row, col = linear_sum_assignment(c[i].numpy())
            indices.append(
                (torch.as_tensor(row, dtype=torch.long),
                 torch.as_tensor(col, dtype=torch.long))
            )
        return indices


def _batch_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU matrix (N, M) for box sets in XYXY format."""
    x1, y1, x2, y2 = boxes1[:, None].unbind(-1)
    gx1, gy1, gx2, gy2 = boxes2[None].unbind(-1)

    inter_x1 = torch.max(x1, gx1)
    inter_y1 = torch.max(y1, gy1)
    inter_x2 = torch.min(x2, gx2)
    inter_y2 = torch.min(y2, gy2)
    inter_w  = (inter_x2 - inter_x1).clamp(min=0)
    inter_h  = (inter_y2 - inter_y1).clamp(min=0)
    inter    = inter_w * inter_h

    area1 = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area2 = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = area1 + area2 - inter + 1e-6
    iou   = inter / union

    enc_x1 = torch.min(x1, gx1)
    enc_y1 = torch.min(y1, gy1)
    enc_x2 = torch.max(x2, gx2)
    enc_y2 = torch.max(y2, gy2)
    enc    = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + 1e-6

    return iou - (enc - union) / enc


# ─────────────────────────────────────────────────────────────────────────────
# Combined criterion
# ─────────────────────────────────────────────────────────────────────────────

class BuzzSpotCriterion(nn.Module):
    """Master loss combining focal, CIoU, and attribute losses.

    Instance-level curriculum weighting is applied per matched pair so that
    blurred or occluded instances contribute more to the box regression loss.
    """

    def __init__(
        self,
        num_classes: int = 4,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        per_class_gamma: Optional[Dict[int, float]] = None,
        bbox_weight: float = 5.0,
        attr_weight: float = 0.5,
        blur_instance_weight: float = 2.0,
        occlusion_instance_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.bbox_weight = bbox_weight
        self.attr_weight = attr_weight
        self.blur_w = blur_instance_weight
        self.occ_w  = occlusion_instance_weight

        self.matcher = HungarianMatcher()
        self.focal   = FocalLoss(
            num_classes, focal_gamma, focal_alpha,
            per_class_gamma=per_class_gamma,
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)

        pred_boxes_matched, gt_boxes_matched = [], []
        pred_logits_flat, gt_labels_flat     = [], []
        instance_weights = []

        for b, (pred_idx, gt_idx) in enumerate(indices):
            pred_boxes_matched.append(outputs["pred_boxes"][b][pred_idx])
            gt_boxes_matched.append(targets[b]["boxes"][gt_idx])
            gt_labels_flat.append(targets[b]["labels"][gt_idx])
            pred_logits_flat.append(outputs["pred_logits"][b][pred_idx])

            blur = targets[b].get("blur",     torch.zeros(len(gt_idx)))[gt_idx]
            occ  = targets[b].get("occlusion", torch.zeros(len(gt_idx)))[gt_idx]
            w = torch.ones(len(gt_idx), device=blur.device)
            w += (self.blur_w - 1) * blur.float()
            w += (self.occ_w  - 1) * occ.float()
            instance_weights.append(w)

        # ── CIoU loss with per-instance weighting ─────────────
        if pred_boxes_matched:
            pred_boxes_m = torch.cat(pred_boxes_matched)
            gt_boxes_m   = torch.cat(gt_boxes_matched)
            inst_w       = torch.cat(instance_weights)
            # reduction="none" → (N,) so we can weight each instance
            per_elem = ciou_loss(pred_boxes_m, gt_boxes_m, reduction="none")
            loss_ciou = (per_elem * inst_w).sum() / (inst_w.sum() + 1e-6)
        else:
            loss_ciou = outputs["pred_boxes"].sum() * 0.0

        # ── Focal classification loss ──────────────────────────
        B, Q, C = outputs["pred_logits"].shape
        query_targets = torch.full(
            (B, Q), -1, dtype=torch.long,
            device=outputs["pred_logits"].device,
        )
        for b, (pred_idx, gt_idx) in enumerate(indices):
            query_targets[b, pred_idx] = targets[b]["labels"][gt_idx].to(
                query_targets.device
            )

        loss_cls = self.focal(outputs["pred_logits"], query_targets)

        # ── Attribute loss ────────────────────────────────────
        loss_attr = outputs["pred_boxes"].sum() * 0.0
        if "pred_attributes" in outputs and pred_boxes_matched:
            pred_attr_m = torch.cat(
                [outputs["pred_attributes"][b][pred_idx]
                 for b, (pred_idx, _) in enumerate(indices)]
            )
            gt_blur = torch.cat(
                [targets[b]["blur"][gt_idx] for b, (_, gt_idx) in enumerate(indices)]
            )
            gt_occ = torch.cat(
                [targets[b]["occlusion"][gt_idx] for b, (_, gt_idx) in enumerate(indices)]
            )
            gt_attr = torch.stack([gt_blur, gt_occ], dim=1).to(pred_attr_m.device)
            loss_attr = F.binary_cross_entropy_with_logits(pred_attr_m, gt_attr)

        total = loss_cls + self.bbox_weight * loss_ciou + self.attr_weight * loss_attr

        return {
            "loss_total": total,
            "loss_cls":   loss_cls,
            "loss_ciou":  loss_ciou,
            "loss_attr":  loss_attr,
        }

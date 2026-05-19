"""
BuzzSpotEvaluator — COCO-style mAP with per-class, per-size,
per-attribute (blur / occlusion) breakdowns.

Primary metric: mAP@0.5:0.95 (competition standard).
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

log = logging.getLogger(__name__)


class BuzzSpotEvaluator:
    """Accumulates model predictions and runs COCO evaluation.

    The COCO annotation JSON is loaded once at construction and reused
    across epochs.  Call reset() at the start of each validation epoch.

    Args:
        ann_file:     Path to COCO-format annotation JSON.
        class_names:  Ordered list of class names (index 0 → category_id 1).
        score_thresh: Minimum score to include a prediction.
    """

    IDX_TO_CAT_ID = {0: 1, 1: 2, 2: 3, 3: 4}

    def __init__(
        self,
        ann_file: str,
        class_names: List[str],
        score_thresh: float = 0.01,
    ) -> None:
        self.ann_file     = ann_file
        self.class_names  = class_names
        self.score_thresh = score_thresh

        self.coco_gt = COCO(ann_file)
        self._predictions: List[Dict] = []
        self._image_ids:   List[int]  = []

    def reset(self) -> None:
        self._predictions = []
        self._image_ids   = []

    # ──────────────────────────────────────────────────────────
    # Accumulate
    # ──────────────────────────────────────────────────────────

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        """Convert model outputs to COCO result format and accumulate."""
        pred_logits = outputs["pred_logits"]  # (B, Q, C)
        pred_boxes  = outputs["pred_boxes"]   # (B, Q, 4) normalised XYXY
        B = pred_logits.shape[0]

        for i in range(B):
            img_id = targets[i]["image_id"].item()
            orig_H, orig_W = targets[i]["orig_size"].tolist()

            scores, labels = pred_logits[i].sigmoid().max(dim=-1)
            boxes = pred_boxes[i]

            keep   = scores >= self.score_thresh
            scores = scores[keep]
            labels = labels[keep]
            boxes  = boxes[keep]

            # Normalised XYXY → absolute XYWH (COCO format)
            abs_boxes = boxes.clone().cpu()
            abs_boxes[:, [0, 2]] *= orig_W
            abs_boxes[:, [1, 3]] *= orig_H
            abs_boxes[:, 2] -= abs_boxes[:, 0]
            abs_boxes[:, 3] -= abs_boxes[:, 1]

            for score, label, box in zip(
                scores.cpu().tolist(),
                labels.cpu().tolist(),
                abs_boxes.tolist(),
            ):
                cat_id = self.IDX_TO_CAT_ID.get(int(label))
                if cat_id is None:
                    continue
                self._predictions.append(
                    {
                        "image_id":    img_id,
                        "category_id": cat_id,
                        "bbox":        [round(v, 2) for v in box],
                        "score":       round(float(score), 4),
                    }
                )

            if img_id not in self._image_ids:
                self._image_ids.append(img_id)

    # ──────────────────────────────────────────────────────────
    # Compute metrics
    # ──────────────────────────────────────────────────────────

    def summarize(self) -> Dict[str, float]:
        """Run COCO eval and return metrics dict (COCO stdout suppressed)."""
        if not self._predictions:
            return {
                "mAP_50_95": 0.0, "mAP_50": 0.0,
                "AP_small":  0.0, "AP_medium": 0.0,
            }

        coco_dt   = self.coco_gt.loadRes(self._predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = self._image_ids
        self._run_eval(coco_eval)

        stats   = coco_eval.stats
        metrics = {
            "mAP_50_95": float(stats[0]),
            "mAP_50":    float(stats[1]),
            "mAP_75":    float(stats[2]),
            "AP_small":  float(stats[3]),
            "AP_medium": float(stats[4]),
            "AR_max1":   float(stats[6]),
            "AR_max10":  float(stats[7]),
        }

        # Per-class breakdown
        for cls_idx, cls_name in enumerate(self.class_names):
            cat_id = self.IDX_TO_CAT_ID.get(cls_idx)
            if cat_id is None:
                continue
            coco_eval.params.catIds = [cat_id]
            self._run_eval(coco_eval)
            metrics[f"AP_{cls_name}_50_95"] = float(coco_eval.stats[0])
            metrics[f"AP_{cls_name}_50"]    = float(coco_eval.stats[1])

        # Per-attribute breakdown (exclusive subsets)
        for attr_name in ("blur", "occlusion"):
            ids_0, ids_1 = self._exclusive_attribute_split(attr_name)
            for label, img_ids in (("0", ids_0), ("1", ids_1)):
                preds = [p for p in self._predictions if p["image_id"] in img_ids]
                if preds:
                    metrics[f"AP_{attr_name}_{label}"] = self._eval_subset(
                        preds, list(img_ids)
                    )

        return metrics

    # ──────────────────────────────────────────────────────────
    # Submission format
    # ──────────────────────────────────────────────────────────

    def to_submission(self, out_path: str) -> str:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self._predictions, f)
        log.info("Saved %d predictions → %s", len(self._predictions), out)
        return str(out)

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _run_eval(coco_eval: COCOeval) -> None:
        """Run evaluate/accumulate/summarize with COCO's stdout suppressed."""
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

    def _exclusive_attribute_split(self, attr_name: str):
        """Split evaluation images into two EXCLUSIVE subsets.

        subset_1: images where ANY annotation has attr=1  (challenging)
        subset_0: images where NO  annotation has attr=1  (clean)
        Mixed images (both attr=0 and attr=1 anns) go to subset_1.
        """
        all_anns = self.coco_gt.loadAnns(self.coco_gt.getAnnIds())
        ids_with_attr1: Set[int] = {
            a["image_id"]
            for a in all_anns
            if a.get("attributes", {}).get(attr_name, 0) == 1
        }
        eval_ids = set(self._image_ids)
        ids_1 = ids_with_attr1 & eval_ids
        ids_0 = eval_ids - ids_1
        return ids_0, ids_1

    def _eval_subset(self, preds: List[Dict], img_ids: List[int]) -> float:
        """Run COCO eval on a subset of images; return mAP@0.5:0.95."""
        if not preds:
            return 0.0
        try:
            coco_dt = self.coco_gt.loadRes(preds)
            ev      = COCOeval(self.coco_gt, coco_dt, "bbox")
            ev.params.imgIds = img_ids
            self._run_eval(ev)
            return float(ev.stats[0])
        except Exception as exc:
            log.debug("Subset eval failed: %s", exc)
            return 0.0
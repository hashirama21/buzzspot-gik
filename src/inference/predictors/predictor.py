"""
Inference pipeline: tiling + TTA + WBF ensemble + temporal context.

Steps:
1. Load full 1920×1080 frame + 5 context frames.
2. Encode context frames once → (1, T-1, N, d_model) features.
3. Slice keyframe into overlapping 256×256 tiles.
4. Run detector on each tile (optionally with TTA), injecting context features.
5. Invert TTA transforms on predicted boxes.
6. Map tile predictions back to full-image coordinates.
7. Merge all tile predictions with Weighted Boxes Fusion (WBF).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.amp import autocast

from src.data.preprocessing.tiling import TileExtractor
from src.data.preprocessing.temporal import TemporalFrameLoader
from src.data.augmentations.transforms import (
    IMAGENET_MEAN, IMAGENET_STD,
    TTATransform, get_tta_transforms,
    to_normalized_tensor,
)

log = logging.getLogger(__name__)


# Weighted Boxes Fusion

def weighted_boxes_fusion(
    boxes_list:   List[np.ndarray],
    scores_list:  List[np.ndarray],
    labels_list:  List[np.ndarray],
    weights:      Optional[List[float]] = None,
    iou_thr:      float = 0.55,
    skip_box_thr: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted Boxes Fusion (pure numpy).

    For production, swap with: `from ensemble_boxes import weighted_boxes_fusion`
    """
    if not boxes_list:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    if weights is None:
        weights = [1.0] * len(boxes_list)

    all_boxes  = np.vstack(boxes_list)
    all_scores = np.hstack(scores_list)
    all_labels = np.hstack(labels_list)

    keep = all_scores >= skip_box_thr
    all_boxes, all_scores, all_labels = all_boxes[keep], all_scores[keep], all_labels[keep]

    if len(all_boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    order = np.argsort(-all_scores)
    all_boxes  = all_boxes[order]
    all_scores = all_scores[order]
    all_labels = all_labels[order]

    merged_boxes, merged_scores, merged_labels = [], [], []
    used = np.zeros(len(all_boxes), dtype=bool)

    for i in range(len(all_boxes)):
        if used[i]:
            continue
        ious       = _iou_batch(all_boxes[i : i + 1], all_boxes)[0]
        same_class = all_labels == all_labels[i]
        overlap    = (ious > iou_thr) & same_class & ~used
        overlap[i] = True

        cluster_scores = all_scores[overlap]
        w = cluster_scores / cluster_scores.sum()
        merged_boxes.append((all_boxes[overlap] * w[:, None]).sum(axis=0))
        merged_scores.append(cluster_scores.max())
        merged_labels.append(all_labels[i])
        used[overlap] = True

    if not merged_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    return (
        np.stack(merged_boxes),
        np.array(merged_scores),
        np.array(merged_labels, dtype=int),
    )


def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None])
    iy1 = np.maximum(ay1[:, None], by1[None])
    ix2 = np.minimum(ax2[:, None], bx2[None])
    iy2 = np.minimum(ay2[:, None], by2[None])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None] - inter + 1e-6
    return inter / union


# Box inversion helpers

def _invert_transform(boxes: np.ndarray, tta_name: str) -> np.ndarray:
    """Invert TTA spatial transforms on normalised XYXY boxes."""
    b = boxes.copy()
    if "hflip" in tta_name:
        b[:, [0, 2]] = 1.0 - b[:, [2, 0]]
    if "vflip" in tta_name:
        b[:, [1, 3]] = 1.0 - b[:, [3, 1]]
    return b


# Main Predictor

class BuzzSpotPredictor:
    """Full inference pipeline: temporal context + tiling + TTA + WBF."""

    IDX_TO_CAT_ID = {0: 1, 1: 2, 2: 3, 3: 4}

    def __init__(
        self,
        model: torch.nn.Module,
        cfg,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model  = model
        self.cfg    = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = self.device.type == "cuda"

        self.model.to(self.device)
        self.model.eval()

        inf_cfg = cfg.inference

        self.tile_extractor = (
            TileExtractor(
                size=inf_cfg.tiling.tile_size,
                overlap=inf_cfg.tiling.overlap,
            )
            if inf_cfg.tiling.enabled
            else None
        )

        # Load frames at full resolution; tiling happens in predict()
        self.frame_loader = TemporalFrameLoader(
            img_dir=Path(cfg.data.root),
            num_frames=cfg.data.temporal.num_frames,
            strategy=cfg.data.temporal.strategy,
            img_size=None,
        )

        # Size for encoding context frames (tile_size for consistency)
        self._ctx_size   = inf_cfg.tiling.tile_size
        self.tta_list: List[TTATransform] = (
            get_tta_transforms(tile_size=inf_cfg.tiling.tile_size)
            if inf_cfg.tta.enabled
            else [TTATransform("original", __import__("albumentations").Compose([]))]
        )

        self.score_thr = inf_cfg.confidence_threshold
        self.wbf_iou   = inf_cfg.tiling.merge_iou_threshold


    @torch.no_grad()
    def predict(self, keyframe_path: str, image_id: int) -> List[Dict[str, Any]]:
        """Run full inference on a single keyframe."""
        kf_path = Path(keyframe_path)
        frames  = self.frame_loader.load(kf_path)
        keyframe = frames[-1]         # HWC uint8, full resolution
        H, W    = keyframe.shape[:2]

        # Encode temporal context once, shared across all tiles
        context_features = self._encode_context(frames)

        tile_coords = (
            self.tile_extractor.get_tile_grid(H, W)
            if self.tile_extractor is not None
            else [(0, 0, W, H)]
        )

        all_boxes, all_scores, all_labels = [], [], []

        for (x0, y0, x1, y1) in tile_coords:
            tile = keyframe[y0:y1, x0:x1]
            t_boxes, t_scores, t_labels = self._run_tta(tile, context_features)

            if len(t_boxes) > 0:
                remapped = self._remap_to_full(
                    t_boxes, x0, y0, x1 - x0, y1 - y0, H, W
                )
                all_boxes.append(remapped)
                all_scores.append(t_scores)
                all_labels.append(t_labels)

        if all_boxes:
            merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
                all_boxes, all_scores, all_labels, iou_thr=self.wbf_iou
            )
        else:
            merged_boxes  = np.zeros((0, 4))
            merged_scores = np.zeros(0)
            merged_labels = np.zeros(0, dtype=int)

        return self._to_coco_results(merged_boxes, merged_scores, merged_labels, image_id, H, W)

    # Temporal context

    def _encode_context(
        self, frames: List[np.ndarray]
    ) -> Optional[torch.Tensor]:
        """Encode context frames → (1, T-1, N, d_model), or None."""
        if not hasattr(self.model, "encode_context_frames") or len(frames) <= 1:
            return None
        if self.model.temporal_fusion is None:
            return None

        context_frames = frames[:-1]
        tensors = []
        for f in context_frames:
            resized = cv2.resize(f, (self._ctx_size, self._ctx_size),
                                 interpolation=cv2.INTER_LINEAR)
            tensors.append(to_normalized_tensor(resized))

        ctx = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)  # (1, T-1, 3, H, W)
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            return self.model.encode_context_frames(ctx)

    # TTA

    def _run_tta(
        self,
        tile: np.ndarray,
        context_features: Optional[torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_boxes, all_scores, all_labels = [], [], []

        for tta in self.tta_list:
            aug_tile = tta.transform(image=tile)["image"]   # numpy HWC
            tensor   = to_normalized_tensor(aug_tile).unsqueeze(0).to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(tensor, context_features=context_features)

            boxes, scores, labels = self._decode_output(out, 0)

            # Invert spatial augmentation on predicted boxes
            if len(boxes) > 0:
                boxes = _invert_transform(boxes, tta.name)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return weighted_boxes_fusion(all_boxes, all_scores, all_labels)

    def _decode_output(
        self,
        out: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits = out["pred_logits"][batch_idx]
        boxes  = out["pred_boxes"][batch_idx]

        scores, labels = logits.sigmoid().max(dim=-1)
        keep = scores >= self.score_thr

        return (
            boxes[keep].cpu().numpy(),
            scores[keep].cpu().numpy(),
            labels[keep].cpu().numpy(),
        )

    # Helpers

    @staticmethod
    def _remap_to_full(
        boxes: np.ndarray,
        x0: int, y0: int, tw: int, th: int,
        full_H: int, full_W: int,
    ) -> np.ndarray:
        b = boxes.copy()
        b[:, [0, 2]] = (b[:, [0, 2]] * tw + x0) / full_W
        b[:, [1, 3]] = (b[:, [1, 3]] * th + y0) / full_H
        return np.clip(b, 0.0, 1.0)

    def _to_coco_results(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        image_id: int,
        H: int,
        W: int,
    ) -> List[Dict]:
        results = []
        for box, score, label in zip(boxes, scores, labels):
            cat_id = self.IDX_TO_CAT_ID.get(int(label))
            if cat_id is None:
                continue
            x1, y1, x2, y2 = box
            results.append(
                {
                    "image_id":    image_id,
                    "category_id": cat_id,
                    "bbox": [
                        round(float(x1 * W), 2),
                        round(float(y1 * H), 2),
                        round(float((x2 - x1) * W), 2),
                        round(float((y2 - y1) * H), 2),
                    ],
                    "score": round(float(score), 4),
                }
            )
        return results

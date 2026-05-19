"""
Pseudo-labeling pipeline for the BuzzSpot test set.

Strategy:
1. Run Grounding DINO with text prompts ("bee", "bumblebee", "hoverfly", "moth")
   on unlabelled test images to get coarse bounding boxes.
2. Refine masks with SAM 2 for precise segmentation.
3. Filter by confidence threshold.
4. Write a pseudo-annotation JSON in COCO format.
5. Self-train the detection model for N epochs on pseudo-labelled data.

Prerequisites (install once):
    pip install groundingdino-py segment-anything-2

Usage:
    python scripts/pseudo_label.py \
        --test-dir data/buzzset/test/ \
        --ann     data/buzzset/annotations/test.json \
        --out     data/buzzset/annotations/pseudo_train.json \
        --gdino-ckpt weights/groundingdino_swint_ogc.pth \
        --sam2-ckpt  weights/sam2_hiera_large.pt
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Grounding DINO wrapper
# ─────────────────────────────────────────────────────────────────────────────

class GroundingDINOWrapper:
    """Thin wrapper around the official GroundingDINO API."""

    PROMPTS = {
        "bee":        1,
        "bumblebee":  2,
        "hoverfly":   3,
        "moth":       4,
    }

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        device: str = "cuda",
    ) -> None:
        self.box_threshold  = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self._model = self._load(config_path, checkpoint_path)

    def _load(self, config_path: str, ckpt: str):
        try:
            from groundingdino.util.inference import load_model
            return load_model(config_path, ckpt, device=self.device)
        except ImportError:
            print("[PseudoLabel] groundingdino not installed — using stub.")
            return None

    def predict(
        self, image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Returns (boxes_xyxy_norm, scores, phrases)."""
        if self._model is None:
            return np.zeros((0, 4)), np.zeros(0), []

        try:
            from groundingdino.util.inference import predict as gd_predict
            import torchvision.transforms as T

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tfm = T.Compose([
                T.ToPILImage(),
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            img_tensor = tfm(image_rgb)

            prompt = " . ".join(self.PROMPTS.keys())
            boxes, logits, phrases = gd_predict(
                model=self._model,
                image=img_tensor,
                caption=prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
            return boxes.numpy(), logits.numpy(), phrases
        except Exception as e:
            print(f"[GroundingDINO] Error: {e}")
            return np.zeros((0, 4)), np.zeros(0), []


# ─────────────────────────────────────────────────────────────────────────────
# SAM 2 mask refiner
# ─────────────────────────────────────────────────────────────────────────────

class SAM2MaskRefiner:
    """Uses SAM 2 to refine coarse G-DINO boxes into precise masks."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        self.device = device
        self._predictor = self._load(checkpoint_path)

    def _load(self, ckpt: str):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2 = build_sam2("sam2_hiera_l.yaml", ckpt, device=self.device)
            return SAM2ImagePredictor(sam2)
        except ImportError:
            print("[PseudoLabel] SAM 2 not installed — using box-as-mask fallback.")
            return None

    def get_tight_boxes(
        self,
        image_rgb: np.ndarray,
        boxes_xyxy: np.ndarray,   # pixel coordinates
    ) -> np.ndarray:
        """Return tight bounding boxes from SAM 2 masks (or original boxes)."""
        if self._predictor is None or len(boxes_xyxy) == 0:
            return boxes_xyxy

        try:
            self._predictor.set_image(image_rgb)
            masks, _, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_xyxy,
                multimask_output=False,
            )
            # Derive tight bounding box from each mask
            tight_boxes = []
            for mask in masks:
                ys, xs = np.where(mask[0] > 0.5)
                if len(xs) == 0:
                    tight_boxes.append(boxes_xyxy[len(tight_boxes)])
                    continue
                tight_boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
            return np.array(tight_boxes, dtype=np.float32)
        except Exception as e:
            print(f"[SAM2] Error: {e}")
            return boxes_xyxy


# ─────────────────────────────────────────────────────────────────────────────
# Full pseudo-labeling pipeline
# ─────────────────────────────────────────────────────────────────────────────

class PseudoLabelGenerator:
    """Generates COCO-format pseudo-annotations for unlabelled test images."""

    CAT_NAME_TO_ID = {"bee": 1, "bumblebee": 2, "hoverfly": 3, "moth": 4}

    def __init__(
        self,
        gdino: GroundingDINOWrapper,
        sam2: SAM2MaskRefiner,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.gdino = gdino
        self.sam2  = sam2
        self.conf  = confidence_threshold

    def run(
        self,
        test_ann_path: str,
        img_dir: str,
        out_path: str,
    ) -> None:
        """Process all test images and write pseudo-annotation JSON."""
        with open(test_ann_path) as f:
            test_coco = json.load(f)

        images = test_coco["images"]
        pseudo_anns = []
        ann_id = 1

        for img_meta in images:
            img_path = Path(img_dir) / img_meta["file_name"]
            if not img_path.exists():
                print(f"[PseudoLabel] Missing: {img_path}")
                continue

            bgr = cv2.imread(str(img_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            H, W = bgr.shape[:2]

            # 1. Grounding DINO detection
            boxes_norm, scores, phrases = self.gdino.predict(bgr)

            if len(boxes_norm) == 0:
                continue

            # Filter confidence
            keep = scores >= self.conf
            boxes_norm = boxes_norm[keep]
            scores     = scores[keep]
            phrases    = [p for i, p in enumerate(phrases) if keep[i]]

            # Convert normalised cx,cy,w,h → pixel XYXY
            boxes_px = self._cxcywh_to_xyxy_px(boxes_norm, H, W)

            # 2. SAM 2 mask refinement
            tight_boxes = self.sam2.get_tight_boxes(rgb, boxes_px)

            # 3. Build COCO annotations
            for box_px, score, phrase in zip(tight_boxes, scores, phrases):
                cat_name = self._match_category(phrase)
                if cat_name is None:
                    continue
                x1, y1, x2, y2 = box_px
                w, h = x2 - x1, y2 - y1
                if w < 4 or h < 4:
                    continue
                pseudo_anns.append({
                    "id":          ann_id,
                    "image_id":    img_meta["id"],
                    "category_id": self.CAT_NAME_TO_ID[cat_name],
                    "bbox":        [round(float(x1), 2), round(float(y1), 2),
                                    round(float(w),  2), round(float(h),  2)],
                    "area":        float(w * h),
                    "iscrowd":     0,
                    "score":       round(float(score), 4),
                    "attributes":  {"blur": 0, "occlusion": 0},
                })
                ann_id += 1

        # Build output COCO JSON
        out_coco = {
            "images":     images,
            "annotations": pseudo_anns,
            "categories": [
                {"id": v, "name": k}
                for k, v in self.CAT_NAME_TO_ID.items()
            ],
        }

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out_coco, f)

        print(f"[PseudoLabel] Generated {len(pseudo_anns)} annotations → {out_path}")

    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _cxcywh_to_xyxy_px(
        boxes_norm: np.ndarray, H: int, W: int
    ) -> np.ndarray:
        cx, cy, bw, bh = boxes_norm[:, 0], boxes_norm[:, 1], boxes_norm[:, 2], boxes_norm[:, 3]
        x1 = np.clip((cx - bw / 2) * W, 0, W)
        y1 = np.clip((cy - bh / 2) * H, 0, H)
        x2 = np.clip((cx + bw / 2) * W, 0, W)
        y2 = np.clip((cy + bh / 2) * H, 0, H)
        return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    def _match_category(self, phrase: str) -> Optional[str]:
        phrase = phrase.lower().strip()
        for name in self.CAT_NAME_TO_ID:
            if name in phrase:
                return name
        if "bumble" in phrase:
            return "bumblebee"
        if "hover" in phrase:
            return "hoverfly"
        return None

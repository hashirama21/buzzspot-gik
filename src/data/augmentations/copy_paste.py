"""
SAM2CopyPaste — uses pre-extracted SAM 2 segmentation masks to
perform copy-paste augmentation, boosting rare pollinator classes.

Workflow (offline, run once before training):
    python scripts/extract_sam2_masks.py \
        --ann data/buzzset/annotations/train.json \
        --img-dir data/buzzset/train \
        --out-dir data/sam2_masks/

At training time, CopyPasteAugmentor pastes random instances from the
mask library onto training images.  It is called AFTER albumentations
spatial transforms (both inputs/outputs are numpy uint8 + COCO XYWH bboxes).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class CopyPasteAugmentor:
    """Paste insect instances from a SAM 2 mask library onto target images.

    Args:
        masks_dir:      Directory with per-instance RGBA crops (output of
                        extract_sam2_masks.py).
        class_names:    List of class names ordered by index.
        target_classes: Classes to boost (e.g. ["moth", "hoverfly"]).
        prob:           Probability of applying any paste at all.
        max_instances:  Maximum insect instances to paste per image.
        scale_range:    Relative scale jitter for pasted instances.
        blend_alpha:    Alpha blending (True) or hard mask paste (False).
    """

    def __init__(
        self,
        masks_dir: str,
        class_names: List[str],
        target_classes: Optional[List[str]] = None,
        prob: float = 0.5,
        max_instances: int = 3,
        scale_range: Tuple[float, float] = (0.8, 1.3),
        blend_alpha: bool = True,
    ) -> None:
        self.masks_dir      = Path(masks_dir)
        self.class_names    = class_names
        self.target_classes = set(target_classes or class_names)
        self.prob           = prob
        self.max_instances  = max_instances
        self.scale_range    = scale_range
        self.blend_alpha    = blend_alpha
        self._library       = self._build_library()

    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[Tuple],
        labels: List[int],
        attributes: List[Dict],
    ) -> Tuple[np.ndarray, List[Tuple], List[int], List[Dict]]:
        """Apply copy-paste augmentation.  Returns updated (image, bboxes, labels, attributes)."""
        if random.random() > self.prob:
            return image, bboxes, labels, attributes

        bboxes     = list(bboxes)
        labels     = list(labels)
        attributes = list(attributes)

        n_paste = random.randint(1, self.max_instances)
        for _ in range(n_paste):
            cls_name = random.choice(list(self.target_classes))
            if cls_name not in self._library or not self._library[cls_name]:
                continue

            cls_idx = self.class_names.index(cls_name)
            rgb, alpha_mask = self._load_random_instance(cls_name)
            if rgb is None:
                continue

            image, new_box = self._paste(image, rgb, alpha_mask)
            if new_box is not None:
                bboxes.append(new_box)
                labels.append(cls_idx)
                attributes.append({
                    "blur": 0, "occlusion": 0,
                    "area": new_box[2] * new_box[3],
                    "ann_id": -1,  # synthetic instance
                })

        return image, bboxes, labels, attributes


    def _build_library(self) -> Dict[str, List[Path]]:
        library: Dict[str, List[Path]] = {}
        if not self.masks_dir.exists():
            return library
        for cls_dir in self.masks_dir.iterdir():
            if cls_dir.is_dir():
                library[cls_dir.name] = list(cls_dir.glob("*.png"))
        return library

    def _load_random_instance(
        self, cls_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        paths = self._library.get(cls_name, [])
        if not paths:
            return None, None
        path = random.choice(paths)
        img  = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None
        if img.ndim == 2 or img.shape[2] == 3:
            alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = np.concatenate([img, alpha], axis=2)
        rgb   = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = img[:, :, 3]
        return rgb, alpha

    def _paste(
        self,
        image: np.ndarray,
        rgb: np.ndarray,
        alpha_mask: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        H, W = image.shape[:2]
        scale = random.uniform(*self.scale_range)
        new_h = max(4, int(rgb.shape[0] * scale))
        new_w = max(4, int(rgb.shape[1] * scale))

        rgb        = cv2.resize(rgb,        (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        alpha_mask = cv2.resize(alpha_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if new_h >= H or new_w >= W:
            return image, None

        x0 = random.randint(0, W - new_w)
        y0 = random.randint(0, H - new_h)

        result  = image.copy()
        roi     = result[y0:y0 + new_h, x0:x0 + new_w]
        alpha_f = alpha_mask.astype(np.float32) / 255.0

        if self.blend_alpha:
            blended = (
                rgb.astype(np.float32) * alpha_f[..., None]
                + roi.astype(np.float32) * (1 - alpha_f[..., None])
            )
            result[y0:y0 + new_h, x0:x0 + new_w] = blended.astype(np.uint8)
        else:
            mask_bool = alpha_f > 0.5
            roi[mask_bool] = rgb[mask_bool]
            result[y0:y0 + new_h, x0:x0 + new_w] = roi

        bbox = (float(x0), float(y0), float(new_w), float(new_h))
        return result, bbox

"""
BuzzSet dataset — temporal multi-frame support.

Folder layout expected:
    data/buzzset/
    ├── annotations/
    │   ├── train.json
    │   ├── valid.json
    │   └── test.json
    └── {train,valid,test}/
        └── video_N/
            ├── video_N_000001.jpg   <- context frame t-5
            ├── ...
            └── video_N_000006.jpg   <- keyframe (annotated)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing.tiling import TileExtractor
from src.data.preprocessing.temporal import TemporalFrameLoader
from src.data.augmentations.copy_paste import CopyPasteAugmentor


class BuzzSetDataset(Dataset):
    """COCO-style BuzzSet dataset with temporal context.

    Transform pipeline (training):
        albumentations spatial  → numpy uint8
        copy-paste augmentor    → numpy uint8
        TemporalFrameLoader     → CHW float32 (ImageNet-normalized)

    Args:
        ann_file:           Path to COCO-format annotation JSON.
        img_dir:            Root directory containing video subfolders.
        transforms:         Albumentations spatial pipeline (numpy in, numpy out).
        num_frames:         Total frames per sample (1 = no temporal, 6 = full).
        temporal_strategy:  "stack" | "diff" | "flow" | "none".
        tile_cfg:           TileExtractor config dict or None.
        split:              "train" | "val" | "test".
        copy_paste_aug:     Optional CopyPasteAugmentor for rare-class boosting.
    """

    CLASSES       = ["bee", "bumblebee", "hoverfly", "moth"]
    CAT_ID_TO_IDX = {1: 0, 2: 1, 3: 2, 4: 3}
    IDX_TO_CAT_ID = {v: k for k, v in CAT_ID_TO_IDX.items()}

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        transforms=None,
        num_frames: int = 6,
        temporal_strategy: str = "stack",
        tile_cfg: Optional[Dict] = None,
        split: str = "train",
        copy_paste_aug: Optional[CopyPasteAugmentor] = None,
    ) -> None:
        self.img_dir           = Path(img_dir)
        self.transforms        = transforms
        self.num_frames        = num_frames
        self.temporal_strategy = temporal_strategy
        self.split             = split
        self.is_test           = split == "test"
        self.copy_paste_aug    = copy_paste_aug
        self._curriculum_phase = 1  # 1 = clean only, 2 = all samples

        with open(ann_file) as f:
            coco = json.load(f)

        self.images: Dict[int, Dict] = {img["id"]: img for img in coco["images"]}
        self.categories: Dict[int, Dict] = {
            cat["id"]: cat for cat in coco.get("categories", [])
        }

        self.annotations: Dict[int, List[Dict]] = {img_id: [] for img_id in self.images}
        for ann in coco.get("annotations", []):
            self.annotations[ann["image_id"]].append(ann)

        self._all_img_ids: List[int] = list(self.images.keys())
        self.img_ids: List[int]      = list(self._all_img_ids)

        self.tile_extractor = TileExtractor(**tile_cfg) if tile_cfg else None

        self.frame_loader = TemporalFrameLoader(
            img_dir=self.img_dir,
            num_frames=num_frames,
            strategy=temporal_strategy,
        )

    # Curriculum interface

    def set_curriculum_phase(self, phase: int) -> None:
        """Switch between curriculum phases.

        Phase 1: exclude images where ALL annotations have blur=1 or occlusion=1.
        Phase 2: all images, weighted higher for degraded instances (done in loss).
        """
        self._curriculum_phase = phase
        if phase == 1:
            self.img_ids = [
                img_id for img_id in self._all_img_ids
                if self._is_clean(img_id)
            ]
        else:
            self.img_ids = list(self._all_img_ids)

    def _is_clean(self, img_id: int) -> bool:
        """True if the image has NO blurred or occluded annotations."""
        for ann in self.annotations.get(img_id, []):
            attrs = ann.get("attributes", {})
            if attrs.get("blur", 0) or attrs.get("occlusion", 0):
                return False
        return True

    # Core interface

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Any:
        img_id   = self.img_ids[idx]
        img_meta = self.images[img_id]
        anns     = self.annotations[img_id]

        img_path = self.img_dir / img_meta["file_name"]
        frames   = self.frame_loader.load(img_path)

        keyframe = frames[-1]  # HWC uint8
        bboxes, labels, ann_ids, attributes = self._parse_annotations(anns)

        if self.transforms and not self.is_test:
            result   = self.transforms(
                image=keyframe,
                bboxes=bboxes,
                class_labels=labels,
                ann_ids=ann_ids,
            )
            keyframe = result["image"]           # still numpy uint8
            bboxes   = list(result["bboxes"])
            labels   = list(result["class_labels"])
            # Filter attributes to only surviving annotation IDs
            surviving = set(result["ann_ids"])
            attributes = [a for a in attributes if a["ann_id"] in surviving]
            frames[-1] = keyframe

        if self.copy_paste_aug is not None and not self.is_test:
            keyframe, bboxes, labels, attributes = self.copy_paste_aug(
                keyframe, bboxes, labels, attributes
            )
            frames[-1] = keyframe

        tensor = self.frame_loader.build_tensor(frames)

        target = self._build_target(img_id, bboxes, labels, attributes, img_meta)

        if self.tile_extractor is not None:
            return self.tile_extractor.extract(tensor, target)

        return {"image": tensor, "target": target, "img_id": img_id}

    # Helpers

    def _parse_annotations(
        self, anns: List[Dict]
    ) -> Tuple[List, List[int], List[int], List[Dict]]:
        """Extract bboxes (COCO XYWH), labels, ann_ids, and attribute dicts."""
        bboxes, labels, ann_ids, attributes = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes.append((x, y, w, h))
            labels.append(self.CAT_ID_TO_IDX[ann["category_id"]])
            ann_ids.append(ann["id"])
            attributes.append(
                {
                    "blur":      ann.get("attributes", {}).get("blur", 0),
                    "occlusion": ann.get("attributes", {}).get("occlusion", 0),
                    "area":      ann.get("area", w * h),
                    "ann_id":    ann["id"],
                }
            )
        return bboxes, labels, ann_ids, attributes

    def _build_target(
        self,
        img_id: int,
        bboxes: List,
        labels: List[int],
        attributes: List[Dict],
        img_meta: Dict,
    ) -> Dict[str, torch.Tensor]:
        H = img_meta["height"]
        W = img_meta["width"]

        if len(bboxes) == 0:
            return {
                "boxes":     torch.zeros((0, 4), dtype=torch.float32),
                "labels":    torch.zeros(0, dtype=torch.long),
                "blur":      torch.zeros(0, dtype=torch.float32),
                "occlusion": torch.zeros(0, dtype=torch.float32),
                "area":      torch.zeros(0, dtype=torch.float32),
                "image_id":  torch.tensor(img_id),
                "orig_size": torch.tensor([H, W]),
            }

        boxes_np = np.array(bboxes, dtype=np.float32)
        boxes_xyxy = np.stack(
            [
                boxes_np[:, 0],
                boxes_np[:, 1],
                boxes_np[:, 0] + boxes_np[:, 2],
                boxes_np[:, 1] + boxes_np[:, 3],
            ],
            axis=1,
        )
        boxes_xyxy[:, [0, 2]] /= W
        boxes_xyxy[:, [1, 3]] /= H
        boxes_xyxy = np.clip(boxes_xyxy, 0.0, 1.0)

        return {
            "boxes":     torch.from_numpy(boxes_xyxy),
            "labels":    torch.tensor(labels, dtype=torch.long),
            "blur":      torch.tensor([a["blur"]      for a in attributes], dtype=torch.float32),
            "occlusion": torch.tensor([a["occlusion"] for a in attributes], dtype=torch.float32),
            "area":      torch.tensor([a["area"]      for a in attributes], dtype=torch.float32),
            "image_id":  torch.tensor(img_id),
            "orig_size": torch.tensor([H, W]),
        }

    def get_category_name(self, cat_id: int) -> str:
        return self.categories.get(cat_id, {}).get("name", str(cat_id))

    @property
    def class_counts(self) -> Dict[str, int]:
        """Count instances per class for imbalance analysis."""
        counts: Dict[str, int] = {c: 0 for c in self.CLASSES}
        for anns in self.annotations.values():
            for ann in anns:
                idx = self.CAT_ID_TO_IDX.get(ann["category_id"])
                if idx is not None:
                    counts[self.CLASSES[idx]] += 1
        return counts

    def compute_sample_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler (inverse class freq)."""
        counts = self.class_counts
        inv_freq = {cls: 1.0 / max(cnt, 1) for cls, cnt in counts.items()}
        weights = []
        for img_id in self.img_ids:
            anns = self.annotations[img_id]
            if not anns:
                weights.append(min(inv_freq.values()))
                continue
            # Use max inverse frequency of any class present in this image
            w = max(
                inv_freq[self.CLASSES[self.CAT_ID_TO_IDX[a["category_id"]]]]
                for a in anns
                if a["category_id"] in self.CAT_ID_TO_IDX
            )
            weights.append(w)
        return weights
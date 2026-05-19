"""
Augmentation pipelines for BuzzSpot.

Design: spatial transforms (geometric + photometric) produce numpy uint8 output.
Normalization and tensor conversion happen downstream in TemporalFrameLoader so
that ALL frames in a temporal window are normalized consistently.

Three pipeline levels:
  - get_train_transforms : full spatial pipeline (no Normalize/ToTensor)
  - get_val_transforms   : resize + pad only (no augmentation)
  - get_tta_transforms   : list of TTATransform (spatial flips, tagged for inversion)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import albumentations as A
import cv2
import numpy as np
import torch


# ImageNet mean/std used by DINOv2 / RF-DETR
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Pre-built tensors for fast normalization (no allocation per call)
_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD_T  = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a CHW float32 tensor in [0, 1]."""
    mean = _MEAN_T.to(tensor.device)
    std  = _STD_T.to(tensor.device)
    return (tensor - mean) / std


def to_normalized_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 numpy → CHW float32, ImageNet-normalized."""
    t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    return normalize_tensor(t)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalised CHW tensor back to HWC uint8 for visualisation."""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255
    return img.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Training transforms  (output: numpy HWC uint8)
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(tile_size: int = 256) -> A.Compose:
    """Spatial augmentation pipeline for training — returns numpy uint8.

    Normalize + ToTensor are intentionally absent: they are applied by
    TemporalFrameLoader so that all temporal frames share the same normalization.
    """
    bbox_params = A.BboxParams(
        format="coco",
        label_fields=["class_labels", "ann_ids"],
        min_visibility=0.1,
        filter_lost_elements=True,
    )

    return A.Compose(
        [
            # ── Geometric ──────────────────────────────────────
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),

            # ── Photometric ────────────────────────────────────
            A.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5
            ),
            A.RandomShadow(p=0.3),
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, alpha_coef=0.1, p=0.1),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5), num_flare_circles_lower=3,
                num_flare_circles_upper=6, src_radius=100, p=0.05,
            ),

            # ── Motion blur — simulates fast insect flight ─────
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 15), p=1.0),
                    A.Defocus(radius=(1, 5), alias_blur=0.1, p=1.0),
                ],
                p=0.4,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),

            # ── Occlusion simulation ──────────────────────────
            A.CoarseDropout(
                max_holes=4, max_height=32, max_width=32,
                min_holes=1, min_height=8, min_width=8,
                fill_value=0, p=0.3,
            ),

            # ── Resize to tile ────────────────────────────────
            A.RandomSizedBBoxSafeCrop(
                height=tile_size, width=tile_size,
                erosion_rate=0.0,
                interpolation=cv2.INTER_LINEAR,
                p=0.3,
            ),
            A.LongestMaxSize(max_size=tile_size),
            A.PadIfNeeded(
                min_height=tile_size, min_width=tile_size,
                border_mode=cv2.BORDER_CONSTANT, value=0,
            ),
            # NOTE: no Normalize / ToTensorV2 — handled by TemporalFrameLoader
        ],
        bbox_params=bbox_params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation transforms  (output: numpy HWC uint8)
# ─────────────────────────────────────────────────────────────────────────────

def get_val_transforms(tile_size: int = 256) -> A.Compose:
    """Resize + pad only — no augmentation, no normalization."""
    bbox_params = A.BboxParams(
        format="coco",
        label_fields=["class_labels", "ann_ids"],
        min_visibility=0.0,
        filter_lost_elements=True,
    )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=tile_size),
            A.PadIfNeeded(
                min_height=tile_size, min_width=tile_size,
                border_mode=cv2.BORDER_CONSTANT, value=0,
            ),
        ],
        bbox_params=bbox_params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TTA transforms  (output: numpy HWC uint8 + inversion tag)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TTATransform:
    """Pairs a spatial transform with a tag used to invert predictions."""
    name: str          # "original" | "hflip" | "vflip" | "hflip_vflip"
    transform: A.Compose


def get_tta_transforms(tile_size: int = 256) -> List[TTATransform]:
    """Return TTA pipelines.  Normalize + ToTensor absent (done in predictor)."""
    base = [
        A.LongestMaxSize(max_size=tile_size),
        A.PadIfNeeded(
            min_height=tile_size, min_width=tile_size,
            border_mode=cv2.BORDER_CONSTANT, value=0,
        ),
    ]
    return [
        TTATransform("original",   A.Compose(base)),
        TTATransform("hflip",      A.Compose([A.HorizontalFlip(p=1.0)] + base)),
        TTATransform("vflip",      A.Compose([A.VerticalFlip(p=1.0)]   + base)),
        TTATransform("hflip_vflip",A.Compose([A.HorizontalFlip(p=1.0),
                                              A.VerticalFlip(p=1.0)]   + base)),
    ]

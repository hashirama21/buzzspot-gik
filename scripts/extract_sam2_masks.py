#!/usr/bin/env python3
"""
Extract per-instance SAM 2 RGBA crops from annotated training frames.

These crops are used by CopyPasteAugmentor during training to boost
rare classes (moths, hoverflies) via copy-paste augmentation.

Usage:
    python scripts/extract_sam2_masks.py \
        --ann      data/buzzset/annotations/train.json \
        --img-dir  data/buzzset/train/ \
        --out-dir  data/sam2_masks/ \
        --sam2-ckpt weights/sam2_hiera_large.pt \
        [--classes moth hoverfly]   # limit to specific classes (optional)
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {1: "bee", 2: "bumblebee", 3: "hoverfly", 4: "moth"}


def extract_with_sam2(img_rgb, box_xyxy, sam2_predictor):
    """Return HWC RGBA uint8 crop, or None if SAM2 unavailable."""
    try:
        sam2_predictor.set_image(img_rgb)
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_xyxy[None],   # (1, 4)
            multimask_output=False,
        )
        mask = masks[0, 0] > 0.5  # H×W bool

        x1, y1, x2, y2 = box_xyxy.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(img_rgb.shape[1], x2)
        y2 = min(img_rgb.shape[0], y2)

        crop_rgb  = img_rgb[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2].astype(np.uint8) * 255

        rgba = np.dstack([crop_rgb, crop_mask])  # RGBA
        return rgba
    except Exception as e:
        print(f"  SAM2 error: {e}")
        return None


def extract_bbox_crop(img_rgb, box_xyxy):
    """Fallback: simple crop with rectangular alpha mask."""
    x1, y1, x2, y2 = box_xyxy.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(img_rgb.shape[1], x2)
    y2 = min(img_rgb.shape[0], y2)
    crop = img_rgb[y1:y2, x1:x2]
    alpha = np.ones((*crop.shape[:2], 1), dtype=np.uint8) * 255
    return np.dstack([crop, alpha])


def load_sam2(ckpt_path: str, device: str = "cuda"):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2("sam2_hiera_l.yaml", ckpt_path, device=device)
        return SAM2ImagePredictor(model)
    except ImportError:
        print("[SAM2] Not installed — using bbox crop fallback.")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann",       required=True)
    parser.add_argument("--img-dir",   required=True)
    parser.add_argument("--out-dir",   required=True)
    parser.add_argument("--sam2-ckpt", default=None)
    parser.add_argument("--classes",   nargs="*", default=None,
                        help="Filter to specific class names")
    args = parser.parse_args()

    with open(args.ann) as f:
        coco = json.load(f)

    img_lookup = {img["id"]: img for img in coco["images"]}
    out_dir    = Path(args.out_dir)

    # Create per-class output dirs
    for cls_name in CLASS_NAMES.values():
        (out_dir / cls_name).mkdir(parents=True, exist_ok=True)

    # Load SAM2
    sam2 = load_sam2(args.sam2_ckpt) if args.sam2_ckpt else None

    target_classes = set(args.classes) if args.classes else set(CLASS_NAMES.values())

    saved = 0
    for ann in coco["annotations"]:
        cat_name = CLASS_NAMES.get(ann["category_id"])
        if cat_name not in target_classes:
            continue

        img_meta = img_lookup[ann["image_id"]]
        img_path = Path(args.img_dir) / img_meta["file_name"]
        if not img_path.exists():
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        x, y, w, h = ann["bbox"]
        box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

        if w < 8 or h < 8:
            continue

        if sam2 is not None:
            rgba = extract_with_sam2(rgb, box_xyxy, sam2)
        else:
            rgba = extract_bbox_crop(rgb, box_xyxy)

        if rgba is None or rgba.shape[0] < 4 or rgba.shape[1] < 4:
            continue

        # Save as PNG (supports alpha)
        out_path = out_dir / cat_name / f"{ann['id']:08d}.png"
        bgra = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        bgra = np.dstack([bgra, rgba[:, :, 3]])
        cv2.imwrite(str(out_path), bgra)
        saved += 1

        if saved % 200 == 0:
            print(f"  Saved {saved} masks…")

    print(f"\nDone — {saved} instance masks saved to {out_dir}")


if __name__ == "__main__":
    main()

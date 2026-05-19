#!/usr/bin/env python3
"""
Run inference on the BuzzSpot test set and generate predictions.json.

Usage:
    python scripts/infer.py \
        --weights outputs/rfdetr_temporal_v1/best.pth \
        --test-ann data/buzzset/annotations/test.json \
        --test-dir data/buzzset/test/ \
        --out      submissions/predictions.json

Then zip:
    cd submissions && zip my_submission.zip predictions.json
"""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictors.predictor import BuzzSpotPredictor
from src.models.rfdetr_temporal import RFDETRTemporal


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = OmegaConf.create(ckpt["config"])

    model = RFDETRTemporal(
        num_classes=cfg.data.num_classes,
        d_model=cfg.model.hidden_dim,
        num_queries=cfg.model.num_queries,
        num_frames=cfg.data.temporal.num_frames,
        temporal_type=cfg.model.temporal_head.type,
        use_attribute_heads=cfg.model.attribute_heads.enabled,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="BuzzSpot inference")
    parser.add_argument("--weights",   required=True, help="Checkpoint .pth file")
    parser.add_argument("--test-ann",  required=True, help="test.json (images only)")
    parser.add_argument("--test-dir",  required=True, help="Test image directory")
    parser.add_argument("--out",       default="submissions/predictions.json")
    parser.add_argument("--zip",       action="store_true", help="Auto-zip output")
    parser.add_argument("--score-thr", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, cfg = load_model_from_checkpoint(args.weights, device)

    # Override inference threshold
    cfg.inference.confidence_threshold = args.score_thr

    predictor = BuzzSpotPredictor(model, cfg, device=device)

    # Load test image list
    with open(args.test_ann) as f:
        test_coco = json.load(f)

    # Run inference only on keyframes (context frames have no annotations and
    # must not be submitted — the validator will reject them).
    all_images = test_coco["images"]
    has_flag = any("is_keyframe" in img for img in all_images)
    keyframes = (
        [img for img in all_images if img.get("is_keyframe", False)]
        if has_flag else all_images
    )

    all_predictions = []
    n = len(keyframes)

    for i, img_meta in enumerate(keyframes):
        img_path = Path(args.test_dir) / img_meta["file_name"]
        preds = predictor.predict(str(img_path), img_meta["id"])
        all_predictions.extend(preds)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] {img_meta['file_name']} — {len(preds)} dets")

    # Save predictions.json
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_predictions, f)

    print(f"\nSaved {len(all_predictions)} predictions → {out_path}")

    # Optional: auto-zip for direct Codabench upload
    if args.zip:
        zip_path = out_path.parent / (out_path.stem + ".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, "predictions.json")
        print(f"Zipped → {zip_path}")
        print("Ready to upload to Codabench.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Codabench inference entrypoint.

Expected environment variables (set by Codabench):
    INPUT_DIR   = directory containing test images and test.json
    OUTPUT_DIR  = where to write predictions.json

The script:
1. Loads the model from /app/buzzspot/weights/best.pth
2. Runs inference on all test images
3. Writes OUTPUT_DIR/predictions.json
"""
import json
import os
import sys
import zipfile
from pathlib import Path

import torch

sys.path.insert(0, "/app/buzzspot")

from src.inference.predictors.predictor import BuzzSpotPredictor
from src.models.rfdetr_temporal import RFDETRTemporal
from omegaconf import OmegaConf


INPUT_DIR  = Path(os.environ.get("INPUT_DIR",  "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
WEIGHTS    = Path(os.environ.get("WEIGHTS",    "/app/buzzspot/weights/best.pth"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"[Entrypoint] Input dir : {INPUT_DIR}")
    print(f"[Entrypoint] Output dir: {OUTPUT_DIR}")
    print(f"[Entrypoint] Weights   : {WEIGHTS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Entrypoint] Device    : {device}")

    # ── Load checkpoint ───────────────────────────────────────
    if not WEIGHTS.exists():
        print(f"ERROR: Weights not found at {WEIGHTS}")
        sys.exit(1)

    ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
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
    print(f"[Entrypoint] Model loaded — epoch {ckpt.get('epoch', '?')}")

    predictor = BuzzSpotPredictor(model, cfg, device=device)

    # ── Load test annotations ─────────────────────────────────
    test_ann = INPUT_DIR / "test.json"
    if not test_ann.exists():
        # Fallback: look for any JSON in input
        candidates = list(INPUT_DIR.glob("*.json"))
        if not candidates:
            print("ERROR: No test.json found in INPUT_DIR")
            sys.exit(1)
        test_ann = candidates[0]

    with open(test_ann) as f:
        test_coco = json.load(f)

    images = test_coco["images"]
    print(f"[Entrypoint] Running inference on {len(images)} images…")

    all_predictions = []
    for i, img_meta in enumerate(images):
        img_path = INPUT_DIR / img_meta["file_name"]
        if not img_path.exists():
            # Try flat directory
            img_path = INPUT_DIR / Path(img_meta["file_name"]).name
        if not img_path.exists():
            print(f"  WARNING: image not found — {img_meta['file_name']}")
            continue

        preds = predictor.predict(str(img_path), img_meta["id"])
        all_predictions.extend(preds)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(images)} — cumulative dets: {len(all_predictions)}")

    # ── Write output ──────────────────────────────────────────
    out_json = OUTPUT_DIR / "predictions.json"
    with open(out_json, "w") as f:
        json.dump(all_predictions, f)

    print(f"\n[Entrypoint] Done — {len(all_predictions)} predictions written to {out_json}")


if __name__ == "__main__":
    main()

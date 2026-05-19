#!/usr/bin/env python3
"""
Generate pseudo-labels for the test set with Grounding DINO + SAM 2,
then self-train the detector for N epochs.

Usage:
    python scripts/pseudo_label.py \
        --weights     outputs/rfdetr_temporal_v1/best.pth \
        --test-ann    data/buzzset/annotations/test.json \
        --test-dir    data/buzzset/test/ \
        --gdino-cfg   weights/GroundingDINO_SwinT_OGC.py \
        --gdino-ckpt  weights/groundingdino_swint_ogc.pth \
        --sam2-ckpt   weights/sam2_hiera_large.pt \
        --out-ann     data/buzzset/annotations/pseudo_train.json \
        --epochs      2
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictors.pseudo_labeler import (
    GroundingDINOWrapper,
    SAM2MaskRefiner,
    PseudoLabelGenerator,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--test-ann",   required=True)
    parser.add_argument("--test-dir",   required=True)
    parser.add_argument("--gdino-cfg",  default=None)
    parser.add_argument("--gdino-ckpt", default=None)
    parser.add_argument("--sam2-ckpt",  default=None)
    parser.add_argument("--out-ann",    default="data/buzzset/annotations/pseudo_train.json")
    parser.add_argument("--conf",       type=float, default=0.5)
    parser.add_argument("--epochs",     type=int,   default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Step 1: Generate pseudo-labels ────────────────────────
    print("=== Step 1 — Generating pseudo-labels ===")
    gdino = GroundingDINOWrapper(
        config_path=args.gdino_cfg or "",
        checkpoint_path=args.gdino_ckpt or "",
        device=device,
    )
    sam2 = SAM2MaskRefiner(
        checkpoint_path=args.sam2_ckpt or "",
        device=device,
    )
    generator = PseudoLabelGenerator(gdino, sam2, confidence_threshold=args.conf)
    generator.run(args.test_ann, args.test_dir, args.out_ann)

    # ── Step 2: Self-train on pseudo-labels ───────────────────
    if args.epochs > 0:
        print(f"\n=== Step 2 — Self-training for {args.epochs} epochs ===")
        # Entry point is src/train.py (Hydra-based)
        subprocess.run(
            [
                sys.executable, "src/train.py",
                f"data.train_ann={args.out_ann}",
                f"training.epochs={args.epochs}",
                "training.experiment_name=pseudo_selftraining",
                "training.curriculum.enabled=false",
            ],
            check=True,
        )

    print("\nPseudo-labeling pipeline complete.")


if __name__ == "__main__":
    main()

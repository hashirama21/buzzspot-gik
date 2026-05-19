# BuzzSpot Challenge — CVPPA@ECCV 2026
# Pollinator Detection: Bees · Bumblebees · Hoverflies · Moths

## Project structure

```
buzzspot/
├── configs/
│   ├── default.yaml           ← master config (Hydra)
│   ├── ablation_baseline.yaml ← RF-DETR without temporal
│   └── rfdetr_mega.yaml       ← RF-DETR + MEGA temporal head
│
├── data/
│   ├── datasets/buzzset.py            ← BuzzSet COCO dataset class
│   ├── preprocessing/
│   │   ├── temporal.py                ← frame loading: stack / diff / flow
│   │   └── tiling.py                  ← SAHI-style tile extractor
│   └── augmentations/
│       ├── transforms.py              ← Albumentations pipelines
│       └── copy_paste.py              ← SAM 2 copy-paste augmentation
│
├── models/
│   └── rfdetr_temporal.py             ← RF-DETR + TemporalQueryFusion + AttributeHead
│
├── training/
│   ├── trainer.py                     ← full train loop (AMP, curriculum, checkpointing)
│   └── losses/criterion.py            ← FocalLoss + CIoU + AttributeLoss + HungarianMatcher
│
├── inference/
│   └── predictors/
│       ├── predictor.py               ← tiling + TTA + WBF inference
│       └── pseudo_labeler.py          ← GroundingDINO + SAM2 pseudo-labeling
│
├── evaluation/
│   └── metrics/coco_eval.py           ← mAP@0.5:0.95 + per-class/size/attribute
│
├── tools/                             ← devkit utilities (ported from official kit)
│   ├── buzzset_loader.py              ← BuzzSetSF / BuzzSetMF dataset classes
│   ├── evaluate.py                    ← official Codabench scoring
│   ├── evaluate_detailed.py           ← PR curves + attribute-stratified metrics
│   ├── convert.py                     ← COCO/YOLO single-frame conversion
│   ├── validate_submission.py         ← Codabench submission validator
│   ├── sahi_inferencer.py             ← SAHI-sliced RF-DETR inference
│   └── visualize.py                   ← keyframe + context GIF visualization
│
├── scripts/
│   ├── train.py                       ← training entry point (Hydra)
│   ├── infer.py                       ← inference + predictions.json generation
│   ├── pseudo_label.py                ← pseudo-labeling + self-training
│   └── extract_sam2_masks.py          ← offline SAM2 mask extraction for copy-paste
│
├── docker/
│   ├── Dockerfile                     ← production image for Codabench
│   └── entrypoint.py                  ← Codabench inference entrypoint
│
├── tests/
│   └── test_pipeline.py               ← unit tests (pytest)
│
└── requirements.txt
```

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
# Then install RF-DETR:
pip install rfdetr
```

### 2. Download BuzzSet
```bash
wget https://phenoroam.phenorob.de/file-uploader/download/public/35261367058-BuzzSet_challenge.zip
unzip 35261367058-BuzzSet_challenge.zip -d data/buzzset/
```

### 3. (Optional) Extract SAM 2 masks for copy-paste augmentation
```bash
python scripts/extract_sam2_masks.py \
    --ann     data/buzzset/annotations/train.json \
    --img-dir data/buzzset/train/ \
    --out-dir data/sam2_masks/ \
    --sam2-ckpt weights/sam2_hiera_large.pt \
    --classes moth hoverfly
```

### 4. Train
```bash
# Baseline (week 1)
python scripts/train.py --config-name ablation_baseline

# RF-DETR + frame stacking (week 1-2)
python scripts/train.py data.temporal.strategy=stack

# RF-DETR + MEGA temporal head (week 3)
python scripts/train.py --config-name rfdetr_mega
```

### 5. Evaluate + generate submission
```bash
python scripts/infer.py \
    --weights outputs/rfdetr_temporal_v1/best.pth \
    --test-ann data/buzzset/annotations/test.json \
    --test-dir data/buzzset/test/ \
    --out      submissions/predictions.json \
    --zip
```

### 6. (Week 4) Pseudo-labeling self-training
```bash
python scripts/pseudo_label.py \
    --weights    outputs/rfdetr_temporal_v1/best.pth \
    --test-ann   data/buzzset/annotations/test.json \
    --test-dir   data/buzzset/test/ \
    --gdino-ckpt weights/groundingdino_swint_ogc.pth \
    --sam2-ckpt  weights/sam2_hiera_large.pt \
    --epochs     2
```

### 7. Docker (Codabench)
```bash
docker build -t buzzspot -f docker/Dockerfile .

# Local test
docker run \
    -e INPUT_DIR=/data/test \
    -e OUTPUT_DIR=/data/output \
    -v $(pwd)/data/buzzset:/data \
    buzzspot

# Push
docker tag buzzspot registry.gitlab.uni-bonn.de:5050/lreissn1/buzzset:latest
docker push registry.gitlab.uni-bonn.de:5050/lreissn1/buzzset:latest
```

### 8. Run tests
```bash
pytest tests/ -v
```

## Key design decisions

| Decision | Rationale |
|---|---|
| RF-DETR as backbone | Official BuzzSet baseline, DINOv2, best small-object AP |
| TemporalQueryFusion | Lightweight cross-attention on 5-frame window, no TransVOD overhead |
| CIoU loss | Precise bbox regression for tiny insects — critical for mAP@0.5:0.95 |
| Per-class focal γ | Moths/hoverflies are rare — higher γ prevents class domination |
| Curriculum learning | Start on clean samples, introduce blur/occlusion progressively |
| WBF ensemble | Transformer + CNN ensemble, better than NMS for overlapping tiny boxes |
| Pseudo-labeling | GroundingDINO + SAM2 on test set — semi-supervised gain without annotation |

## Devkit tools (`src/tools/`)

Ported from the [official BuzzSpot dev-kit](https://github.com/lereiss/buzzspot-devkit) (MIT) into `src/tools/` for direct use in our pipeline:

| Module | Contents |
|---|---|
| `buzzset_loader.py` | `BuzzSetSF` / `BuzzSetMF` — authoritative dataset loaders |
| `evaluate.py` | `evaluate()` — official Codabench scoring (mAP@0.5:0.95 + per-class) |
| `evaluate_detailed.py` | `evaluate_detailed()` — PR curves, confusion matrix, attribute-stratified recall |
| `_eval_helper.py` | Greedy matching, AP, plotting helpers |
| `_generate_report.py` | HTML eval report generator |
| `convert.py` | `to_coco()` / `to_yolo()` — convert multi-frame BuzzSet to single-frame |
| `validate_submission.py` | `run()` — validate submission ZIP before Codabench upload |
| `sahi_inferencer.py` | `RFDETR_SAHI_Inferencer` — SAHI-sliced RF-DETR inference |
| `visualize.py` | `show_keyframe()`, `make_context_gif()`, `display_context_gif()` |

```bash
# Convert to single-frame COCO (for YOLOv11 baseline)
python -m src.tools.convert data/buzzset/ data/buzzset_sf/ --format coco

# Validate submission before upload
python -m src.tools.validate_submission --zipfile submissions/predictions.zip \
    --test_file data/buzzset/annotations/test.json

# Run official evaluation locally
python -m src.tools.evaluate \
    --res submissions/predictions.json \
    --ref data/buzzset/annotations/valid.json \
    -o outputs/eval/
```

## Competition details

- **Challenge**: CVPPA@ECCV 2026 — BuzzSpot
- **Metric**: mAP@0.5:0.95 (primary) + per-class, per-size, per-attribute
- **Submission**: `predictions.json` zipped, uploaded to Codabench
- **Deadline**: 26 June 2026 (dev phase)
- **Contact**: buzzspot.challenge.cvppa@uni-bonn.de
- **License**: MIT (devkit) — see `LICENSE`

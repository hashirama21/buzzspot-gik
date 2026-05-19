import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASS_NAMES = {1: "bee", 2: "bumblebee", 3: "hoverfly", 4: "moth"}


def scoring_failed(error_msg):
    """Emit a zeroed-out scores file and exit with non-zero status."""
    print(f"SCORING FAILED: {error_msg}", file=sys.stderr)
    sys.exit(1)


def load_ground_truth(ref_file=None):
    if not ref_file.exists():
        scoring_failed(f"Missing reference file at {ref_file}")
    try:
        return COCO(str(ref_file))
    except Exception as e:
        scoring_failed(f"Could not load ground_truth.json: {e}")


def load_predictions(res_file,coco_gt):
    """Load predictions, with a fallback that maps file_name -> image_id."""
    if not res_file.exists():
        scoring_failed(f"Missing predictions file at {res_file}")

    try:
        with open(res_file, encoding="utf-8") as f:
            preds = json.load(f)
    except Exception as e:
        scoring_failed(f"Could not parse predictions.json: {e}")

    if not isinstance(preds, list):
        scoring_failed("predictions.json must be a JSON list of detection dicts")
    if len(preds) == 0:
        # Empty list is valid: load it and let mAP come out as 0.
        try:
            return coco_gt.loadRes([])
        except Exception as e:
            scoring_failed(f"loadRes failed on empty list: {e}")

    # First attempt: assume predictions already use image_id.
    try:
        return coco_gt.loadRes(preds)
    except Exception as e:
        scoring_failed(f"COCO loadRes failed: {e}")


def print_coco_style_scores(scores):
    """
    Prints a COCO scores dictionary with dynamic per-class support.
    """
    # Define primary summary metrics for special formatting
    summary_map = {
        "mAP_05095": "mAP @ [ IoU=0.50:0.95 ]",
        "mAP_05":    "mAP @ [ IoU=0.50      ]",
        "mAP_Small": "mAP @ [ Small Objects ]",
        "mAP_Medium":"mAP @ [ Medium Objects]",
        "mAP_Large": "mAP @ [ Large Objects ]"
    }

    print(f"\n{'='*55}")
    print(f"{'METRIC / CLASS':<35} | {'SCORE':>15}")
    print(f"{'='*55}")

    # 1. Print Summary Metrics first
    for key, label in summary_map.items():
        if key in scores:
            val = scores[key]
            val_str = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
            print(f"{label:<35} | {val_str:>15}")

    # 2. Print Per-Class Metrics
    print(f"{'-'*35}-|-{'-'*15}")
    print(f"{'PER-CLASS PRECISION (AP)':<35} |")
    print(f"{'-'*35}-|-{'-'*15}")

    for key, val in scores.items():
        # Only print keys that start with AP_ and aren't in our summary map
        if key.startswith("AP_") and key not in summary_map:
            class_name = key.replace("AP_", "").capitalize()
            val_str = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
            print(f"  > {class_name:<31} | {val_str:>15}")

    print(f"{'='*55}\n")


def evaluate(output_dir: str = None, res_file: str = None, ref_file: str = None)-> dict | None:
    """ Evaluate predictions against reference annotations, writing scores to output_dir/scores.json.
    Args:
        output_dir:(str) Directory where scores.json will be written.
        res_file:(str) Path to predictions.json (participant's output).
        ref_file:(str) Path to ground_truth.json (reference annotations).    
    Returns:
        dict: COCO evaluation results.
    """
    # convert every input str to Path and check existence
    if output_dir is not None:
        output_dir = Path(output_dir)
    if res_file is not None:
        res_file = Path(res_file)
    if ref_file is not None:
        ref_file = Path(ref_file)
    

    scores_file = output_dir / "scores.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    coco_gt = load_ground_truth(ref_file)
    coco_dt = load_predictions(res_file, coco_gt)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    #coco_eval.params.iouThrs = IOU_THRS

    coco_eval.evaluate()
    coco_eval.accumulate()
    print("\n\n")
    coco_eval.summarize()
    print("\n\n")

  
    map_05095 = float(coco_eval.stats[0])  # AP @ IoU 0.50:0.95, all areas


    # precision shape: [T, R, K, A, M]
    #   T: IoU thresholds (index 0 = 0.50)
    #   R: recall thresholds (101 points)
    #   K: categories
    #   A: area ranges  (0=all, 1=small, 2=medium, 3=large)
    #   M: max detections (index -1 = 100)
    precision = coco_eval.eval["precision"]

    def _ap_at(iou_idx,area_idx):
        p = precision[iou_idx, :, :, area_idx, -1]
        valid = p[p > -1]
        return float(valid.mean()) if valid.size else 0.0


    iou_50_idx = 0  # index for IoU=0.50 in coco_eval.params.iouThrs
    map_05 = _ap_at(iou_50_idx, 0)
    s_map = _ap_at(iou_50_idx, 1)
    m_map = _ap_at(iou_50_idx, 2)
    l_map = _ap_at(iou_50_idx, 3)


    scores ={
        "mAP_05095": map_05095,
        "mAP_05": map_05,
        "mAP_Small": s_map,
        "mAP_Medium": m_map,
        "mAP_Large": l_map,
    }

    cat_ids = coco_gt.getCatIds()
    per_class_ap = {}
    for k, cat_id in enumerate(cat_ids):
        p = precision[:, :, k, 0, -1]  # all IoU thresholds, all areas
        valid = p[p > -1]
        ap_k = float(valid.mean()) if valid.size else 0.0
        name = CLASS_NAMES.get(cat_id, f"class_{cat_id}")
        per_class_ap[f"AP_{name}"] = ap_k


    scores.update(per_class_ap)

    print_coco_style_scores(scores)
    scores_file.write_text(json.dumps(scores))
    return coco_eval.stats


def run():
    parser = argparse.ArgumentParser(description="Evaluate object detection predictions given COCO-style jsons.")
    

    parser.add_argument("--res", required=True, help="Input Prediction json file path")
    parser.add_argument("--ref", required=True, help="Input Reference json file path")
    parser.add_argument("-o", "--output", required=True, help="Output folder path")

    args = parser.parse_args()

    evaluate(output_dir=Path(args.output),
             res_file=Path(args.res),
             ref_file=Path(args.ref))    


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""
BuzzSet — Detailed COCO Evaluation

Takes a COCO ground-truth JSON (with optional per-annotation `attributes` dict
containing `blur` and `occluded`) and a prediction JSON in COCO-results
format, and produces:

  * Official COCO metrics (mAP@.50:.95, mAP@.50, mAP@.75, mAP_s/m/l, AR)
  * Per-class AP@.50
  * Overall TP/FP/FN, precision, recall, F1 at a fixed confidence threshold
  * Attribute-stratified recall and TP-confidence (auto-detected attributes)
  * Confusion matrix (incl. background row/col)
  * Size distribution (GT vs prediction)
  * A `summary.json` containing all numbers, plus PNG plots in `output_dir/`

No inference is performed — predictions are read from disk.

Usage
-----
    Or import and call directly:
        from eval_detailed import eval_detailed
        eval_detailed("gt.json", "predictions.json", "results/")
"""

from __future__ import annotations

import json

from collections import defaultdict
from pathlib import Path
import argparse

import numpy as np


from ._eval_helper import *
from ._generate_report import generate_report

# CONFIG  (edit and run, or call eval_detailed() directly from elsewhere)

ATTR_LABEL_MAP = {
    "blur":      {0: "Sharp", 1: "Blurry"},
    "occluded":  {1: "full vis", 2: "self occ.", 3: "obj. occ."},
}

# LOADING

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_gt(gt_path: str) -> dict:
    """Load COCO GT and build lookup structures.

    Returns a dict with:
      categories     : {cat_id: name}
      cat_id_to_idx  : {cat_id: dense_idx}
      class_names    : list of names ordered by dense_idx
      images         : {image_id: image_info}
      anns_by_image  : {image_id: [ann, ...]}
      has_attributes : bool
      attr_keys      : set of attribute keys observed in the data
      raw            : the raw COCO dict
    """
    coco = _load_json(gt_path)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    cat_id_to_idx = {cid: i for i, cid in enumerate(sorted(categories.keys()))}
    class_names = [categories[cid] for cid in sorted(categories.keys())]

    images = {}
    for img in coco["images"]:
        if img.get("is_keyframe", False):
            images[img["id"]] = img

    anns_by_image: dict = defaultdict(list)
    attr_keys: set = set()
    for ann in coco["annotations"]:
        if ann["image_id"] not in images:
            continue
        anns_by_image[ann["image_id"]].append(ann)
        if isinstance(ann.get("attributes"), dict):
            attr_keys.update(ann["attributes"].keys())

    return {
        "categories":     categories,
        "cat_id_to_idx":  cat_id_to_idx,
        "class_names":    class_names,
        "images":         images,
        "anns_by_image":  dict(anns_by_image),
        "has_attributes": len(attr_keys) > 0,
        "attr_keys":      attr_keys,
        "raw":            coco,
    }


def load_predictions(pred_path: str) -> list:
    """Load predictions in COCO-results format.

    Accepts either a JSON list of `{image_id, category_id, bbox, score}`
    or a dict with an `annotations` key (some tools wrap the list).
    """
    data = _load_json(pred_path)
    if isinstance(data, dict) and "annotations" in data:
        data = data["annotations"]
    if not isinstance(data, list):
        raise ValueError(f"Prediction JSON must be a list, got {type(data)}")
    return data


# MAIN EVALUATION

def evaluate_detailed(ref_file: str, res_file: str, output_dir: str,
                  conf_threshold: float = 0.5,
                  iou_threshold:  float = 0.5,
                  generate_report_html: bool = False) -> dict:
    """Run full evaluation of a COCO predictions file against a GT file.

    Parameters
    ----------
    ref_file : str
        Path to COCO ground-truth JSON. Annotations may carry an
        `attributes` dict; stratified metrics are computed automatically
        for whichever attribute keys are present.
    res_file : str
        Path to predictions in COCO-results list format
        (`[{image_id, category_id, bbox, score}, ...]`).
    output_dir : str
        Directory to write plots + `summary.json`. Created if missing.
    conf_threshold : float
        Score threshold.Does NOT affect pycocotools mAP.
    iou_threshold : float
        IoU threshold for greedy matching.
    generate_report_html : bool
        Whether to generate a detailed evaluation report. If False, only the summary JSON is written.

    Returns
    -------
    dict
        Full summary (also written to `output_dir/summary.json`).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading GT      : {ref_file}")
    gt = load_gt(ref_file)
    n_gt_total_load = sum(len(v) for v in gt["anns_by_image"].values())
    print(f"  {len(gt['images'])} images, {n_gt_total_load} annotations")
    print(f"  Classes ({len(gt['class_names'])}): {gt['class_names']}")
    print(f"  Attribute keys present: "
          f"{sorted(gt['attr_keys']) if gt['has_attributes'] else 'none'}")

    print(f"Loading preds   : {res_file}")
    preds_all = load_predictions(res_file)
    print(f"  {len(preds_all)} predictions total")

    valid_image_ids = set(gt["images"].keys())
    valid_cat_ids   = set(gt["cat_id_to_idx"].keys())
    preds_filt = [p for p in preds_all
                  if p["image_id"]    in valid_image_ids
                  and p["category_id"] in valid_cat_ids
                  and p["score"]      >= conf_threshold]
    print(f"  {len(preds_filt)} predictions @ score>={conf_threshold} "
          f"on kept images / known classes")

    preds_by_img: dict = defaultdict(list)
    for p in preds_filt:
        preds_by_img[p["image_id"]].append(p)

    n_classes = len(gt["class_names"])

    all_records   = []
    n_gt_per_cls  = {i: 0 for i in range(n_classes)}
    gt_areas, pred_areas = [], []
    confusion = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

    attr_keys = sorted(gt["attr_keys"])
    tp_by_attr: dict = defaultdict(list)   # attr_name -> [(value, score), ...]
    fn_by_attr: dict = defaultdict(list)   # attr_name -> [value, ...]

    for img_id in gt["images"]:
        anns = gt["anns_by_image"].get(img_id, [])

        # GT
        gts = []
        for a in anns:
            cls_idx = gt["cat_id_to_idx"][a["category_id"]]
            gts.append({"xyxy": xywh_to_xyxy(a["bbox"]),
                        "cat_idx": cls_idx, "ann": a})
            n_gt_per_cls[cls_idx] += 1
            x, y, w, h = a["bbox"]
            gt_areas.append(np.sqrt(w * h))

        # Preds
        pds = []
        for p in preds_by_img.get(img_id, []):
            x, y, w, h = p["bbox"]
            pds.append({"xyxy": [x, y, x + w, y + h],
                        "score": float(p["score"]),
                        "cat_idx": gt["cat_id_to_idx"][p["category_id"]]})
            pred_areas.append(np.sqrt(w * h))

        # Class-aware matching → PR / AP / recall / attribute stratification
        records, fn_idxs = greedy_match(pds, gts, iou_threshold, class_aware=True)
        all_records.extend(records)

        for rec in records:
            if rec["is_tp"]:
                ann_attrs = gts[rec["gt_idx"]]["ann"].get("attributes") or {}
                for k in attr_keys:
                    if k in ann_attrs:
                        tp_by_attr[k].append((ann_attrs[k], rec["score"]))
        for j in fn_idxs:
            ann_attrs = gts[j]["ann"].get("attributes") or {}
            for k in attr_keys:
                if k in ann_attrs:
                    fn_by_attr[k].append(ann_attrs[k])

        # Class-agnostic matching → confusion matrix
        c_records, c_fn = greedy_match(pds, gts, iou_threshold, class_aware=False)
        for rec in c_records:
            if rec["is_tp"]:
                gt_cls = gts[rec["gt_idx"]]["cat_idx"]
                confusion[gt_cls, rec["cat_idx"]] += 1
            else:
                confusion[n_classes, rec["cat_idx"]] += 1
        for j in c_fn:
            confusion[gts[j]["cat_idx"], n_classes] += 1

    n_tp = sum(1 for r in all_records if r["is_tp"])
    n_fp = sum(1 for r in all_records if not r["is_tp"])
    total_gt = sum(n_gt_per_cls.values())
    n_fn = total_gt - n_tp
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall    = n_tp / total_gt      if total_gt        else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    ap50 = per_class_ap50(all_records, n_gt_per_cls, n_classes)

    # Attribute-stratified results
    attr_results: dict = {}
    attr_inputs:  dict = {}
    for attr_name in attr_keys:
        labels = ATTR_LABEL_MAP.get(attr_name)
        if labels is None:
            observed = sorted({v for v, _ in tp_by_attr[attr_name]}
                              | set(fn_by_attr[attr_name]))
            labels = {v: str(v) for v in observed}
        m = attribute_stratified(tp_by_attr[attr_name],
                                 fn_by_attr[attr_name], labels)
        attr_results[attr_name] = {
            str(k): {kk: vv for kk, vv in v.items() if kk != "tp_scores"}
            for k, v in m.items()
        }
        attr_inputs[attr_name] = (m, labels)

    # Official COCO metrics — run on the unfiltered prediction file
    print("\nOfficial COCO metrics:")
    coco_stats = run_pycocotools_eval(ref_file, res_file, out)

    print("\nGenerating plots ...")
    plot_pr_curves(all_records, n_gt_per_cls, gt["class_names"],
                   out / "pr_curves.png", iou_threshold)
    plot_confusion_matrix(confusion, gt["class_names"],
                          out / "confusion_matrix.png")
    plot_score_histogram(all_records, out / "score_histogram.png")
    plot_per_class_ap(gt["class_names"], ap50, out / "per_class_ap50.png")
    if gt_areas or pred_areas:
        plot_size_distribution(gt_areas, pred_areas,
                               out / "size_distribution.png")

    if attr_inputs:
        attr_dir = out / "attribute_analysis"
        attr_dir.mkdir(exist_ok=True)
        for attr_name, (m, labels) in attr_inputs.items():
            plot_attr_recall(m, attr_name, labels,
                             attr_dir / f"{attr_name}_recall.png")
            plot_attr_confidence(m, attr_name, labels,
                                 attr_dir / f"{attr_name}_confidence.png")
        plot_attr_overview(attr_inputs, attr_dir / "attribute_overview.png")

    summary = {
        "config": {
            "ref_file":             str(ref_file),
            "res_file":             str(res_file),
            "confidence_threshold": conf_threshold,
            "iou_threshold":        iou_threshold,
        },
        "n_images":            len(gt["images"]),
        "n_gt":                total_gt,
        "n_predictions_total": len(preds_all),
        "n_predictions_used":  len(preds_filt),
        f"TP@{conf_threshold}":         n_tp,
        f"FP@{conf_threshold}":         n_fp,
        f"FN@{conf_threshold}":         n_fn,
        f"precision@{conf_threshold}":  round(precision, 4),
        f"recall@{conf_threshold}":     round(recall, 4),
        f"f1@{conf_threshold}":         round(f1, 4),
        "per_class_AP50": {gt["class_names"][i]: (round(v, 4) if not np.isnan(v) else None)
                           for i, v in enumerate(ap50)},
        "per_class_n_gt": {gt["class_names"][i]: n_gt_per_cls[i]
                           for i in range(n_classes)},
        "attribute_metrics": attr_results,
        "coco": coco_stats,
    }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Images       : {summary['n_images']}")
    print(f"  GT objects   : {total_gt}")
    print(f"  Predictions  : {len(preds_all)}  "
          f"({len(preds_filt)} above conf={conf_threshold})")
    print(f"  TP / FP / FN : {n_tp} / {n_fp} / {n_fn}")
    print(f"  P / R / F1   : {precision:.4f} / {recall:.4f} / {f1:.4f}")
    print(f"  mAP@.50      : {coco_stats['mAP@50']:.4f}")
    print(f"  mAP@.50:.95  : {coco_stats['mAP@50:95']:.4f}")
    print(f"  mAP small/med/large: "
          f"{coco_stats['mAP_small']:.4f} / {coco_stats['mAP_medium']:.4f} / "
          f"{coco_stats['mAP_large']:.4f}")
    print()
    for i, (cls, ap) in enumerate(zip(gt["class_names"], ap50)):
        s = f"{ap:.4f}" if not np.isnan(ap) else "n/a"
        print(f"    {cls:15s} AP@50: {s}  (n_gt={n_gt_per_cls[i]})")

    for attr_name, (m, labels) in attr_inputs.items():
        print(f"\n  Recall by {attr_name}:")
        for v in sorted(labels.keys()):
            d = m.get(v, {})
            r = d.get("recall", float("nan"))
            r_s = f"{r:.4f}" if not np.isnan(r) else "n/a"
            mc = d.get("mean_conf", float("nan"))
            mc_s = f"{mc:.3f}" if not np.isnan(mc) else "n/a"
            print(f"    {labels[v]:15s}: recall={r_s}  mean_conf={mc_s}  "
                  f"TP={d.get('n_tp', 0)}  FN={d.get('n_fn', 0)}  GT={d.get('n_gt', 0)}")

    print(f"\nAll results written to: {out.resolve()}")

    ###############################################
    if generate_report_html:
        print("\nGenerating detailed report ...")

        generate_report(output_dir)

    return summary


def run():
    parser = argparse.ArgumentParser(description="Process data via folder or specific reference files.")
    

    parser.add_argument("--res", required=True, help="Input Prediction json file path")
    parser.add_argument("--ref", required=True, help="Input Reference json file path")
    parser.add_argument("-o", "--output", required=True, help="Output folder path")

    parser.add_argument("-conf", required=False, help="Confidence threshold", type=float, default=0.5)
    parser.add_argument("-iou", required=False, help="IoU threshold", type=float, default=0.5)
    parser.add_argument("-r","--report_html", action="store_true", help="Whether a html-overview should be generated")
    args = parser.parse_args()

    # Logic execution based on which command was used


    evaluate_detailed(
                        res_file=args.res,
                        ref_file=args.ref,
                        output_dir=args.output,
                        conf_threshold=args.conf,
                        iou_threshold=args.iou,
                        generate_report_html=args.report_html)    


if __name__ == "__main__":
    run()

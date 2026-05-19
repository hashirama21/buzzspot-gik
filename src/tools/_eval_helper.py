
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# GEOMETRY / MATCHING

def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds: list, gts: list, iou_threshold: float,
                 class_aware: bool = True) -> tuple:
    """Greedy IoU matching, descending score.

    Each `pred` is a dict with keys {xyxy, score, cat_idx}.
    Each `gt`   is a dict with keys {xyxy, cat_idx, ann}.

    Returns:
      records    : list of {score, cat_idx, is_tp, gt_idx} in score-descending order.
      fn_indices : list of unmatched GT indices.
    """
    if not preds:
        return [], list(range(len(gts)))

    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    matched_gt: set = set()
    records = []

    for pi in order:
        p = preds[pi]
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched_gt:
                continue
            if class_aware and g["cat_idx"] != p["cat_idx"]:
                continue
            iou = compute_iou(p["xyxy"], g["xyxy"])
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_threshold and best_j >= 0:
            matched_gt.add(best_j)
            records.append({"score": p["score"], "cat_idx": p["cat_idx"],
                            "is_tp": True, "gt_idx": best_j})
        else:
            records.append({"score": p["score"], "cat_idx": p["cat_idx"],
                            "is_tp": False, "gt_idx": None})

    fn_indices = [j for j in range(len(gts)) if j not in matched_gt]
    return records, fn_indices


# METRICS

def compute_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    rp = np.linspace(0, 1, 101)
    out = np.zeros_like(rp)
    for i, r in enumerate(rp):
        mask = recalls >= r
        if mask.any():
            out[i] = precisions[mask].max()
    return float(out.mean())


def pr_curve(records: list, cls_idx: int, n_gt: int):
    """Return (recalls, precisions, AP) for one class. NaN AP if no GT and no preds."""
    entries = [(r["score"], r["is_tp"]) for r in records if r["cat_idx"] == cls_idx]
    if n_gt == 0 and not entries:
        return None, None, float("nan")
    if n_gt == 0:
        return None, None, 0.0
    if not entries:
        return np.array([0.0]), np.array([0.0]), 0.0

    entries.sort(key=lambda x: -x[0])
    tp = np.cumsum([int(e[1]) for e in entries])
    fp = np.cumsum([int(not e[1]) for e in entries])
    rec  = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1)
    return rec, prec, compute_ap_101(rec, prec)


def per_class_ap50(records: list, n_gt_per_class: dict, n_classes: int) -> list:
    return [pr_curve(records, c, n_gt_per_class.get(c, 0))[2]
            for c in range(n_classes)]


def attribute_stratified(tp_with_attr: list, fn_with_attr: list,
                         attr_labels: dict) -> dict:
    """Bucket TPs and FNs by an attribute value.

    tp_with_attr : list of (attr_value, score)
    fn_with_attr : list of attr_value
    attr_labels  : dict {attr_value: human_label}

    Returns: {attr_val: {n_gt, n_tp, n_fn, recall, mean_conf, tp_scores}}
    """
    counts = {v: {"n_tp": 0, "n_fn": 0, "tp_scores": []} for v in attr_labels}
    for v, s in tp_with_attr:
        if v in counts:
            counts[v]["n_tp"] += 1
            counts[v]["tp_scores"].append(float(s))
    for v in fn_with_attr:
        if v in counts:
            counts[v]["n_fn"] += 1

    out = {}
    for v, d in counts.items():
        n_gt = d["n_tp"] + d["n_fn"]
        out[v] = {
            "n_gt":      n_gt,
            "n_tp":      d["n_tp"],
            "n_fn":      d["n_fn"],
            "recall":    d["n_tp"] / n_gt if n_gt > 0 else float("nan"),
            "mean_conf": float(np.mean(d["tp_scores"])) if d["tp_scores"] else float("nan"),
            "tp_scores": d["tp_scores"],
        }
    return out


def run_pycocotools_eval(gt_path: str, pred_path: str, output_dir: Path) -> dict:
    """Run pycocotools COCOeval and return the 12 standard stats as a dict.

    If the pred file is a wrapped dict (`{annotations: [...]}`), a normalised
    list-only copy is written next to the outputs and used for the eval.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(gt_path)

    # Normalise pred file if necessary.
    with open(pred_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "annotations" in data:
        norm_path = output_dir / "_predictions_normalised.json"
        with open(norm_path, "w") as f:
            json.dump(data["annotations"], f)
        pred_path = str(norm_path)

    coco_dt = coco_gt.loadRes(pred_path)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.evaluate(); e.accumulate(); e.summarize()
    s = e.stats
    return {
        "mAP@50:95": float(s[0]),
        "mAP@50":    float(s[1]),
        "mAP@75":    float(s[2]),
        "mAP_small": float(s[3]),
        "mAP_medium":float(s[4]),
        "mAP_large": float(s[5]),
        "AR@1":      float(s[6]),
        "AR@10":     float(s[7]),
        "AR@100":    float(s[8]),
        "AR_small":  float(s[9]),
        "AR_medium": float(s[10]),
        "AR_large":  float(s[11]),
    }


# PLOTS

def plot_pr_curves(records, n_gt_per_class, class_names, out_path, iou_threshold):
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(class_names):
        rec, prec, ap = pr_curve(records, i, n_gt_per_class.get(i, 0))
        if rec is not None:
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                    color=cmap(i % 10), lw=2)
    ax.set(xlabel="Recall", ylabel="Precision",
           title=f"Precision-Recall Curves (IoU={iou_threshold})",
           xlim=(0, 1), ylim=(0, 1.05))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_confusion_matrix(confusion, class_names, out_path):
    labels = list(class_names) + ["BG/Miss"]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.8), max(6, n * 0.7)))
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = confusion / row_sums

    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (normalised by GT count)")

    for i in range(n):
        for j in range(n):
            v = int(confusion[i, j])
            if v > 0:
                color = "white" if norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=8, color=color)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_score_histogram(records, out_path):
    tp = [r["score"] for r in records if r["is_tp"]]
    fp = [r["score"] for r in records if not r["is_tp"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 40)
    ax.hist(tp, bins=bins, alpha=0.6, label=f"TP ({len(tp)})", color="seagreen")
    ax.hist(fp, bins=bins, alpha=0.6, label=f"FP ({len(fp)})", color="indianred")
    ax.set(xlabel="Confidence Score", ylabel="Count",
           title="Score Distribution: TP vs FP")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_per_class_ap(class_names, ap_values, out_path):
    valid = [(n, a) for n, a in zip(class_names, ap_values) if not np.isnan(a)]
    if not valid:
        return
    valid.sort(key=lambda x: x[1])
    names, aps = zip(*valid)
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.45)))
    colors = plt.get_cmap("RdYlGn")(np.array(aps))
    ax.barh(range(len(names)), aps, color=colors)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AP"); ax.set_title("Per-Class AP@50")
    for i, v in enumerate(aps):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_size_distribution(gt_areas, pred_areas, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    max_v = max(max(gt_areas, default=1), max(pred_areas, default=1)) * 1.1
    bins = np.linspace(0, max_v, 50)
    ax.hist(gt_areas, bins=bins, alpha=0.5, label=f"GT ({len(gt_areas)})", color="steelblue")
    ax.hist(pred_areas, bins=bins, alpha=0.5, label=f"Pred ({len(pred_areas)})", color="darkorange")
    ax.set(xlabel="√Area (px)", ylabel="Count",
           title="Detection Size Distribution")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_attr_recall(metrics, attr_name, labels, out_path):
    vals = sorted(labels.keys())
    xs   = [labels[v] for v in vals]
    rs   = [metrics.get(v, {}).get("recall", float("nan")) for v in vals]
    ns   = [metrics.get(v, {}).get("n_gt", 0) for v in vals]
    rs_safe = np.array([0.0 if (r != r) else r for r in rs])

    fig, ax = plt.subplots(figsize=(max(6, len(xs) * 1.4), 5))
    colors = plt.get_cmap("RdYlGn")(rs_safe)
    bars = ax.bar(xs, rs_safe, color=colors, edgecolor="grey")
    for bar, r, n in zip(bars, rs, ns):
        txt = f"{r:.3f}\n(n={n})" if r == r else f"n/a\n(n={n})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                txt, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set(xlabel=attr_name.capitalize(), ylabel="Recall",
           title=f"Recall by {attr_name.capitalize()} Level")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_attr_confidence(metrics, attr_name, labels, out_path):
    vals = sorted(labels.keys())
    data = [metrics.get(v, {}).get("tp_scores", []) for v in vals]
    xs   = [labels[v] for v in vals]
    valid = [(x, d) for x, d in zip(xs, data) if d]
    if not valid:
        return
    xs_v, data_v = zip(*valid)
    fig, ax = plt.subplots(figsize=(max(6, len(xs_v) * 1.4), 5))
    bp = ax.boxplot(data_v, tick_labels=xs_v, patch_artist=True)
    for patch, d in zip(bp["boxes"], data_v):
        patch.set_facecolor(plt.get_cmap("Blues")(0.4 + 0.4 * (np.mean(d) if d else 0)))
    ax.set(xlabel=attr_name.capitalize(), ylabel="Confidence Score",
           title=f"TP Confidence by {attr_name.capitalize()}", ylim=(0, 1.05))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_attr_overview(attr_inputs: dict, out_path):
    """Side-by-side recall plot for ALL detected attributes."""
    if not attr_inputs:
        return
    n = len(attr_inputs)
    fig, axes = plt.subplots(1, n, figsize=(max(6, 5 * n), 5), squeeze=False)
    fig.suptitle("Attribute Impact on Recall", fontsize=14, fontweight="bold")
    for ax, (attr_name, (metrics, labels)) in zip(axes[0], attr_inputs.items()):
        vals = sorted(labels.keys())
        xs   = [labels[v] for v in vals]
        rs   = [metrics.get(v, {}).get("recall", float("nan")) for v in vals]
        ns   = [metrics.get(v, {}).get("n_gt", 0) for v in vals]
        rs_safe = np.array([0.0 if (r != r) else r for r in rs])
        colors  = plt.get_cmap("RdYlGn")(rs_safe)
        bars = ax.bar(xs, rs_safe, color=colors, edgecolor="grey")
        for bar, r, nn in zip(bars, rs, ns):
            txt = f"{r:.3f}\n(n={nn})" if r == r else f"n/a\n(n={nn})"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    txt, ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.2)
        ax.set_title(f"Recall by {attr_name.capitalize()}")
        ax.set_ylabel("Recall")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


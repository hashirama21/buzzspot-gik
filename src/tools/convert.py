"""
Convert BuzzSet (`annotations/` + per-split frame folders) into
standard single-frame COCO and YOLO detection datasets, keeping only keyframes.

Input layout:
    BuzzSet/
        annotations/{train,valid,test}.json
        {train,valid,test}/{video_name}/{video_name}_{frame_idx}.jpg

Output layouts:

  standard_coco/
      annotations/{train,valid,test}.json
      {train,valid,test}/*.jpg               # flat folder per split

  standard_yolo/
      data.yaml
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt

Notes:
- Keyframes are detected via `image["is_keyframe"]`, with a fallback to
  `image["attributes"]["is_keyframe"]`.
- Image and annotation IDs are renumbered from 0 in the COCO output.
- YOLO class indices are 0-based, ordered by category id in the source JSON.
- The conventional YOLO split name `val` is used in place of `valid`.
"""
from pathlib import Path
import argparse
from collections import defaultdict
import csv
import json
import shutil

SPLITS = ["train", "valid", "test_devphase"]


def _is_keyframe(img: dict) -> bool:
    if "is_keyframe" in img:
        return bool(img["is_keyframe"])
    return bool(img.get("attributes", {}).get("is_keyframe", False))


def _resolve_src_image(split_dir: Path, file_name: str) -> Path | None:
    """Resolve a JSON file_name to an actual image path, robust to whether
    file_name is `video/frame.jpg` or just `frame.jpg`."""
    direct = split_dir / file_name
    if direct.is_file():
        return direct
    base = Path(file_name).name
    for video_dir in split_dir.iterdir():
        if video_dir.is_dir():
            candidate = video_dir / base
            if candidate.is_file():
                return candidate
    return None


def _src_json_path(buzzset_path: Path, split: str) -> Path:
    return buzzset_path / "annotations" / f"{split}.json"


def _load_categories(buzzset_path: Path) -> list[dict]:
    """Read categories from the first available split JSON."""
    for split in SPLITS:
        p = _src_json_path(buzzset_path, split)
        if p.exists():
            with p.open() as f:
                return json.load(f).get("categories", [])
    raise FileNotFoundError(f"No annotations found under {buzzset_path / 'annotations'}")


def to_coco(buzzset_path, output_dir) -> None:
    """Write a standard single-frame COCO dataset of keyframes only."""

    print(f"Converting BuzzSet at {buzzset_path} to COCO format at {output_dir}...")
    buzzset_path = Path(buzzset_path)
    output_dir = Path(output_dir)
    ann_out = output_dir / "annotations"
    ann_out.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        src_json = _src_json_path(buzzset_path, split)
        src_split_dir = buzzset_path / split

        if not src_json.exists():
            print(f"[coco/{split}] skipping (no annotations/{split}.json)")
            continue
        if not src_split_dir.is_dir():
            print(f"[coco/{split}] skipping (no {split}/ folder)")
            continue

        with src_json.open() as f:
            coco = json.load(f)

        keyframes = [img for img in coco["images"] if _is_keyframe(img)]
        kf_ids = {img["id"] for img in keyframes}
        kf_anns = [a for a in coco.get("annotations", []) if a["image_id"] in kf_ids]

        # Keep original IDs — no reindexing
        new_images = []
        for img in keyframes:
            new_img = dict(img)
            new_img["file_name"] = Path(img["file_name"]).name  # flatten
            new_images.append(new_img)

        new_anns = list(kf_anns)  # annotations already reference correct original image_ids

        new_coco = {
            "images": new_images,
            "annotations": new_anns,
            "categories": coco.get("categories", []),
        }

        # Copy images into the flat per-split folder
        dst_img_dir = output_dir / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)

        copied = skipped = missing = 0
        seen_names: set[str] = set()
        for img in keyframes:
            src_img = _resolve_src_image(src_split_dir, img["file_name"])
            if src_img is None:
                missing += 1
                continue
            name = Path(img["file_name"]).name
            if name in seen_names:
                print(f"[coco/{split}]  ! basename collision: {name} (flatten unsafe)")
            seen_names.add(name)
            dst_img = dst_img_dir / name
            if dst_img.exists():
                skipped += 1
            else:
                shutil.copy2(src_img, dst_img)
                copied += 1

        with (ann_out / f"{split}.json").open("w") as f:
            json.dump(new_coco, f)

        print(f"[coco/{split}] {len(keyframes)} keyframes "
              f"({len(coco['images'])} total), {len(new_anns)} annotations | "
              f"copied={copied} skipped={skipped} missing={missing}")


def to_yolo(buzzset_path, output_dir) -> None:
    """Write a standard YOLO dataset (Ultralytics layout) of keyframes only."""

    print(f"Converting BuzzSet at {buzzset_path} to YOLO format at {output_dir}...")
    buzzset_path = Path(buzzset_path)
    output_dir = Path(output_dir)

    categories = sorted(_load_categories(buzzset_path), key=lambda c: c["id"])
    cat_to_cls = {c["id"]: i for i, c in enumerate(categories)}
    class_names = [c["name"] for c in categories]

    split_name_map = {"train": "train", "valid": "val", "test_devphase": "test"}
    mapping_rows: list[dict] = []

    for split in SPLITS:
        src_json = _src_json_path(buzzset_path, split)
        src_split_dir = buzzset_path / split

        if not src_json.exists():
            print(f"[yolo/{split}] skipping (no annotations/{split}.json)")
            continue
        if not src_split_dir.is_dir():
            print(f"[yolo/{split}] skipping (no {split}/ folder)")
            continue

        with src_json.open() as f:
            coco = json.load(f)

        keyframes = [img for img in coco["images"] if _is_keyframe(img)]
        kf_ids = {img["id"] for img in keyframes}

        anns_by_img: dict[int, list[dict]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            if ann["image_id"] in kf_ids:
                anns_by_img[ann["image_id"]].append(ann)

        yolo_split = split_name_map[split]
        dst_img_dir = output_dir / "images" / yolo_split
        dst_lbl_dir = output_dir / "labels" / yolo_split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        copied = skipped = missing = 0
        for img in keyframes:
            src_img = _resolve_src_image(src_split_dir, img["file_name"])
            if src_img is None:
                missing += 1
                continue
            base = Path(img["file_name"]).name
            dst_img = dst_img_dir / base
            if dst_img.exists():
                skipped += 1
            else:
                shutil.copy2(src_img, dst_img)
                copied += 1

            mapping_rows.append({
                "image_id": img["id"],
                "split": yolo_split,
                "relative_path": f"images/{yolo_split}/{base}",
            })

            # YOLO label file (empty file = image with no objects = background)
            W, H = img["width"], img["height"]
            lines = []
            for ann in anns_by_img[img["id"]]:
                if ann["category_id"] not in cat_to_cls:
                    continue
                x, y, w, h = ann["bbox"]
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                nw = w / W
                nh = h / H
                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
                cls_id = cat_to_cls[ann["category_id"]]
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            (dst_lbl_dir / (Path(base).stem + ".txt")).write_text("\n".join(lines))

        print(f"[yolo/{split}] {len(keyframes)} keyframes "
              f"({len(coco['images'])} total) | "
              f"copied={copied} skipped={skipped} missing={missing}")

    # image_id mapping CSV
    mapping_path = output_dir / "image_id_mapping.csv"
    with mapping_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "split", "relative_path"])
        writer.writeheader()
        writer.writerows(mapping_rows)
    print(f"[yolo] wrote {mapping_path} ({len(mapping_rows)} entries)")

    # data.yaml
    yaml_lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        yaml_lines.append(f"  {i}: {name}")
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(f"[yolo] wrote {output_dir / 'data.yaml'} ({len(class_names)} classes: {class_names})")


def run():
    parser = argparse.ArgumentParser(description="Convert BuzzSet to single-frame YOLO or COCO format.")
    parser.add_argument("buzzset_path", type=Path, help="Path to original BuzzSet folder.")
    parser.add_argument("output_dir", type=Path, help="Output directory for converted dataset.")
    parser.add_argument("--format", choices=["coco", "yolo"], default="coco",
                        help="Output format (default: coco).")
    args = parser.parse_args()

    if args.format == "coco":
        to_coco(args.buzzset_path, args.output_dir)
    if args.format == "yolo":
        to_yolo(args.buzzset_path, args.output_dir)

if __name__ == "__main__":
    run()

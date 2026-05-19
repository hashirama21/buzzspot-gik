"""BuzzSet dataset utilities.

The BuzzSet annotation file follows COCO Video conventions:

* ``images`` entries carry an ``is_keyframe`` flag.
* Annotations only exist for frames where ``is_keyframe`` is True.
* Each keyframe is preceded by 5 unannotated context frames in the same video.
  The frame index is encoded as the trailing integer of ``file_name``,
  e.g. ``video03_frame00042.jpg`` -> frame 42.

Two Dataset classes are provided:

* :class:`BuzzSetSF` -- single-frame mode, yields one keyframe per sample.
* :class:`BuzzSetMF` -- multi-frame mode, yields a keyframe plus T preceding context frames.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image


_FRAME_NUM_RE = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")


def _frame_index_from_name(file_name: str) -> Optional[int]:
    """Extract the trailing integer from a file name (frame index)."""
    m = _FRAME_NUM_RE.search(Path(file_name).name)
    return int(m.group(1)) if m else None


#  dataclasses

@dataclass
class BuzzSetSFSample:
    """One keyframe plus its annotations (single-frame mode)."""
    image_id: int
    file_name: str
    image: np.ndarray              # H x W x 3, uint8 RGB
    boxes_xywh: np.ndarray         # N x 4, COCO xywh (absolute pixels)
    category_ids: np.ndarray       # N, int
    annotations: list[dict]        # raw COCO annotation dicts
    image_info: dict               # raw COCO image dict


@dataclass
class BuzzSetMFSample(BuzzSetSFSample):
    """A keyframe plus T preceding context frames (multi-frame mode).

    Context frames are ordered oldest -> newest, so ``context_images[-1]``
    """
    context_images: np.ndarray             # T x H x W x 3, uint8 RGB
    context_frame_indices: np.ndarray      # T, int (relative frame numbers)


# Datasets

class BuzzSetSF:
    """Single-frame BuzzSet dataset: yields one keyframe per sample.

    Parameters
    ----------
    root_folder : str | Path
        Dataset root. Expected layout::

            root_folder/
                annotations/{split}.json
                {split}/<file_name>

    split : str, default "train"
        Annotation split to load.
    """

    def __init__(
        self,
        root_folder: str | Path,
        split: str = "train",
    ):
        self.split = split
        self.annotation_file = Path(root_folder) / "annotations" / f"{split}.json"
        self.images_root = Path(root_folder) / split

        with self.annotation_file.open("r", encoding="utf-8") as f:
            self.coco: dict[str, Any] = json.load(f)

        # Index annotations by image_id.
        self.anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in self.coco.get("annotations", []):
            self.anns_by_image[ann["image_id"]].append(ann)

        # All-images lookups 
        all_images = self.coco.get("images", [])
        self._all_images_by_id = {img["id"]: img for img in all_images}
        self._frames_by_video: dict[Any, list[dict]] = defaultdict(list)
        for img in all_images:
            self._frames_by_video[img.get("video_id")].append(img)
        for v in self._frames_by_video.values():
            v.sort(key=lambda img: _frame_index_from_name(img["file_name"]) or 0)

        # (video_id, frame_index) -> image dict, for O(1) context lookup.
        self._frame_lookup: dict[tuple[Any, int], dict] = {}
        for img in all_images:
            f = _frame_index_from_name(img["file_name"])
            if f is not None:
                self._frame_lookup[(img.get("video_id"), f)] = img

        # Filter to keyframes only, then sort, then apply subclass-specific filter.
        keyframes = [img for img in all_images if img.get("is_keyframe", False)]
        keyframes.sort(
            key=lambda img: (
                img.get("video_id", 0),
                _frame_index_from_name(img["file_name"]) or 0,
                img["file_name"],
            )
        )
        self.images: list[dict] = keyframes

        self.categories: list[dict] = self.coco.get("categories", [])
        self.cat_id_to_name = {c["id"]: c["name"] for c in self.categories}


    def __len__(self) -> int:
        return len(self.images)

    def _load_keyframe(self, info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Load image + boxes + categories + raw annotations for a keyframe."""
        path = self.images_root / info["file_name"]
        image = np.array(Image.open(path).convert("RGB"))

        anns = self.anns_by_image.get(info["id"], [])
        if anns:
            boxes = np.array([a["bbox"] for a in anns], dtype=np.float32)
            cats = np.array([a["category_id"] for a in anns], dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            cats = np.zeros((0,), dtype=np.int64)
        return image, boxes, cats, anns

    def __getitem__(self, idx: int) -> BuzzSetSFSample:
        info = self.images[idx]
        image, boxes, cats, anns = self._load_keyframe(info)

        sample = BuzzSetSFSample(
            image_id=info["id"],
            file_name=info["file_name"],
            image=image,
            boxes_xywh=boxes,
            category_ids=cats,
            annotations=anns,
            image_info=info,
        )

        return sample


    def get_context_frames(self, keyframe_image_id: int, n: int = 5) -> list[dict]:
        """Return the N frames immediately preceding a keyframe (oldest first).
        """
        kf = self._all_images_by_id[keyframe_image_id]
        video_id = kf.get("video_id")
        kf_frame = _frame_index_from_name(kf["file_name"])
        siblings = self._frames_by_video.get(video_id, [])

        prev = []
        for img in siblings:
            f = _frame_index_from_name(img["file_name"])
            if f is None or kf_frame is None:
                continue
            if f < kf_frame:
                prev.append(img)
        return prev[-n:]


class BuzzSetMF(BuzzSetSF):
    """Multi-frame BuzzSet dataset: yields a keyframe plus T context frames.

    Parameters
    ----------
    root_folder, split :
        See :class:`BuzzSetSF`.
    num_context_frames : int, default 5
        Number of preceding frames to load per keyframe (T).
    """

    def __init__(
        self,
        root_folder: str | Path,
        split: str = "train",
        num_context_frames: int = 5,

    ):
        if num_context_frames < 1:
            raise ValueError("num_context_frames must be >= 1")
        self.num_context_frames = num_context_frames


        super().__init__(root_folder=root_folder, split=split)

    def _filter_keyframes(self, keyframes: list[dict]) -> list[dict]:
        return [kf for kf in keyframes if self._resolve_context(kf) is not None]

    def _resolve_context(self, kf: dict) -> Optional[list[dict]]:
        """Return T context image dicts (oldest -> newest) for this keyframe,
        or None if any required frame is missing."""
        video_id = kf.get("video_id")
        kf_frame = _frame_index_from_name(kf["file_name"])
        if kf_frame is None:
            return None

        frames: list[dict] = []
        # Walk oldest -> newest: offsets -T, ..., -1.
        for k in range(self.num_context_frames, 0, -1):
            target = kf_frame - k
            img = self._frame_lookup.get((video_id, target))
            if img is None:
                return None
            frames.append(img)
        return frames

    def __getitem__(self, idx: int) -> BuzzSetMFSample:
        info = self.images[idx]
        image, boxes, cats, anns = self._load_keyframe(info)

        # _filter_keyframes guarantees this is not None.
        ctx_infos = self._resolve_context(info)
        assert ctx_infos is not None

        ctx_images = np.stack(
            [
                np.array(
                    Image.open(self.images_root / ci["file_name"]).convert("RGB")
                )
                for ci in ctx_infos
            ],
            axis=0,
        )
        ctx_indices = np.array([ci['id']-info['id'] for ci in ctx_infos], dtype=np.int64)

        sample = BuzzSetMFSample(
            image_id=info["id"],
            file_name=info["file_name"],
            image=image,
            boxes_xywh=boxes,
            category_ids=cats,
            annotations=anns,
            image_info=info,
            context_images=ctx_images,
            context_frame_indices=ctx_indices,
        )

        return sample


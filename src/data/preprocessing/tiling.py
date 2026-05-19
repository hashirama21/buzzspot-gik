"""
TileExtractor — slices high-resolution frames into overlapping tiles
while remapping bounding-box annotations to tile coordinates.

Used at TRAINING TIME to exploit the native 256×256 BuzzSet structure,
and at INFERENCE TIME (SAHI-style) to handle full 1920×1080 frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class TileConfig:
    size: int = 256
    overlap: int = 32
    min_visibility: float = 0.1   # fraction of bbox area that must remain in tile


class TileExtractor:
    """Extracts overlapping tiles from an image and remaps annotations."""

    def __init__(
        self,
        size: int = 256,
        overlap: int = 32,
        min_visibility: float = 0.1,
    ) -> None:
        self.cfg = TileConfig(size=size, overlap=overlap, min_visibility=min_visibility)

    # Training: single sample → list of tile samples

    def extract(
        self,
        image: torch.Tensor,              # C×H×W float32
        target: Dict[str, torch.Tensor],  # XYXY normalised boxes
    ) -> List[Dict]:
        """Return a list of tile dicts, each with 'image' and 'target'."""
        C, H, W = image.shape
        stride = self.cfg.size - self.cfg.overlap
        results = []

        for y0 in range(0, max(1, H - self.cfg.overlap), stride):
            for x0 in range(0, max(1, W - self.cfg.overlap), stride):
                x1 = min(x0 + self.cfg.size, W)
                y1 = min(y0 + self.cfg.size, H)
                x0_ = max(0, x1 - self.cfg.size)
                y0_ = max(0, y1 - self.cfg.size)

                tile_img = image[:, y0_:y1, x0_:x1]

                tile_target = self._remap_boxes(
                    target, x0_, y0_, x1 - x0_, y1 - y0_, H, W
                )

                results.append(
                    {
                        "image": tile_img,
                        "target": tile_target,
                        "tile_offset": (x0_, y0_),
                        "img_id": target["image_id"],
                    }
                )

        return results if results else [{"image": image, "target": target, "tile_offset": (0, 0), "img_id": target["image_id"]}]

    # Inference: build tile grid from image array

    def get_tile_grid(self, H: int, W: int) -> List[Tuple[int, int, int, int]]:
        """Return list of (x0, y0, x1, y1) pixel coordinates."""
        stride = self.cfg.size - self.cfg.overlap
        tiles = []
        for y0 in range(0, max(1, H - self.cfg.overlap), stride):
            for x0 in range(0, max(1, W - self.cfg.overlap), stride):
                x1 = min(x0 + self.cfg.size, W)
                y1 = min(y0 + self.cfg.size, H)
                x0_ = max(0, x1 - self.cfg.size)
                y0_ = max(0, y1 - self.cfg.size)
                tiles.append((x0_, y0_, x1, y1))
        return tiles

    def remap_predictions_to_full(
        self,
        tile_boxes: np.ndarray,    # N×4 XYXY in tile pixel coords
        tile_offset: Tuple[int, int],
        full_H: int,
        full_W: int,
    ) -> np.ndarray:
        """Map tile-level predicted boxes back to full-image coordinates (normalised)."""
        x_off, y_off = tile_offset
        boxes = tile_boxes.copy().astype(np.float32)
        boxes[:, [0, 2]] += x_off
        boxes[:, [1, 3]] += y_off
        boxes[:, [0, 2]] /= full_W
        boxes[:, [1, 3]] /= full_H
        return np.clip(boxes, 0.0, 1.0)

    # Private

    def _remap_boxes(
        self,
        target: Dict[str, torch.Tensor],
        x0: int, y0: int,
        tw: int, th: int,
        img_H: int, img_W: int,
    ) -> Dict[str, torch.Tensor]:
        """Remap normalised XYXY boxes to tile coordinate space."""
        boxes = target["boxes"]  # (N, 4) normalised XYXY

        if len(boxes) == 0:
            return {**target, "boxes": boxes, "orig_size": torch.tensor([th, tw])}

        # Denormalise to full-image pixel coords
        px = boxes.clone()
        px[:, [0, 2]] *= img_W
        px[:, [1, 3]] *= img_H

        # Clip to tile window
        cx0 = px[:, 0].clamp(min=x0, max=x0 + tw)
        cy0 = px[:, 1].clamp(min=y0, max=y0 + th)
        cx1 = px[:, 2].clamp(min=x0, max=x0 + tw)
        cy1 = px[:, 3].clamp(min=y0, max=y0 + th)

        # Visibility filter
        orig_area = (px[:, 2] - px[:, 0]) * (px[:, 3] - px[:, 1])
        clipped_area = (cx1 - cx0) * (cy1 - cy0)
        visibility = clipped_area / (orig_area + 1e-6)
        keep = visibility >= self.cfg.min_visibility

        # Translate to tile-local coords and renormalise
        cx0 = (cx0[keep] - x0) / tw
        cy0 = (cy0[keep] - y0) / th
        cx1 = (cx1[keep] - x0) / tw
        cy1 = (cy1[keep] - y0) / th

        new_boxes = torch.stack([cx0, cy0, cx1, cy1], dim=1).clamp(0.0, 1.0)

        return {
            **{k: v[keep] if v.shape[0] == boxes.shape[0] else v for k, v in target.items()},
            "boxes": new_boxes,
            "orig_size": torch.tensor([th, tw]),
        }

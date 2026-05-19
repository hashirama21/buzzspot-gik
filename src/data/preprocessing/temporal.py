"""
TemporalFrameLoader — loads N previous frames relative to a keyframe
and builds the temporal tensor according to the chosen strategy.

All output tensors are ImageNet-normalized float32, so that ALL frames
in a window are treated uniformly regardless of which strategy is used.

Strategies
----------
stack   : concatenate all frames → C = N * 3 channels
diff    : keyframe + absolute differences → C = 3 + (N-1)*3  (= N*3)
flow    : keyframe + optical flow (Farneback / RAFT) → C = 3 + (N-1)*2
none    : keyframe only → C = 3
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from src.data.augmentations.transforms import normalize_tensor


class TemporalFrameLoader:
    """Loads temporal context frames and builds multi-channel tensors.

    The dataset stores frames as:
        video_N/video_N_000001.jpg  ← t-5
        video_N/video_N_000002.jpg
        ...
        video_N/video_N_000006.jpg  ← keyframe (annotated)

    Args:
        img_dir:    Root directory of the split (train/val/test).
        num_frames: Total frames including the keyframe (max 6).
        strategy:   "stack" | "diff" | "flow" | "none".
        img_size:   Resize all frames to (img_size, img_size). None = no resize.
    """

    def __init__(
        self,
        img_dir: Path,
        num_frames: int = 6,
        strategy: str = "stack",
        img_size: Optional[int] = 256,
    ) -> None:
        self.img_dir   = Path(img_dir)
        self.num_frames = max(1, min(num_frames, 6))
        self.strategy  = strategy
        self.img_size  = img_size

        if strategy not in ("stack", "diff", "flow", "none"):
            raise ValueError(f"Unknown temporal strategy: {strategy!r}")

        if strategy == "flow":
            self._raft = self._init_raft()

    # ──────────────────────────────────────────────────────────

    def load(self, keyframe_path: Path) -> List[np.ndarray]:
        """Return list of frames [t-(N-1), ..., t-1, t] as HWC uint8."""
        keyframe_path = Path(keyframe_path)
        context_paths = self._resolve_context_paths(keyframe_path)
        return [self._load_image(p) for p in context_paths]

    def build_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Convert loaded frames to a C×H×W float32 tensor (ImageNet-normalized)."""
        if self.strategy == "none" or self.num_frames == 1:
            return self._to_tensor(frames[-1])

        if self.strategy == "stack":
            return self._stack(frames)

        if self.strategy == "diff":
            return self._diff(frames)

        # strategy == "flow"
        return self._flow(frames)

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _resolve_context_paths(self, keyframe_path: Path) -> List[Path]:
        """Resolve paths for t-(num_frames-1) … t, padding with duplicates."""
        match = re.search(r"_(\d+)\.(jpg|jpeg|png)$", keyframe_path.name, re.IGNORECASE)
        if not match:
            return [keyframe_path] * self.num_frames

        frame_idx   = int(match.group(1))
        ext         = match.group(2)
        stem_prefix = keyframe_path.name[: match.start()]
        video_dir   = keyframe_path.parent

        paths: List[Path] = []
        for offset in range(self.num_frames - 1, -1, -1):
            target_idx = frame_idx - offset
            if target_idx < 1:
                paths.append(keyframe_path)
                continue
            candidate = video_dir / f"{stem_prefix}{target_idx:06d}.{ext}"
            if candidate.exists():
                paths.append(candidate)
            else:
                # Use nearest already-resolved frame (or keyframe if none yet)
                paths.append(paths[-1] if paths else keyframe_path)

        return paths

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            h = w = self.img_size or 256
            return np.zeros((h, w, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.img_size is not None:
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)
        return img

    @staticmethod
    def _to_tensor(frame: np.ndarray) -> torch.Tensor:
        """HWC uint8 → CHW float32, ImageNet-normalized."""
        t = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        return normalize_tensor(t)

    def _stack(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Concatenate all N frames → C = N*3 channels."""
        return torch.cat([self._to_tensor(f) for f in frames], dim=0)

    def _diff(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Keyframe + abs frame differences → C = 3 + (N-1)*3 = N*3."""
        keyframe_t = self._to_tensor(frames[-1])
        key_f32 = frames[-1].astype(np.float32)
        diffs = []
        for ctx_frame in frames[:-1]:
            diff = np.abs(ctx_frame.astype(np.float32) - key_f32) / 255.0
            diffs.append(
                torch.from_numpy(diff.transpose(2, 0, 1).astype(np.float32))
            )
        return torch.cat([keyframe_t] + diffs, dim=0)

    def _flow(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Keyframe + optical flow fields → C = 3 + (N-1)*2.

        Uses RAFT (torchvision) when available, falls back to Farneback.
        """
        keyframe_t = self._to_tensor(frames[-1])
        gray_key   = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2GRAY)
        flows = []

        for ctx_frame in frames[:-1]:
            if self._raft is not None:
                flow = self._raft_flow(ctx_frame, frames[-1])
            else:
                gray_ctx = cv2.cvtColor(ctx_frame, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    gray_ctx, gray_key, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )

            flow_norm = np.clip(flow / 20.0, -1.0, 1.0).astype(np.float32)
            flows.append(torch.from_numpy(flow_norm.transpose(2, 0, 1)))

        return torch.cat([keyframe_t] + flows, dim=0)

    def _raft_flow(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Compute flow with RAFT (torchvision ≥ 0.13)."""
        import torchvision.transforms.functional as TF
        from torchvision.models.optical_flow import raft_small

        def _to_raft(img):
            return TF.to_tensor(img).unsqueeze(0) * 255.0  # RAFT expects [0, 255]

        with torch.no_grad():
            flow = self._raft(_to_raft(src), _to_raft(dst))[-1]  # (1, 2, H, W)
        return flow[0].permute(1, 2, 0).numpy()  # H, W, 2

    @staticmethod
    def _init_raft():
        try:
            from torchvision.models.optical_flow import raft_small
            model = raft_small(pretrained=True).eval()
            return model
        except Exception:
            return None

    @property
    def out_channels(self) -> int:
        """Number of output channels given strategy + num_frames."""
        if self.strategy == "none" or self.num_frames == 1:
            return 3
        if self.strategy in ("stack", "diff"):
            return self.num_frames * 3
        if self.strategy == "flow":
            return 3 + (self.num_frames - 1) * 2
        return 3

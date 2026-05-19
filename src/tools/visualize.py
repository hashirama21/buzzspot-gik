"""Visualization helpers for BuzzSet samples.

Two entry points:

* :func:`show_keyframe` -- static matplotlib display of a keyframe with
  its bounding boxes. Works for both SF and MF samples.
* :func:`display_context_gif` -- animated GIF of the context frames
  followed by the annotated keyframe, returned as an IPython display
  object for inline use in a notebook. Requires an MF sample.

Use from a tutorial notebook like::

    from buzzset_loader import BuzzSetMF
    from visualise import show_keyframe, display_context_gif

    ds = BuzzSetMF("/path/to/buzzset", split="train", num_context_frames=5)
    sample = ds[0]

    show_keyframe(sample, category_names=ds.cat_id_to_name)
    display_context_gif(sample, category_names=ds.cat_id_to_name)
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

from .buzzset_loader import BuzzSetSFSample, BuzzSetMFSample


# Category colors

# RGB triples, picked for contrast on green/brown agricultural backgrounds.
_DEFAULT_COLORS: dict[int, tuple[int, int, int]] = {
    1: (255, 215, 0),     # bee        -> gold
    2: (255, 120, 0),     # bumblebee  -> orange
    3: (0, 200, 255),     # hoverfly   -> cyan
    4: (240, 60, 220),    # moth       -> magenta
}
_FALLBACK_COLOR = (255, 60, 60)
_CATEGORY_NAMES={1: "bee", 2: "bumblebee", 3: "hoverfly", 4: "moth"}

def _color_for(cat_id: int) -> tuple[int, int, int]:
    return _DEFAULT_COLORS.get(int(cat_id), _FALLBACK_COLOR)


# Static frame display
def draw_bboxes(ax: Axes,
                image: Image | str | Path,
                bboxes: List[Dict],
                category_ids: List[int],
                confidence: Optional[List[float]] = None,
                colors: Dict[int, Tuple[int, int, int]] = {
                    0: (255, 255, 0),
                    1: (0, 255, 0),
                    2: (255, 0, 0),
                    3: (0, 255, 255),
                    4: (255, 0, 255)
                },
                linewidth=2):
    """ Draw the given bounding boxes on the image.

    Args:
        ax (Axes): axes object to draw into.
        image (Image | str | Path): image that should be drawn.
        bboxes (List[Dict]): list of bounding boxes in the format (x, y, width, height), where (x,y) is the top-left corner of the bounding box.
        category_ids (List[int]): list of category IDs for each bounding box.
        confidence (List[float]): list of confidence scores for each bounding box.
        colors(Dict[int, tuple]): mapping of class ids to colors. Default: 1(bee) = (0,255,0), 2(bumblebee) = (0,255,0), 3(hoverfly)= (0,255,255), 4(moth) = (255,0,255)
        linewidth(int): thickness of the lines of the bounding box. Default: 2
    """
    if isinstance(image, (str, Path)):
        image = np.array(Image.open(image).convert("RGB"))
    ax.imshow(image)
    ax.set_axis_off()
    for i, bbox in enumerate(bboxes):
        x, y = bbox[0], bbox[1]
        color=np.array(colors[category_ids[i]]) / 255.0
        rect = patches.Rectangle((x, y),
                                bbox[2],
                                bbox[3],
                                linewidth=linewidth,
                                edgecolor=color,
                                facecolor='none')
        ax.add_patch(rect)
        ax.text(
                    x, max(0, y - 4), f"{confidence[i]:.2f}" if confidence is not None else str(category_ids[i]),
                    color="black", fontsize=9,
                    bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=1.5),
            )
        
        
def show_keyframe(
    sample: BuzzSetSFSample,
    ax: Optional[plt.Axes] = None,
    box_width: float = 2.0,
    show_labels: bool = True,
    figsize: tuple[float, float] = (10, 7),
) -> plt.Axes:
    """Display a keyframe with bounding boxes via matplotlib.

    Accepts SF or MF samples (only the keyframe image is shown).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.imshow(sample.image)
    ax.set_axis_off()

    for box, cat in zip(sample.boxes_xywh, sample.category_ids):
        x, y, w, h = box
        color = np.array(_color_for(int(cat))) / 255.0
        ax.add_patch(
            Rectangle(
                (x, y), w, h,
                linewidth=box_width, edgecolor=color, facecolor="none",
            )
        )
        if show_labels:
            name = _CATEGORY_NAMES.get(int(cat), str(cat))
            ax.text(
                x, max(0, y - 4), name,
                color="black", fontsize=9,
                bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=1.5),
            )

    n = len(sample.boxes_xywh)
    ax.set_title(f"{sample.file_name}  ({n} annotation{'s' if n != 1 else ''})" )
    return ax


# GIF building

def _resize_if_needed(img: Image.Image, max_size: Optional[int]) -> Image.Image:
    """Downsample so the longer side is at most ``max_size`` pixels."""
    if max_size is None:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= max_size:
        return img
    scale = max_size / longest
    return img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)


def _scale_boxes(boxes: np.ndarray, original_size: tuple[int, int],
                 new_size: tuple[int, int]) -> np.ndarray:
    """Scale xywh boxes from original to new image size."""
    if boxes.size == 0:
        return boxes
    sx = new_size[0] / original_size[0]
    sy = new_size[1] / original_size[1]
    scaled = boxes.copy().astype(np.float32)
    scaled[:, 0] *= sx
    scaled[:, 1] *= sy
    scaled[:, 2] *= sx
    scaled[:, 3] *= sy
    return scaled


def _draw_corner_label(draw: ImageDraw.ImageDraw, text: str, img_w: int,
                       font: ImageFont.ImageFont) -> None:
    """Black text on a translucent background, top-left of the frame."""
    pad = 4
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:  # very old PIL
        tw, th = draw.textsize(text, font=font)
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)


def _annotated_frame(
    image_array: np.ndarray,
    label: str,
    boxes_xywh: Optional[np.ndarray] = None,
    category_ids: Optional[np.ndarray] = None,
    category_names: Optional[dict[int, str]] = None,
    box_width: int = 3,
    max_size: Optional[int] = 800,
) -> Image.Image:
    """Render a single GIF frame: resize, draw boxes (if any), corner label."""
    original = Image.fromarray(image_array)
    img = _resize_if_needed(original, max_size).convert("RGB")

    if boxes_xywh is not None and len(boxes_xywh) > 0:
        boxes = _scale_boxes(boxes_xywh, original.size, img.size)
        draw = ImageDraw.Draw(img)
        for box, cat in zip(boxes, category_ids if category_ids is not None else []):
            x, y, w, h = box
            color = _color_for(int(cat))
            draw.rectangle([x, y, x + w, y + h], outline=color, width=box_width)
            if category_names is not None:
                name = category_names.get(int(cat), str(cat))
                draw.text((x + 3, max(0, y - 12)), name, fill='black',)
                #name = _CATEGORY_NAMES.get(int(cat), str(cat))

                
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    _draw_corner_label(draw, label, img.size[0], font)
    return img


def make_context_gif(
    sample: BuzzSetMFSample,
    output_path: Union[str, Path, None] = None,
    fps: float = 4.0,
    category_names: Optional[dict[int, str]] = _CATEGORY_NAMES,
    box_width: int = 3,
    hold_keyframe_frames: int = 4,
    max_size: Optional[int] = 800,
    loop: int = 0,
) -> bytes:
    """Build a GIF of context frames -> annotated keyframe.

    Parameters
    ----------
    sample : BuzzSetMFSample
        Multi-frame sample. SF samples raise ``TypeError``.
    output_path : path, optional
        If given, the GIF is also written to disk.
    fps : float
        Playback rate. 2 fps reads naturally for context inspection.
    category_names : dict, optional
        ``{category_id: display_name}``; pass ``ds.cat_id_to_name``.
    box_width : int
        Box outline width in pixels (post-resize).
    hold_keyframe_frames : int
        Times to repeat the keyframe at the end of the loop (so viewers
        can read the annotations before it loops).
    max_size : int, optional
        Longest-side resize cap. ``None`` keeps the original resolution
        (can produce very large GIFs).
    loop : int
        0 = infinite (default), otherwise number of loops.

    Returns
    -------
    bytes
        The encoded GIF.
    """
    if not isinstance(sample, BuzzSetMFSample):
        raise TypeError(
            "make_context_gif requires a BuzzSetMFSample (multi-frame). "
            "For single-frame samples, use show_keyframe instead."
        )

    keyframe_idx_in_name = Path(sample.file_name).stem
    duration_ms = int(round(1000 / fps))

    frames: list[Image.Image] = []

    # Context: oldest -> newest, no boxes.
    T = len(sample.context_images)
    for i, (ctx, ctx_frame_idx) in enumerate(
        zip(sample.context_images, sample.context_frame_indices)
    ):
        offset = -(T - i)  # -T, -T+1, ..., -1
        label = f"context  t{offset:+d}   frame {int(ctx_frame_idx)}"
        frames.append(_annotated_frame(ctx, label, max_size=max_size))

    # Keyframe with boxes, held for a few frames.
    key_label = f"keyframe t 0   {keyframe_idx_in_name}"
    annotated = _annotated_frame(
        sample.image, key_label,
        boxes_xywh=sample.boxes_xywh,
        category_ids=sample.category_ids,
        category_names=category_names,
        box_width=box_width,
        max_size=max_size,
    )
    for _ in range(max(1, hold_keyframe_frames)):
        frames.append(annotated)

    buf = BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        disposal=2,
        optimize=False,
    )
    data = buf.getvalue()

    if output_path is not None:
        Path(output_path).write_bytes(data)

    return data


def display_context_gif(
    sample: BuzzSetMFSample,
    fps: float = 4.0,
    **kwargs,
):
    """Build the context GIF and return an IPython.display.Image.
    """
    from IPython.display import Image as IPyImage

    data = make_context_gif(
        sample, fps=fps, **kwargs
    )
    return IPyImage(data=data, format="gif")

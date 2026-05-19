"""SAHI + RT-DETR single-frame inference for BuzzSet.

Wraps an Ultralytics RT-DETR checkpoint with SAHI's sliced prediction so that
small pollinators in high-resolution agricultural footage are still detected.
Outputs predictions in COCO results format, ready for `evaluate_bboxes.py`.

Usage:
    1) Edit the CONFIG block below.
    2) `python inference.py`

Or import:
    from inference import BuzzSetInferencer
    infer = BuzzSetInferencer(
        model_path="weights/rtdetr_buzzset.pt",
        annotation_file="annotations/val.json",
    )
    preds = infer.predict_image(image_rgb_uint8, image_id=42)
"""

from __future__ import annotations
from pathlib import Path

from typing import Any, List, Optional

import numpy as np
from PIL import Image

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

import numpy as np
from PIL import Image

from sahi.predict import get_sliced_prediction


from .buzzset_loader import BuzzSetSFSample
from rfdetr import RFDETRSmall
# CONFIG

DEVICE              = "cuda"           # or "cpu"
CONF_THRESHOLD      = 0.8
SLICE_HEIGHT        = 700
SLICE_WIDTH         = 700
OVERLAP_HEIGHT      = 0.20
OVERLAP_WIDTH       = 0.20
POSTPROCESS_TYPE    = "GREEDYNMM"        # or "NMS"
POSTPROCESS_MATCH   = "IOS"              
POSTPROCESS_THRESH  = 0.5
_PUBLIC_LINK = "https://uni-bonn.sciebo.de/s/DAEXJAKmeSzpJnn"
_PW = "XP6pLCsrM6"
_MODEL_WEIGHTS_PATH = Path(__file__).resolve().parent / "model_weights"/ "rfdetr_buzzset.pth"

def get_model_weights() -> Path:
    """Downloads pretrained model weights from a shared Sciebo link if not already present locally; returns the local path to the weights.
    """
    _MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _MODEL_WEIGHTS_PATH.exists():
        print(f"Accessing shared resource...")
        try:
            raise ImportError("pyocclient not installed. Run: pip install pyocclient")

            # List the files/folders in the share
            files = oc.list('/')
            for file_info in files:
                filename = file_info.get_name()
                if filename.endswith('.pth'):
                    print(f"Downloading {filename}...")
                    oc.get_file(f"/{filename}", str(_MODEL_WEIGHTS_PATH))
                    print("Download complete.")
                    return _MODEL_WEIGHTS_PATH
            print("Model weights file not found in the share.")
        except Exception as e:
            print(f"Failed to download: {e}")
    else:
        print(f"Model weights already exist at {_MODEL_WEIGHTS_PATH}")
        return _MODEL_WEIGHTS_PATH


class RFDETRDetectionModel(DetectionModel):
    """SAHI wrapper for locally trained RF-DETR checkpoints."""

    def load_model(self) -> None:
        weights = str(self.model_path)


        model = RFDETRSmall(pretrain_weights=weights)
        model.optimize_for_inference()


        self.model = model
        print(f"Loaded RF-DETR from {weights}")

    def perform_inference(self, image: np.ndarray) -> None:
        # Results stored to self._original_predictions per SAHI base class contract
        self._original_predictions = self.model.predict(
            Image.fromarray(image),
            threshold=self.confidence_threshold,
        )

    def convert_original_predictions(
                                    self,
                                    shift_amount: Optional[List] = None,
                                    full_shape: Optional[List] = None,
                                ) -> None:
        shift_amount = fix_shift_amount_list(shift_amount)
        full_shape   = fix_full_shape_list(full_shape)

        detections = self._original_predictions
        predictions: List[ObjectPrediction] = []

        if detections is None or len(detections) == 0:
            self._object_prediction_list_per_image = [predictions]
            return

        for box, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            predictions.append(
                ObjectPrediction(
                    bbox=box.tolist(),
                    score=float(score),
                    category_id=int(class_id),
                    category_name=str(int(class_id)),#self.category_mapping.get(str(int(class_id)), str(int(class_id))),
                    shift_amount=shift_amount[0],
                    full_shape=full_shape[0],
                )
            )

        self._object_prediction_list_per_image = [predictions]


class RFDETR_SAHI_Inferencer:
    """Single-frame inferencer wrapping SAHI sliced prediction"""

    def __init__(
        self,
        model_path: str | Path,
        device: str = DEVICE,
        conf_threshold: float = CONF_THRESHOLD,
        slice_height: int = SLICE_HEIGHT,
        slice_width: int = SLICE_WIDTH,
        overlap_height: float = OVERLAP_HEIGHT,
        overlap_width: float = OVERLAP_WIDTH,
        postprocess_type: str = POSTPROCESS_TYPE,
        postprocess_match: str = POSTPROCESS_MATCH,
        postprocess_thresh: float = POSTPROCESS_THRESH,
    ):

        self.model = RFDETRDetectionModel(model_path=str(model_path),confidence_threshold=conf_threshold,device=device)

        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height = overlap_height
        self.overlap_width = overlap_width
        self.postprocess_type = postprocess_type
        self.postprocess_match = postprocess_match
        self.postprocess_thresh = postprocess_thresh


    def predict_bbox(self, image: str | Path | np.ndarray,image_id: int=None) -> list[dict]:
        """Run sliced inference on a uint8 RGB array; return COCO results dicts."""
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))

        result = get_sliced_prediction(
            image,
            self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height,
            overlap_width_ratio=self.overlap_width,
            postprocess_type=self.postprocess_type,
            postprocess_match_metric=self.postprocess_match,
            postprocess_match_threshold=self.postprocess_thresh,
            verbose=0,
        )

        coco_results = result.to_coco_predictions()
        # Add image_id to each result for COCO evaluation compatibility
        
        for r in coco_results:
            if image_id is not None:
                r["image_id"] = image_id    
            r['category_id'] = int(r['category_id']+1)  # COCO category IDs are 1-indexed; adjust if your model is 0-indexed
            r['category_name'] = str(r['category_id'])
        return coco_results

    def predict_buzzsample(self, sample: BuzzSetSFSample) -> list[dict]:
        """Run sliced inference on a uint8 RGB array; return COCO results dicts."""
        return self.predict_bbox(sample.image, image_id=sample.image_id)



"""
Unit tests for BuzzSpot pipeline components.

Run:
    pytest tests/ -v
"""
import pytest
import numpy as np
import torch


# TemporalFrameLoader

class TestTemporalFrameLoader:
    def _make_frames(self, n: int, h: int = 64, w: int = 64):
        return [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]

    def test_stack_shape_and_dtype(self):
        from src.data.preprocessing.temporal import TemporalFrameLoader
        loader = TemporalFrameLoader.__new__(TemporalFrameLoader)
        loader.strategy   = "stack"
        loader.num_frames = 6
        loader.img_size   = 64

        t = loader.build_tensor(self._make_frames(6))
        assert t.shape == (18, 64, 64), f"Expected (18,64,64) got {t.shape}"
        assert t.dtype == torch.float32

    def test_diff_shape(self):
        from src.data.preprocessing.temporal import TemporalFrameLoader
        loader = TemporalFrameLoader.__new__(TemporalFrameLoader)
        loader.strategy   = "diff"
        loader.num_frames = 6
        loader.img_size   = 64

        t = loader.build_tensor(self._make_frames(6))
        assert t.shape == (18, 64, 64)

    def test_none_strategy(self):
        from src.data.preprocessing.temporal import TemporalFrameLoader
        loader = TemporalFrameLoader.__new__(TemporalFrameLoader)
        loader.strategy   = "none"
        loader.num_frames = 1
        loader.img_size   = 64

        t = loader.build_tensor(self._make_frames(1))
        assert t.shape == (3, 64, 64)
        assert t.dtype == torch.float32

    def test_imagenet_normalized(self):
        """build_tensor output must be ImageNet-normalized (not raw [0,1])."""
        from src.data.preprocessing.temporal import TemporalFrameLoader
        loader = TemporalFrameLoader.__new__(TemporalFrameLoader)
        loader.strategy   = "stack"
        loader.num_frames = 1
        loader.img_size   = 32

        # White frame: RGB (255,255,255) after /255 → 1.0
        # After ImageNet norm: (1.0 - 0.485) / 0.229 ≈ 2.25  (> 1.0)
        white = [np.full((32, 32, 3), 255, dtype=np.uint8)]
        t     = loader.build_tensor(white)
        assert t.max().item() > 1.0, "Expected values outside [0,1] after normalization"

    def test_out_channels_property(self):
        from src.data.preprocessing.temporal import TemporalFrameLoader
        for strategy, n_frames, expected_c in [
            ("stack", 6, 18),
            ("diff",  6, 18),
            ("flow",  6,  3 + 5 * 2),
            ("none",  1,  3),
        ]:
            loader = TemporalFrameLoader.__new__(TemporalFrameLoader)
            loader.strategy   = strategy
            loader.num_frames = n_frames
            assert loader.out_channels == expected_c, (
                f"{strategy}/{n_frames} → {loader.out_channels} != {expected_c}"
            )


# TileExtractor

class TestTileExtractor:
    def test_grid_covers_image(self):
        from src.data.preprocessing.tiling import TileExtractor
        te    = TileExtractor(size=256, overlap=32)
        tiles = te.get_tile_grid(1080, 1920)
        assert len(tiles) > 0
        for x0, y0, x1, y1 in tiles:
            assert x1 - x0 == 256
            assert y1 - y0 == 256
            assert x0 >= 0 and y0 >= 0
            assert x1 <= 1920 and y1 <= 1080

    def test_remap_to_full(self):
        from src.data.preprocessing.tiling import TileExtractor
        te       = TileExtractor(size=256, overlap=32)
        boxes    = np.array([[0.5, 0.5, 0.75, 0.75]])
        remapped = te.remap_predictions_to_full(boxes, (512, 256), 1080, 1920)
        expected_x = (0.5 * 256 + 512) / 1920
        assert abs(remapped[0, 0] - expected_x) < 0.001

    def test_bbox_remapping_preserves_count(self):
        from src.data.preprocessing.tiling import TileExtractor
        te     = TileExtractor(size=64, overlap=8)
        image  = torch.rand(3, 128, 128)
        target = {
            "boxes":     torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
            "labels":    torch.tensor([0]),
            "blur":      torch.tensor([0.0]),
            "occlusion": torch.tensor([0.0]),
            "area":      torch.tensor([0.64]),
            "image_id":  torch.tensor(1),
            "orig_size": torch.tensor([128, 128]),
        }

        tiles      = te.extract(image, target)
        assert len(tiles) > 1
        boxes_kept = [t["target"]["boxes"] for t in tiles if len(t["target"]["boxes"]) > 0]
        assert len(boxes_kept) > 0


# Losses

class TestLosses:
    def test_focal_loss_shape(self):
        from src.training.losses.criterion import FocalLoss
        loss    = FocalLoss(num_classes=4)
        logits  = torch.randn(2, 10, 4)
        targets = torch.randint(0, 4, (2, 10))
        out     = loss(logits, targets)
        assert out.ndim == 0
        assert out.item() >= 0.0

    def test_focal_loss_ignore(self):
        from src.training.losses.criterion import FocalLoss
        loss    = FocalLoss(num_classes=4)
        logits  = torch.randn(1, 5, 4)
        targets = torch.full((1, 5), -1)
        out     = loss(logits, targets)
        assert out.item() == 0.0

    def test_ciou_loss_zero_for_perfect(self):
        from src.training.losses.criterion import ciou_loss
        boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        out   = ciou_loss(boxes, boxes)
        assert out.item() < 1e-4

    def test_ciou_loss_positive(self):
        from src.training.losses.criterion import ciou_loss
        pred   = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        target = torch.tensor([[0.6, 0.6, 0.9, 0.9]])
        out    = ciou_loss(pred, target)
        assert out.item() > 0.0

    def test_ciou_loss_reduction_none_shape(self):
        from src.training.losses.criterion import ciou_loss
        pred   = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]])
        target = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.0, 0.0, 0.3, 0.3]])
        out    = ciou_loss(pred, target, reduction="none")
        assert out.shape == (2,)
        assert out[0].item() < 1e-4   # perfect
        assert out[1].item() > 0.0    # mismatch

    def test_criterion_returns_dict(self):
        from src.training.losses.criterion import BuzzSpotCriterion
        criterion = BuzzSpotCriterion(num_classes=4)

        B, Q    = 2, 10
        outputs = {
            "pred_logits": torch.randn(B, Q, 4),
            "pred_boxes":  torch.sigmoid(torch.randn(B, Q, 4)),
        }
        targets = [
            {
                "boxes":     torch.tensor([[0.1, 0.1, 0.4, 0.4]]),
                "labels":    torch.tensor([0]),
                "blur":      torch.tensor([0.0]),
                "occlusion": torch.tensor([0.0]),
                "image_id":  torch.tensor(i),
            }
            for i in range(B)
        ]

        losses = criterion(outputs, targets)
        assert "loss_total" in losses
        assert "loss_cls"   in losses
        assert "loss_ciou"  in losses
        assert losses["loss_total"].item() >= 0.0


# WBF

class TestWBF:
    def test_single_box_passthrough(self):
        from src.inference.predictors.predictor import weighted_boxes_fusion
        boxes  = [np.array([[0.1, 0.1, 0.5, 0.5]])]
        scores = [np.array([0.9])]
        labels = [np.array([0])]
        b, s, l = weighted_boxes_fusion(boxes, scores, labels)
        assert len(b) == 1
        assert s[0] == pytest.approx(0.9, abs=0.01)

    def test_overlapping_boxes_merged(self):
        from src.inference.predictors.predictor import weighted_boxes_fusion
        boxes  = [
            np.array([[0.1,  0.1,  0.5,  0.5]]),
            np.array([[0.11, 0.11, 0.51, 0.51]]),
        ]
        scores = [np.array([0.9]), np.array([0.85])]
        labels = [np.array([0]),   np.array([0])]
        b, s, l = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.5)
        assert len(b) == 1

    def test_different_classes_not_merged(self):
        from src.inference.predictors.predictor import weighted_boxes_fusion
        boxes  = [
            np.array([[0.1, 0.1, 0.5, 0.5]]),
            np.array([[0.1, 0.1, 0.5, 0.5]]),
        ]
        scores = [np.array([0.9]), np.array([0.8])]
        labels = [np.array([0]),   np.array([1])]
        b, s, l = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.5)
        assert len(b) == 2


# TTA inversion

class TestTTAInversion:
    def test_hflip(self):
        from src.inference.predictors.predictor import _invert_transform
        boxes = np.array([[0.1, 0.2, 0.4, 0.8]])
        inv   = _invert_transform(boxes, "hflip")
        assert abs(inv[0, 0] - (1.0 - 0.4)) < 1e-6
        assert abs(inv[0, 2] - (1.0 - 0.1)) < 1e-6
        assert inv[0, 1] == pytest.approx(0.2)

    def test_vflip(self):
        from src.inference.predictors.predictor import _invert_transform
        boxes = np.array([[0.1, 0.2, 0.4, 0.8]])
        inv   = _invert_transform(boxes, "vflip")
        assert abs(inv[0, 1] - (1.0 - 0.8)) < 1e-6
        assert abs(inv[0, 3] - (1.0 - 0.2)) < 1e-6

    def test_original_no_change(self):
        from src.inference.predictors.predictor import _invert_transform
        boxes = np.array([[0.1, 0.2, 0.4, 0.8]])
        inv   = _invert_transform(boxes, "original")
        np.testing.assert_array_almost_equal(inv, boxes)


# Model

class TestModel:
    def test_forward_no_temporal(self):
        from src.models.rfdetr_temporal import RFDETRTemporal
        model = RFDETRTemporal(
            num_classes=4, d_model=64, num_queries=10,
            num_frames=1, temporal_type="none",
            use_attribute_heads=False,
        )
        out = model(torch.randn(2, 3, 256, 256))
        assert out["pred_logits"].shape == (2, 10, 4)
        assert out["pred_boxes"].shape  == (2, 10, 4)

    def test_forward_with_attributes(self):
        from src.models.rfdetr_temporal import RFDETRTemporal
        model = RFDETRTemporal(
            num_classes=4, d_model=64, num_queries=10,
            num_frames=1, temporal_type="none",
            use_attribute_heads=True,
        )
        out = model(torch.randn(1, 3, 256, 256))
        assert "pred_attributes" in out
        assert out["pred_attributes"].shape == (1, 10, 2)

    def test_boxes_in_unit_range(self):
        from src.models.rfdetr_temporal import RFDETRTemporal
        model = RFDETRTemporal(
            num_classes=4, d_model=64, num_queries=10,
            num_frames=1, temporal_type="none",
            use_attribute_heads=False,
        )
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 256, 256))
        assert out["pred_boxes"].min().item() >= 0.0
        assert out["pred_boxes"].max().item() <= 1.0

    def test_invalid_temporal_type_raises(self):
        from src.models.rfdetr_temporal import RFDETRTemporal
        with pytest.raises(NotImplementedError):
            RFDETRTemporal(temporal_type="channel_stack")

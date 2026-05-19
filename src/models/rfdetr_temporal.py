"""
RF-DETR with optional temporal fusion head for BuzzSpot.

Wraps the official RF-DETR (DINOv2 backbone + RT-DETR decoder) and
adds a lightweight cross-attention Temporal Query module that
aggregates features from the 5 context frames into the current-frame
detection queries.

Architecture:
    [Frame t-5..t] ──► TemporalEncoder (per-frame DINOv2 features)
                              │
                    TemporalQueryFusion  ◄── Detection queries
                              │
                    RF-DETR Decoder ──► {boxes, classes, attributes}
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal encoding for temporal position (frame index 0 = oldest)."""

    def __init__(self, d_model: int, max_len: int = 16) -> None:
        super().__init__()
        # Ensure even d_model for clean sin/cos split
        d_even = d_model if d_model % 2 == 0 else d_model + 1
        pe  = torch.zeros(max_len, d_even)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_even, 2).float() * (-math.log(10000.0) / d_even)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe[:, :d_model])  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N, d_model) — add positional encoding over T."""
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).unsqueeze(2)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Query Fusion
# ─────────────────────────────────────────────────────────────────────────────

class TemporalQueryFusion(nn.Module):
    """Cross-attention between detection queries and temporal frame features.

    Given:
        queries: (B, Q, d_model) — current-frame detection queries
        memory:  (B, T, N, d_model) — per-frame spatial features

    Returns updated queries of same shape.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        num_frames: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.temporal_pe = TemporalPositionalEncoding(d_model, max_len=num_frames)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,  # (B, Q, d_model)
        memory: torch.Tensor,   # (B, T, N, d_model)
    ) -> torch.Tensor:
        B, T, N, D = memory.shape
        memory = self.temporal_pe(memory)
        kv = memory.view(B, T * N, D)
        attn_out, _ = self.cross_attn(queries, kv, kv)
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))
        return queries


# ─────────────────────────────────────────────────────────────────────────────
# Attribute head (blur / occlusion)
# ─────────────────────────────────────────────────────────────────────────────

class AttributeHead(nn.Module):
    """Binary classification head for blur and occlusion attributes."""

    def __init__(self, d_model: int = 256, num_attributes: int = 2) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_attributes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, Q, d_model) → (B, Q, num_attributes)."""
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# RF-DETR Temporal — full model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RFDETRTemporal(nn.Module):
    """RF-DETR + TemporalQueryFusion + AttributeHead.

    Attempts to import the official `rfdetr` package.  If unavailable,
    falls back to a minimal stub that preserves the correct tensor shapes
    for development and unit testing.

    Supported temporal_type values:
        "query_fusion"  — cross-attention between queries and temporal memory
        "none"          — no temporal fusion (single-frame mode)
    """

    def __init__(
        self,
        num_classes: int = 4,
        d_model: int = 256,
        num_queries: int = 300,
        num_frames: int = 6,
        temporal_type: str = "query_fusion",
        num_temporal_heads: int = 4,
        use_attribute_heads: bool = True,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes  = num_classes
        self.d_model      = d_model
        self.num_queries  = num_queries
        self.num_frames   = num_frames
        self.temporal_type = temporal_type

        if temporal_type not in ("query_fusion", "none"):
            raise NotImplementedError(
                f"temporal_type={temporal_type!r} is not implemented. "
                "Supported: 'query_fusion', 'none'."
            )

        # ── Core RF-DETR backbone + decoder ──────────────────
        self.rfdetr, self._using_real_rfdetr = self._build_backbone(
            num_classes, d_model, num_queries, pretrained
        )

        # ── Temporal fusion ───────────────────────────────────
        if temporal_type == "query_fusion" and num_frames > 1:
            self.temporal_fusion: Optional[nn.Module] = TemporalQueryFusion(
                d_model=d_model,
                num_heads=num_temporal_heads,
                num_frames=num_frames,
            )
        else:
            self.temporal_fusion = None

        # ── Attribute heads ───────────────────────────────────
        self.use_attribute_heads = use_attribute_heads
        if use_attribute_heads:
            self.attribute_head = AttributeHead(d_model=d_model, num_attributes=2)

    # ──────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,                        # (B, C, H, W)
        context_features: Optional[torch.Tensor] = None,  # (B, T-1, N, d_model)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values:     Current frame (or stacked multi-frame) tensor.
            context_features: Pre-computed encoder features for context frames.
                              None → temporal fusion is skipped.
        Returns:
            dict with keys: pred_logits, pred_boxes, [pred_attributes]
        """
        encoder_out = self.rfdetr.backbone(pixel_values)  # (B, N, d_model)

        queries = self.rfdetr.query_embed.weight.unsqueeze(0).expand(
            pixel_values.size(0), -1, -1
        )  # (B, Q, d_model)

        if self.temporal_fusion is not None and context_features is not None:
            current = encoder_out.unsqueeze(1)
            memory  = torch.cat([context_features, current], dim=1)  # (B, T, N, d)
            queries = self.temporal_fusion(queries, memory)

        decoder_out = self.rfdetr.decoder(tgt=queries, memory=encoder_out)

        pred_logits = self.rfdetr.class_head(decoder_out)
        pred_boxes  = self.rfdetr.bbox_head(decoder_out).sigmoid()

        out = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
        if self.use_attribute_heads:
            out["pred_attributes"] = self.attribute_head(decoder_out)

        return out

    def encode_context_frames(
        self, context_pixels: torch.Tensor  # (B, T, C, H, W)
    ) -> torch.Tensor:
        """Encode context frames → (B, T, N, d_model)."""
        B, T, C, H, W = context_pixels.shape
        flat     = context_pixels.view(B * T, C, H, W)
        features = self.rfdetr.backbone(flat)   # (B*T, N, d_model)
        N, D     = features.shape[1], features.shape[2]
        return features.view(B, T, N, D)

    # ──────────────────────────────────────────────────────────
    # Backbone builder
    # ──────────────────────────────────────────────────────────

    def _build_backbone(
        self,
        num_classes: int,
        d_model: int,
        num_queries: int,
        pretrained: bool,
    ) -> Tuple[nn.Module, bool]:
        """Try official rfdetr package; fall back to stub with warning."""
        try:
            from rfdetr import RFDETRLarge
            model = RFDETRLarge(
                num_classes=num_classes,
                pretrained=pretrained,
            )
            return model, True
        except ImportError:
            warnings.warn(
                "rfdetr package not installed — using stub backbone. "
                "Install with: pip install rfdetr  "
                "(see https://github.com/roboflow/rf-detr)",
                RuntimeWarning,
                stacklevel=3,
            )
            return self._build_rfdetr_stub(num_classes, d_model, num_queries), False

    def _build_rfdetr_stub(
        self, num_classes: int, d_model: int, num_queries: int
    ) -> nn.Module:
        """Minimal stub for architecture testing (preserves correct shapes)."""

        _d = d_model  # capture for closure

        class _Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self._conv       = nn.Conv2d(3, _d, 32, 32)
                self._act        = nn.GELU()
                self.query_embed = nn.Embedding(num_queries, _d)
                self.decoder     = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(_d, 4, batch_first=True), 2
                )
                self.class_head  = nn.Linear(_d, num_classes)
                self.bbox_head   = nn.Linear(_d, 4)

            # Expose a `backbone` attribute that returns (B, N, D)
            class _Backbone(nn.Module):
                def __init__(self, conv, act):
                    super().__init__()
                    self.conv = conv
                    self.act  = act

                def forward(self, x):
                    feat = self.act(self.conv(x))    # (B, D, h, w)
                    B, D, h, w = feat.shape
                    return feat.view(B, D, h * w).permute(0, 2, 1)  # (B, N, D)

        stub = _Stub()
        stub.backbone = _Stub._Backbone(stub._conv, stub._act)
        return stub

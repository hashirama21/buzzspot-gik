"""
RF-DETR with temporal fusion for BuzzSpot.

Architecture (real RF-DETR path):
    Stacked input (B, T*3, H, W)
      ├─ current frame      → rfdetr_module → pred_logits, pred_boxes
      └─ context frames     → ContextFrameEncoder → (B, T-1, N, d_model)
                                    ↓
                           QueryProjection(logits, boxes) → proxy queries
                                    ↓
                           TemporalQueryFusion (cross-attn) → refined queries
                                    ↓
                  temporal_delta_cls / temporal_delta_box  (residual)
                           → final pred_logits, pred_boxes

Architecture (stub path, development only):
    backbone → TemporalQueryFusion → decoder → heads
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Positional encoding

class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal encoding for temporal position (frame index 0 = oldest)."""

    def __init__(self, d_model: int, max_len: int = 16) -> None:
        super().__init__()
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


# Temporal Query Fusion

class TemporalQueryFusion(nn.Module):
    """Cross-attention between detection queries and temporal frame features.

    queries: (B, Q, d_model)  — current-frame query tokens
    memory:  (B, T, N, d_model) — per-frame spatial features

    Returns updated queries of the same shape.
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
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
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


# Context frame encoder

class ContextFrameEncoder(nn.Module):
    """Lightweight ConvNet that maps a context frame to spatial tokens.

    Input : (B, 3, 256, 256)
    Output: (B, N=64, d_model)   [stride-32 → 8×8 tokens]
    """

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3,       64,      4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64,      128,     4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128,     d_model, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model), nn.GELU(),
            nn.Conv2d(d_model, d_model, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
        )  # effective stride = 4×2×2×2 = 32 → 256/32 = 8 → 64 tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, N, d_model)"""
        feat = self.stem(x)          # (B, d_model, 8, 8)
        B, D, h, w = feat.shape
        return feat.flatten(2).permute(0, 2, 1)  # (B, 64, d_model)


# Query projection (real rfdetr path)

class QueryProjection(nn.Module):
    """Projects rfdetr outputs (logits + boxes) to d_model query tokens.

    These tokens serve as the query input to TemporalQueryFusion on the
    real RF-DETR path, where decoder query internals are inaccessible.
    """

    def __init__(self, num_logits: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_logits + 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, logits: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """logits: (B,Q,C)  boxes: (B,Q,4) → (B,Q,d_model)"""
        return self.proj(torch.cat([logits, boxes], dim=-1))


# Attribute head (blur / occlusion)

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


# RF-DETR Temporal — full model wrapper

class RFDETRTemporal(nn.Module):
    """RF-DETR + TemporalQueryFusion + optional AttributeHead.

    Real RF-DETR path:
        Stacked multi-channel input → extract context channels → ContextFrameEncoder
        → QueryProjection(rfdetr outputs) → TemporalQueryFusion → residual refinement.

    Stub path (no rfdetr package):
        Direct backbone access → TemporalQueryFusion on decoder queries.

    Supported temporal_type values:
        "query_fusion"  — temporal cross-attention (requires num_frames > 1)
        "none"          — single-frame, no temporal module
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

        self.num_classes   = num_classes
        self.d_model       = d_model
        self.num_queries   = num_queries
        self.num_frames    = num_frames
        self.temporal_type = temporal_type

        if temporal_type not in ("query_fusion", "none"):
            raise NotImplementedError(
                f"temporal_type={temporal_type!r} not supported. "
                "Use 'query_fusion' or 'none'."
            )

        _raw, self._using_real_rfdetr = self._build_backbone(
            num_classes, d_model, num_queries, pretrained
        )

        if self._using_real_rfdetr:
            self.rfdetr_module: nn.Module = self._extract_inner_model(_raw)
            self._rfdetr_wrapper = _raw
        else:
            self.rfdetr_module: nn.Module = _raw

        # TemporalQueryFusion (shared between both paths)
        if temporal_type == "query_fusion" and num_frames > 1:
            self.temporal_fusion: Optional[nn.Module] = TemporalQueryFusion(
                d_model=d_model, num_heads=num_temporal_heads, num_frames=num_frames,
            )
        else:
            self.temporal_fusion = None

        # Real rfdetr-specific temporal modules
        if self._using_real_rfdetr and self.temporal_fusion is not None:
            self.context_encoder    = ContextFrameEncoder(d_model)
            # Projects (logits+boxes) → d_model as proxy query tokens
            self.query_proj         = QueryProjection(num_classes + 1, d_model)
            # Residual refinement heads (small init keeps early training stable)
            self.temporal_delta_cls = nn.Linear(d_model, num_classes)
            self.temporal_delta_box = nn.Linear(d_model, 4)
            nn.init.zeros_(self.temporal_delta_cls.weight)
            nn.init.zeros_(self.temporal_delta_cls.bias)
            nn.init.zeros_(self.temporal_delta_box.weight)
            nn.init.zeros_(self.temporal_delta_box.bias)
        else:
            self.context_encoder    = None
            self.query_proj         = None
            self.temporal_delta_cls = None
            self.temporal_delta_box = None

        # AttributeHead — requires temporal fusion for a meaningful query signal.
        # On real rfdetr: uses refined queries from TemporalQueryFusion.
        # On stub path:   uses decoder output directly.
        self.use_attribute_heads = use_attribute_heads and self.temporal_fusion is not None
        if self.use_attribute_heads:
            self.attribute_head = AttributeHead(d_model, num_attributes=2)
        else:
            self.attribute_head = None
            if use_attribute_heads and self.temporal_fusion is None:
                warnings.warn(
                    "AttributeHead disabled: temporal_type='none' or num_frames=1 "
                    "provides no query signal. Enable temporal fusion to use it.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # Forward

    def forward(
        self,
        pixel_values: torch.Tensor,                        # (B, C, H, W)
        context_features: Optional[torch.Tensor] = None,  # (B, T-1, N, d_model)
    ) -> Dict[str, torch.Tensor]:
        if self._using_real_rfdetr:
            return self._forward_real(pixel_values, context_features)
        return self._forward_stub(pixel_values, context_features)

    def _forward_real(
        self,
        pixel_values: torch.Tensor,
        context_features: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Real RF-DETR path with temporal refinement.

        If pixel_values is a stacked multi-channel tensor (C = T*3), context
        frames are encoded on-the-fly from the non-current channels.
        If context_features is pre-computed (e.g. from the predictor), it is
        used directly.
        """
        # Extract current frame; optionally encode context from stacked channels
        if pixel_values.shape[1] > 3:
            B, C, H, W = pixel_values.shape
            T = C // 3
            current = pixel_values[:, -3:, :, :]   # last 3 ch = current frame

            if self.context_encoder is not None and context_features is None:
                ctx = pixel_values[:, :-3, :, :].reshape(B * (T - 1), 3, H, W)
                ctx_feat = self.context_encoder(ctx)          # (B*(T-1), N, d)
                N, D = ctx_feat.shape[1], ctx_feat.shape[2]
                context_features = ctx_feat.view(B, T - 1, N, D)
        else:
            current = pixel_values

        out = self.rfdetr_module(current)

        if isinstance(out, dict):
            pred_logits = out.get("pred_logits", out.get("logits"))
            pred_boxes  = out.get("pred_boxes",  out.get("boxes"))
        else:
            pred_logits = out.pred_logits
            pred_boxes  = out.pred_boxes

        result: Dict[str, torch.Tensor] = {}

        if self.temporal_fusion is not None and context_features is not None:
            # Project rfdetr output to query tokens, fuse with context, refine
            proxy   = self.query_proj(pred_logits, pred_boxes)      # (B, Q, d)
            refined = self.temporal_fusion(proxy, context_features)  # (B, Q, d)

            delta_cls = self.temporal_delta_cls(refined)
            delta_box = self.temporal_delta_box(refined).tanh() * 0.1  # small residual

            result["pred_logits"] = pred_logits[..., :self.num_classes] + delta_cls
            result["pred_boxes"]  = (pred_boxes + delta_box).clamp(0.0, 1.0)

            if self.use_attribute_heads:
                result["pred_attributes"] = self.attribute_head(refined)
        else:
            # No temporal context available: return rfdetr output as-is
            result["pred_logits"] = pred_logits[..., :self.num_classes]
            result["pred_boxes"]  = pred_boxes

        return result

    def _forward_stub(
        self,
        pixel_values: torch.Tensor,
        context_features: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Development stub path (full temporal fusion via backbone access)."""
        encoder_out = self.rfdetr_module.backbone(pixel_values)  # (B, N, d_model)

        queries = self.rfdetr_module.query_embed.weight.unsqueeze(0).expand(
            pixel_values.size(0), -1, -1
        )  # (B, Q, d_model)

        if self.temporal_fusion is not None and context_features is not None:
            current = encoder_out.unsqueeze(1)
            memory  = torch.cat([context_features, current], dim=1)
            queries = self.temporal_fusion(queries, memory)

        decoder_out = self.rfdetr_module.decoder(tgt=queries, memory=encoder_out)
        pred_logits = self.rfdetr_module.class_head(decoder_out)
        pred_boxes  = self.rfdetr_module.bbox_head(decoder_out).sigmoid()

        out = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
        if self.use_attribute_heads:
            out["pred_attributes"] = self.attribute_head(decoder_out)
        return out

    def encode_context_frames(
        self, context_pixels: torch.Tensor  # (B, T, C, H, W)
    ) -> Optional[torch.Tensor]:
        """Encode context frames → (B, T, N, d_model).

        Used by the predictor to pre-compute context features once per keyframe
        and reuse them across all tiles.
        """
        B, T, C, H, W = context_pixels.shape
        flat = context_pixels.view(B * T, C, H, W)
        if self._using_real_rfdetr:
            if self.context_encoder is None:
                return None
            feat = self.context_encoder(flat)
        else:
            feat = self.rfdetr_module.backbone(flat)
        N, D = feat.shape[1], feat.shape[2]
        return feat.view(B, T, N, D)

    # Inner-model extraction

    @staticmethod
    def _extract_inner_model(rfdetr_wrapper) -> nn.Module:
        """Find the nn.Module inside the rfdetr high-level wrapper."""
        mc = getattr(rfdetr_wrapper, 'model', None)
        if mc is None:
            raise RuntimeError(
                "rfdetr wrapper has no 'model' attribute — "
                f"available: {[a for a in dir(rfdetr_wrapper) if not a.startswith('__')]}"
            )
        for attr in ('model', '_model', 'module', 'net', '_net'):
            candidate = getattr(mc, attr, None)
            if isinstance(candidate, nn.Module):
                return candidate
        if isinstance(mc, nn.Module):
            return mc
        for attr in dir(mc):
            if attr.startswith('__'):
                continue
            try:
                candidate = getattr(mc, attr)
            except Exception:
                continue
            if isinstance(candidate, nn.Module):
                return candidate
        raise RuntimeError(
            f"Cannot find nn.Module inside rfdetr.model ({type(mc).__name__}). "
            f"Attributes: {[a for a in dir(mc) if not a.startswith('__')]}"
        )

    # Backbone builder

    def _build_backbone(
        self, num_classes: int, d_model: int, num_queries: int, pretrained: bool,
    ) -> Tuple[object, bool]:
        try:
            from rfdetr import RFDETRLarge
            kwargs = {"num_classes": num_classes}
            if not pretrained:
                kwargs["pretrain_weights"] = None
            return RFDETRLarge(**kwargs), True
        except ImportError:
            warnings.warn(
                "rfdetr package not installed — using stub backbone. "
                "Install with: pip install rfdetr",
                RuntimeWarning,
                stacklevel=3,
            )
            return self._build_rfdetr_stub(num_classes, d_model, num_queries), False

    def _build_rfdetr_stub(
        self, num_classes: int, d_model: int, num_queries: int
    ) -> nn.Module:
        """Minimal stub for unit testing (correct tensor shapes)."""
        _d = d_model

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

            class _Backbone(nn.Module):
                def __init__(self, conv, act):
                    super().__init__()
                    self.conv = conv
                    self.act  = act

                def forward(self, x):
                    feat = self.act(self.conv(x))
                    B, D, h, w = feat.shape
                    return feat.view(B, D, h * w).permute(0, 2, 1)

        stub = _Stub()
        stub.backbone = _Stub._Backbone(stub._conv, stub._act)
        return stub

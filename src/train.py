#!/usr/bin/env python3
"""
Train BuzzSpot detector.

Usage:
    python src/train.py
    python src/train.py training.experiment_name=rfdetr_mega_v1 model.name=rfdetr_mega

Hydra config overrides work on any key in configs/default.yaml.
"""
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import BuzzSetDataset
from src.data.augmentations.transforms import get_train_transforms, get_val_transforms
from src.data.augmentations.copy_paste import CopyPasteAugmentor
from src.models.rfdetr_temporal import RFDETRTemporal
from src.training import BuzzSpotTrainer
from src.training.losses.criterion import BuzzSpotCriterion

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_model(cfg: DictConfig) -> torch.nn.Module:
    return RFDETRTemporal(
        num_classes=cfg.data.num_classes,
        d_model=cfg.model.hidden_dim,
        num_queries=cfg.model.num_queries,
        num_frames=cfg.data.temporal.num_frames,
        temporal_type=cfg.model.temporal_head.type,
        num_temporal_heads=cfg.model.temporal_head.cross_attn_heads,
        use_attribute_heads=cfg.model.attribute_heads.enabled,
        pretrained=cfg.model.get("pretrained", True),
    )


def build_criterion(cfg: DictConfig) -> BuzzSpotCriterion:
    class_names     = list(cfg.data.class_names)
    per_class_gamma: dict = {}
    if hasattr(cfg.losses.classification, "per_class_gamma"):
        for cls_name, gamma in cfg.losses.classification.per_class_gamma.items():
            if cls_name in class_names:
                per_class_gamma[class_names.index(cls_name)] = float(gamma)

    return BuzzSpotCriterion(
        num_classes=cfg.data.num_classes,
        focal_gamma=cfg.losses.classification.gamma,
        focal_alpha=cfg.losses.classification.alpha,
        per_class_gamma=per_class_gamma or None,
        bbox_weight=cfg.losses.bbox_regression.weight,
        attr_weight=cfg.losses.attribute.weight,
        blur_instance_weight=cfg.training.curriculum.blur_weight,
        occlusion_instance_weight=cfg.training.curriculum.occlusion_weight,
    )


def build_dataloaders(cfg: DictConfig):
    tile_cfg = (
        dict(
            size=cfg.data.tile.size,
            overlap=cfg.data.tile.overlap,
            min_visibility=cfg.data.tile.min_visibility,
        )
        if cfg.data.tile.enabled
        else None
    )

    cp_cfg = cfg.augmentations.train.copy_paste
    copy_paste_aug = None
    if cp_cfg.enabled:
        copy_paste_aug = CopyPasteAugmentor(
            masks_dir=cp_cfg.masks_dir,
            class_names=list(cfg.data.class_names),
            target_classes=list(cp_cfg.classes),
            prob=cp_cfg.prob,
        )

    train_ds = BuzzSetDataset(
        ann_file=cfg.data.train_ann,
        img_dir=cfg.data.root,
        transforms=get_train_transforms(tile_size=cfg.data.tile.size),
        num_frames=cfg.data.temporal.num_frames,
        temporal_strategy=cfg.data.temporal.strategy,
        tile_cfg=tile_cfg,
        split="train",
        copy_paste_aug=copy_paste_aug,
    )

    val_ds = BuzzSetDataset(
        ann_file=cfg.data.val_ann,
        img_dir=cfg.data.root,
        transforms=get_val_transforms(tile_size=cfg.data.tile.size),
        num_frames=cfg.data.temporal.num_frames,
        temporal_strategy=cfg.data.temporal.strategy,
        split="val",
    )

    log.info("Dataset sizes — train: %d | val: %d", len(train_ds), len(val_ds))
    log.info("Class counts: %s", train_ds.class_counts)

    # Curriculum and WeightedRandomSampler cannot be combined:
    # curriculum modifies img_ids (changes dataset length), while the sampler
    # is built with fixed-length indices.  Use shuffle when curriculum is on.
    curriculum_on = cfg.training.curriculum.enabled
    if curriculum_on:
        train_ds.set_curriculum_phase(1)

    sampler = None
    use_shuffle = True
    if cfg.training.sampler.name == "class_balanced" and not curriculum_on:
        sample_weights = train_ds.compute_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        use_shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=use_shuffle,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=cfg.data.num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
        persistent_workers=cfg.data.num_workers > 0,
    )

    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate handling variable-size targets and optional tiling."""
    flat = []
    for item in batch:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    images  = torch.stack([b["image"]  for b in flat])
    targets = [b["target"] for b in flat]
    img_ids = [b["img_id"] for b in flat]

    result = {"image": images, "target": targets, "img_id": img_ids}

    if "context_features" in flat[0]:
        result["context_features"] = torch.stack(
            [b["context_features"] for b in flat]
        )

    return result


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    log.info(OmegaConf.to_yaml(cfg))
    set_seed(cfg.project.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    model            = build_model(cfg)
    criterion        = build_criterion(cfg)
    train_loader, val_loader = build_dataloaders(cfg)

    trainer = BuzzSpotTrainer(
        model=model,
        criterion=criterion,
        cfg=cfg,
        device=device,
    )

    trainer.fit(train_loader, val_loader)
    log.info("Training complete.")


if __name__ == "__main__":
    main()

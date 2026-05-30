"""
BuzzSpot Trainer — PyTorch training loop.

Features:
- Mixed precision (AMP) with fp16 / bf16 selection
- torch.compile support (PyTorch >= 2.0)
- Backbone freeze schedule (unfreeze at epoch N, rebuilds optimizer)
- Fused AdamW (CUDA, PyTorch >= 2.0)
- Curriculum learning (clean → all samples, with instance weighting)
- Gradient accumulation with correct last-batch handling
- Per-class mAP logging via WandB / TensorBoard
- Early stopping + checkpoint saving
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.evaluation.metrics.coco_eval import BuzzSpotEvaluator

log = logging.getLogger(__name__)


class BuzzSpotTrainer:
    """End-to-end trainer for the BuzzSpot pipeline.

    Args:
        model:      Detection model (RFDETRTemporal or similar).
        criterion:  Combined loss (BuzzSpotCriterion).
        cfg:        OmegaConf config loaded from configs/default.yaml.
        device:     torch.device for training.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        cfg: DictConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model     = model
        self.criterion = criterion
        self.cfg       = cfg
        self.device    = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # AMP dtype: bf16 preferred on Ampere+ (A100, L4); fp16 on T4/V100
        amp_dtype_str = cfg.training.get("amp_dtype", "fp16")
        self._amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
        self.use_amp = cfg.training.precision == 16 and self.device.type == "cuda"

        # Linear LR scaling: reference effective batch = 32 (4 samples × 8 accum steps).
        eff_batch = cfg.training.batch_size * cfg.training.accumulate_grad_batches
        self._lr_scale = eff_batch / 32

        self.model.to(self.device)
        self.criterion.to(self.device)

        # Backbone freeze: train decoder+temporal head first, unfreeze at epoch N
        self._freeze_epochs = cfg.training.get("freeze_backbone_epochs", 0)
        if self._freeze_epochs > 0:
            self._set_backbone_grad(False)
            log.info("Backbone frozen for first %d epochs.", self._freeze_epochs)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler    = GradScaler(device=self.device.type, enabled=self.use_amp)

        # torch.compile: ~60s warmup cost, then +15-35% throughput
        if cfg.training.get("compile", False) and hasattr(torch, "compile"):
            log.info("Compiling model with torch.compile …")
            self.model = torch.compile(self.model)

        self.current_epoch     = 0
        self.global_step       = 0
        self.best_metric       = 0.0
        self.epochs_no_improve = 0

        self.out_dir = Path(cfg.project.output_dir) / cfg.project.experiment_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._evaluator = BuzzSpotEvaluator(
            ann_file=cfg.data.val_ann,
            class_names=list(cfg.data.class_names),
        )

        raw_monitor = cfg.training.checkpoint.monitor
        self._monitor_key = raw_monitor.rsplit("/", 1)[-1]

        self.logger = self._build_logger()

    # Main train loop

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        cfg = self.cfg.training

        for epoch in range(self.current_epoch, cfg.epochs):
            self.current_epoch = epoch

            # Unfreeze backbone and rebuild optimizer at the scheduled epoch
            if self._freeze_epochs > 0 and epoch == self._freeze_epochs:
                self._set_backbone_grad(True)
                self.optimizer = self._build_optimizer()
                self.scheduler = self._build_scheduler()
                log.info("[Epoch %d] Backbone unfrozen — optimizer rebuilt.", epoch)

            if cfg.curriculum.enabled:
                self._apply_curriculum_phase(train_loader, epoch)

            train_metrics = self._train_epoch(train_loader)
            self._log(train_metrics, prefix="train", epoch=epoch)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            val_metrics = self._val_epoch(val_loader)
            self._log(val_metrics, prefix="val", epoch=epoch)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            self.scheduler.step()

            metric  = val_metrics.get(self._monitor_key, 0.0)
            improved = metric > self.best_metric
            self._save_checkpoint(metric, epoch, improved)

            if cfg.early_stopping.enabled and self._check_early_stop(improved):
                log.info("Early stopping triggered at epoch %d.", epoch)
                break

            log.info(
                "[Epoch %03d] train_loss=%.4f  val_mAP@50:95=%.4f",
                epoch,
                train_metrics.get("loss_total", float("nan")),
                val_metrics.get("mAP_50_95", 0.0),
            )

    # Training step

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        accum_steps = self.cfg.training.accumulate_grad_batches
        log_every   = self.cfg.training.get("log_every_n_steps", 50)
        total_losses: Dict[str, float] = {}
        n_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Train {self.current_epoch:03d}", leave=False, dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            images  = batch["image"].to(self.device, non_blocking=True)
            targets = [
                {k: v.to(self.device, non_blocking=True) for k, v in t.items()}
                for t in batch["target"]
            ]
            context = batch.get("context_features")
            if context is not None:
                context = context.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp,
                          dtype=self._amp_dtype):
                outputs = self.model(images, context_features=context)
                losses  = self.criterion(outputs, targets)
                loss    = losses["loss_total"] / accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                self._optimizer_step()
                if self.global_step % log_every == 0:
                    self._log_step(losses)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{losses['loss_total'].item():.3f}",
                cls=f"{losses['loss_cls'].item():.3f}",
                box=f"{losses['loss_ciou'].item():.3f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.1e}",
            )

        if n_batches % accum_steps != 0:
            self._optimizer_step()

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def _log_step(self, losses: Dict[str, torch.Tensor]) -> None:
        if self.logger is None:
            return
        payload = {f"step/{k}": v.item() for k, v in losses.items()}
        payload["step/lr"] = self.optimizer.param_groups[0]["lr"]
        try:
            self.logger.log(payload, step=self.global_step)
        except Exception as exc:
            log.debug("wandb step log failed: %s", exc)

    # Validation step

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        self._evaluator.reset()

        pbar = tqdm(loader, desc="Val  ", leave=False, dynamic_ncols=True)
        for batch in pbar:
            images  = batch["image"].to(self.device, non_blocking=True)
            targets = [
                {k: v.to(self.device, non_blocking=True) for k, v in t.items()}
                for t in batch["target"]
            ]
            context = batch.get("context_features")
            if context is not None:
                context = context.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp,
                          dtype=self._amp_dtype):
                outputs = self.model(images, context_features=context)

            self._evaluator.update(outputs, targets)

        return self._evaluator.summarize()

    # Backbone freeze helpers

    def _set_backbone_grad(self, requires_grad: bool) -> None:
        for name, p in self.model.named_parameters():
            if "backbone" in name:
                p.requires_grad_(requires_grad)

    # Curriculum scheduling

    def _apply_curriculum_phase(self, loader: DataLoader, epoch: int) -> None:
        if not hasattr(loader.dataset, "set_curriculum_phase"):
            return
        phase1  = self.cfg.training.curriculum.phase1_epochs
        ds      = loader.dataset
        target  = 1 if epoch < phase1 else 2
        if getattr(ds, "_curriculum_phase", None) != target:
            ds.set_curriculum_phase(target)
            log.info("[Curriculum] Epoch %d → phase %d.", epoch, target)

    # Checkpoint & early stopping

    def _save_checkpoint(self, metric: float, epoch: int, improved: bool) -> None:
        ckpt = {
            "epoch":            epoch,
            "global_step":      self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "scheduler_state":  self.scheduler.state_dict(),
            "best_metric":      self.best_metric,
            "config":           OmegaConf.to_container(self.cfg),
        }
        torch.save(ckpt, self.out_dir / "last.pth")

        if improved:
            self.best_metric = metric
            torch.save(ckpt, self.out_dir / "best.pth")
            log.info("  ✓ New best checkpoint — %.4f", metric)

    def _check_early_stop(self, improved: bool) -> bool:
        if improved:
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        return self.epochs_no_improve >= self.cfg.training.early_stopping.patience

    # Optimiser & scheduler builders

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.cfg.training.optimizer
        backbone_params, other_params = [], []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(p)
            else:
                other_params.append(p)

        lr = opt_cfg.lr * self._lr_scale
        param_groups = [
            {"params": other_params,    "lr": lr},
            {"params": backbone_params, "lr": lr * opt_cfg.backbone_lr_factor},
        ]

        # Fused AdamW: ~20% faster optimizer step on CUDA (PyTorch >= 2.0)
        use_fused = self.device.type == "cuda"
        try:
            return torch.optim.AdamW(
                param_groups,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
                fused=use_fused,
            )
        except TypeError:
            return torch.optim.AdamW(
                param_groups,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
            )

    def _build_scheduler(self):
        sch_cfg = self.cfg.training.scheduler
        warmup  = sch_cfg.warmup_epochs
        total   = self.cfg.training.epochs
        min_lr  = sch_cfg.min_lr
        base_lr = self.cfg.training.optimizer.lr

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return max(epoch / max(warmup, 1), 1e-6)
            progress = (epoch - warmup) / max(total - warmup, 1)
            cosine   = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr / base_lr + (1 - min_lr / base_lr) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # Logger

    def _build_logger(self):
        try:
            import wandb
            wandb.init(
                project=self.cfg.project.name,
                name=self.cfg.project.experiment_name,
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )
            return wandb
        except ImportError:
            log.warning("wandb not installed — logging to stdout only.")
            return None
        except Exception as exc:
            log.warning("wandb init failed (%s) — logging to stdout only.", exc)
            return None

    def _log(self, metrics: Dict[str, float], prefix: str, epoch: int) -> None:
        if self.logger is not None:
            try:
                self.logger.log(
                    {f"{prefix}/{k}": v for k, v in metrics.items()},
                    step=epoch,
                )
            except Exception as exc:
                log.debug("wandb log failed: %s", exc)

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from .base import BaseTrainer, Batch, TrainConfig
from .utils import OptimConfig, ExportConfig, export_artifacts
from .losses import DSMLoss


class ContinuousTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optim_cfg: OptimConfig,
        train_cfg: TrainConfig,
        *,
        dsm_sigma: float = 1.0,
        n_noise: int = 1,
        temperature: float = 1.0,
        monitor_fn: Optional[Callable[[nn.Module, torch.utils.data.DataLoader, torch.device], float]] = None,
        export_cfg: Optional[ExportConfig] = None,
    ) -> None:
        super().__init__(
            model=model,
            optim_cfg=optim_cfg,
            train_cfg=train_cfg,
            monitor_fn=monitor_fn,
        )

        setattr(self.train_cfg, "eval_requires_grad", True)
        self.dsm_loss = DSMLoss(sigma=float(dsm_sigma), n_noise=int(n_noise), temperature=float(temperature))
        self.export_cfg = export_cfg if export_cfg is not None else ExportConfig()
            
    def training_step(self, batch: Batch) -> torch.Tensor:
        xb, yb = batch
        loss = self.dsm_loss(self.model, xb, yb)
        return loss

    def validation_step(self, batch: Batch) -> torch.Tensor:
        return self.training_step(batch)
    
    def _export_if_enabled(self, result=None):
        export_artifacts(self.model, self.export_cfg, result)

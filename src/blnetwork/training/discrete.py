from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from .base import BaseTrainer, Batch, TrainConfig
from .utils import OptimConfig, ExportConfig, export_artifacts
from .losses import CELoss


class DiscreteTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optim_cfg: OptimConfig,
        train_cfg: TrainConfig,
        monitor_fn: Optional[Callable[[nn.Module, torch.utils.data.DataLoader, torch.device], float]] = None,
        export_cfg: Optional[ExportConfig] = None,
    ) -> None:
        super().__init__(
            model=model,
            optim_cfg=optim_cfg,
            train_cfg=train_cfg,
            monitor_fn=monitor_fn,
        )
        self.loss_fn = CELoss()
        self.export_cfg = export_cfg if export_cfg is not None else ExportConfig()

    def training_step(self, batch: Batch) -> torch.Tensor:
        x, y = batch
        bl_vec = self.model.logits(x)   
        return self.loss_fn(bl_vec, y)    

    def validation_step(self, batch: Batch) -> torch.Tensor:
        return self.training_step(batch)
    
    def _export_if_enabled(self, result=None):
        export_artifacts(self.model, self.export_cfg, result)

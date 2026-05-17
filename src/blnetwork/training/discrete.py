from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from .base import BaseTrainer, Batch, TrainConfig
from .utils import OptimConfig, ExportConfig, ModelConfig, export_artifacts
from .losses import CELoss


class DiscreteTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module | ModelConfig | None = None,
        optim_cfg: Optional[OptimConfig] = None,
        train_cfg: Optional[TrainConfig] = None,
        *,
        model_cfg: Optional[ModelConfig] = None,
        monitor_fn: Optional[Callable[[nn.Module, torch.utils.data.DataLoader, torch.device], float]] = None,
        export_cfg: Optional[ExportConfig] = None,
    ) -> None:
        if model is not None and model_cfg is not None:
            raise TypeError("Pass either model=... or model_cfg=..., but not both.")
        if model is None:
            model = model_cfg
        if model is None:
            raise TypeError("Missing model. Pass model=... or model_cfg=....")

        if isinstance(model, ModelConfig):
            model = model.build()
        elif not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module or ModelConfig.")

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

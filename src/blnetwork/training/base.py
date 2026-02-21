from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader  
from . import utils as U

Batch = Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]


@dataclass
class EarlyStop:
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "min"  

    best_value: Optional[float] = None
    best_epoch: int = -1
    patience_counter: int = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < (self.best_value - self.min_delta)
        if self.mode == "max":
            return value > (self.best_value + self.min_delta)
        raise ValueError("mode must be 'min' or 'max'.")

    def step(self, value: float, epoch: int) -> bool:
        value = float(value)
        if self._is_improvement(value):
            self.best_value = value
            self.best_epoch = int(epoch)
            self.patience_counter = 0
            return False

        self.patience_counter += 1
        return self.patience_counter >= self.patience


@dataclass
class TrainConfig:
    max_epochs: int = 200
    batch_size: int = 256  
    num_workers: int = 0   
    pin_memory: Optional[bool] = None  

    grad_clip_norm: Optional[float] = None
    mixed_precision: bool = False

    early_stop: bool = True
    patience: int = 20
    min_delta: float = 0.0
    mode: str = "min"
    monitor_name: str = "val_loss"

    log_every: int = 1
    verbose: bool = False  
    device: Optional[str] = None

    eval_requires_grad: bool = False
    
    seed: Optional[int] = None
    deterministic: bool = False


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        optim_cfg: U.OptimConfig,
        train_cfg: TrainConfig,
        monitor_fn: Optional[Callable[[nn.Module, DataLoader, torch.device], float]] = None,
    ) -> None:
        self.model = model
        self.optim_cfg = optim_cfg
        self.train_cfg = train_cfg

        self.device = U.auto_device(train_cfg.device)
        self.model.to(self.device)
        self.monitor_fn = monitor_fn
        self.monitor_name = str(train_cfg.monitor_name)
        if self.monitor_fn is not None and self.monitor_name == "val_loss":
            self.monitor_name = "val_metric"

        self.optimizer: Optional[torch.optim.Optimizer] = None  

        use_mixed_precision = bool(train_cfg.mixed_precision)
        if use_mixed_precision and self.device.type != "cuda":
            if train_cfg.verbose:
                print("Warning: mixed_precision only works on CUDA. Disabled for CPU training.")
            use_mixed_precision = False
        
        self.amp: U.AMPState = U.make_amp(self.device, mixed_precision=use_mixed_precision)
        self.early_stop: Optional[EarlyStop] = None
        if train_cfg.early_stop:
            self.early_stop = EarlyStop(
                patience=train_cfg.patience,
                min_delta=train_cfg.min_delta,
                mode=train_cfg.mode,
            )

        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_monitor": [],
        }

    def training_step(self, batch: Batch) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, batch: Batch) -> torch.Tensor:
        return self.training_step(batch)
    
    def _make_loader(self, dataset: Dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        cfg = self.train_cfg
        pin_memory = cfg.pin_memory if cfg.pin_memory is not None else (self.device.type == "cuda")
        
        return DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=bool(shuffle),
            num_workers=int(cfg.num_workers),
            pin_memory=bool(pin_memory),
            drop_last=bool(drop_last),
        )
    
    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        n_batches = 0

        for batch in loader:
            batch = U.move_batch_to_device(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with U.autocast_ctx(self.amp):
                loss = self.training_step(batch)

            if self.amp.scaler.is_enabled():
                self.amp.scaler.scale(loss).backward()

                if self.train_cfg.grad_clip_norm is not None:
                    self.amp.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip_norm)

                self.amp.scaler.step(self.optimizer)
                self.amp.scaler.update()
            else:
                loss.backward()
                if self.train_cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip_norm)
                self.optimizer.step()

            total += float(loss.detach().item())
            n_batches += 1

        return total / max(n_batches, 1)

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        n_batches = 0

        grad_enabled = bool(getattr(self.train_cfg, "eval_requires_grad", False))
        with torch.set_grad_enabled(grad_enabled):
            for batch in loader:
                batch = U.move_batch_to_device(batch, self.device)
                with U.autocast_ctx(self.amp):
                    loss = self.validation_step(batch)
                total += float(loss.detach().item())
                n_batches += 1

        return total / max(n_batches, 1)

    def evaluate_monitor(self, loader: DataLoader, fallback_loss: float) -> float:
        if self.monitor_fn is None:
            return float(fallback_loss)

        self.model.eval()
        with torch.no_grad():
            return float(self.monitor_fn(self.model, loader, self.device))
    
    def fit(
        self,
        X: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        *,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        return_state_dict: bool = False,
    ) -> Dict[str, Any]:
        cfg = self.train_cfg
        if cfg.seed is not None:
            U.set_seed(cfg.seed, deterministic=cfg.deterministic)

        if train_loader is not None:
            if X is not None or y is not None:
                raise ValueError("Cannot specify both (X, y) and train_loader.")
            train_dl = train_loader
            val_dl = val_loader
        elif X is not None and y is not None:
            train_ds = TensorDataset(X, y)
            train_dl = self._make_loader(train_ds, shuffle=shuffle, drop_last=drop_last)
            val_dl = None
            if X_val is not None and y_val is not None:
                val_ds = TensorDataset(X_val, y_val)
                val_dl = self._make_loader(val_ds, shuffle=False, drop_last=False)
        else:
            raise ValueError(
                "You must provide either:\n"
                "1. X and y tensors: fit(X, y)\n"
                "2. train_loader: fit(train_loader=loader)"
            )

        if self.optimizer is None:
            init_batch = next(iter(train_dl))
            init_batch = U.move_batch_to_device(init_batch, self.device)

            if hasattr(self.model, 'build') and hasattr(self.model, 'backbone'):
                if self.model.backbone is None:
                    X_init, y_init = init_batch
                    self.model.build(X_init, y_init)

            self.optimizer = U.build_optimizer(
                model=self.model,
                cfg=self.optim_cfg,
            )

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_metric: Optional[float] = None
        best_epoch: int = -1

        for epoch in range(cfg.max_epochs):
            train_loss = self._train_one_epoch(train_dl)
            self.history["train_loss"].append(float(train_loss))

            val_loss = None
            if val_dl is not None:
                val_loss = self.evaluate(val_dl)
                self.history["val_loss"].append(float(val_loss))
                monitor_value = self.evaluate_monitor(val_dl, val_loss)
                self.history["val_monitor"].append(float(monitor_value))
            else:
                self.history["val_loss"].append(float("nan"))
                monitor_value = float(train_loss)
                self.history["val_monitor"].append(float("nan"))

            stop_training = False
            if self.early_stop is not None and val_dl is not None:
                prev_best_epoch = self.early_stop.best_epoch
                stop_training = self.early_stop.step(monitor_value, epoch)
                improved = self.early_stop.best_epoch != prev_best_epoch
            elif val_dl is not None:
                improved = (best_metric is None) or (
                    monitor_value < best_metric if cfg.mode == "min" else monitor_value > best_metric
                )
            else:
                improved = False

            if improved:
                best_metric = monitor_value
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            if cfg.verbose and (epoch + 1) % cfg.log_every == 0:
                if val_loss is None:
                    print(f"[Epoch {epoch+1:04d}] train_loss={train_loss:.6f}")
                else:
                    print(f"[Epoch {epoch+1:04d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if self.early_stop is not None and val_dl is not None:
                if stop_training:
                    if cfg.verbose:
                        print(
                            f"Early stopping at epoch {epoch+1}, "
                            f"best_epoch={self.early_stop.best_epoch+1}, "
                            f"best_{self.monitor_name}={self.early_stop.best_value:.6f}"
                        )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        elif val_dl is None and len(self.history["train_loss"]) > 0:
            best_epoch = len(self.history["train_loss"]) - 1
            best_metric = float(self.history["train_loss"][-1])

        result: Dict[str, Any] = {
            "history": self.history,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "best_metric_name": self.monitor_name,
            "train_cfg": asdict(cfg),
            "optim_cfg": asdict(self.optim_cfg),
            "device": str(self.device),
            "amp_enabled": bool(self.amp.enabled),
            "amp_dtype": str(self.amp.dtype).replace("torch.", ""),
        }

        if return_state_dict:
            result["state_dict"] = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        
        if hasattr(self, '_export_if_enabled') and callable(getattr(self, '_export_if_enabled', None)):
            self._export_if_enabled(result=result)
        
        return result

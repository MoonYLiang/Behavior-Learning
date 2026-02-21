from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils as U
from .base import EarlyStop


@dataclass
class AmortizedConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0

    w_bl: float = 1.0 
    w_mse: float = 1.0  

    optim: Literal["adam", "adamw"] = "adam"
    shuffle: bool = True

    patience: int = 20
    min_delta: float = 0.0
    mode: str = "min"

    seed: int | None = None
    device: str | torch.device | None = None
    verbose: bool = False


def _loss_terms(
    *,
    bl_score: torch.Tensor,
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    w_bl: float,
    w_mse: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    mse = F.mse_loss(y_hat, y_true, reduction="mean")
    bl_mean = bl_score.mean()

    loss = float(w_bl) * (-bl_mean) + float(w_mse) * mse
    return loss, mse, bl_mean


def _run_epoch(
    *,
    predictor: nn.Module,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    bl_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    w_bl: float,
    w_mse: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    total_loss = 0.0
    total_mse = 0.0
    total_bl = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)

        y_hat = predictor(xb)
        bl_score = bl_model(xb, y_hat)

        loss, mse, bl_mean = _loss_terms(
            bl_score=bl_score,
            y_hat=y_hat,
            y_true=yb,
            w_bl=w_bl,
            w_mse=w_mse,
        )

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_mse += float(mse.detach().cpu())
        total_bl += float(bl_mean.detach().cpu())
        n_batches += 1

    return total_loss / max(n_batches, 1), total_mse / max(n_batches, 1), total_bl / max(n_batches, 1)


@torch.no_grad()
def _eval_epoch(
    *,
    predictor: nn.Module,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    bl_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    w_bl: float,
    w_mse: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    total_loss = 0.0
    total_mse = 0.0
    total_bl = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        y_hat = predictor(xb)
        bl_score = bl_model(xb, y_hat)

        loss, mse, bl_mean = _loss_terms(
            bl_score=bl_score,
            y_hat=y_hat,
            y_true=yb,
            w_bl=w_bl,
            w_mse=w_mse,
        )

        total_loss += float(loss.cpu())
        total_mse += float(mse.cpu())
        total_bl += float(bl_mean.cpu())
        n_batches += 1

    return total_loss / max(n_batches, 1), total_mse / max(n_batches, 1), total_bl / max(n_batches, 1)

def fit_amortized_predictor(
    predictor: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    bl_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: AmortizedConfig = AmortizedConfig(),
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
) -> Dict[str, list]:
    
    if cfg.seed is not None:
        U.set_seed(cfg.seed)

    dev = U.resolve_device(model=predictor, tensor=x_train, device=cfg.device)
    predictor.to(dev)
    U.freeze_module(bl_model)

    optim_cfg = U.OptimConfig(optim=cfg.optim, lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt = U.build_optimizer(predictor, optim_cfg)

    train_loader = U.make_data_loader(x_train, y_train, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    val_loader = None
    if x_val is not None and y_val is not None:
        val_loader = U.make_data_loader(x_val, y_val, batch_size=cfg.batch_size, shuffle=False)

    result: Dict[str, list] = {
        "train_loss": [],
        "train_mse": [],
        "train_bl": [],
    }
    if val_loader is not None:
        result.update({
            "val_loss": [],
            "val_mse": [],
            "val_bl": [],
        })

    best_state = None
    early_stop = EarlyStop(
        patience=cfg.patience,
        min_delta=cfg.min_delta,
        mode=cfg.mode,
    )

    for epoch in range(1, cfg.epochs + 1):
        predictor.train()

        train_loss, train_mse, train_bl = _run_epoch(
            predictor=predictor,
            loader=train_loader,
            bl_model=bl_model,
            w_bl=cfg.w_bl,
            w_mse=cfg.w_mse,
            optimizer=opt,
            device=dev,
        )

        result["train_loss"].append(train_loss)
        result["train_mse"].append(train_mse)
        result["train_bl"].append(train_bl)

        if val_loader is not None:
            predictor.eval()
            with torch.no_grad():
                val_loss, val_mse, val_bl = _eval_epoch(
                    predictor=predictor,
                    loader=val_loader,
                    bl_model=bl_model,
                    w_bl=cfg.w_bl,
                    w_mse=cfg.w_mse,
                    device=dev,
                )

            result["val_loss"].append(val_loss)
            result["val_mse"].append(val_mse)
            result["val_bl"].append(val_bl)

            if early_stop._is_improvement(val_loss):
                best_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}

            if cfg.verbose:
                print(
                    f"[amortized] epoch {epoch:03d} | "
                    f"train loss={train_loss:.6f} mse={train_mse:.6f} | "
                    f"val loss={val_loss:.6f} mse={val_mse:.6f}"
                )

            if early_stop.step(val_loss, epoch):
                if cfg.verbose:
                    print(f"[amortized] early stop at epoch {epoch}")
                break

        else:
            if cfg.verbose:
                print(
                    f"[amortized] epoch {epoch:03d} | "
                    f"train loss={train_loss:.6f} mse={train_mse:.6f}"
                )

    if best_state is not None:
        predictor.load_state_dict(best_state)

    return result

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List
from pathlib import Path

import torch 

import random
import numpy as np
import os
import torch.nn as nn

try:
    from torch.amp import autocast  
except Exception:
    from torch.cuda.amp import autocast 

try:
    from torch.amp import GradScaler  
except Exception:  
    from torch.cuda.amp import GradScaler 

from blnetwork.export import export_structure

Batch = Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]

def set_seed(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_device(
    model: Optional[nn.Module] = None,
    tensor: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.device:
    if device is not None:
        return torch.device(device)
    if model is not None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    if tensor is not None:
        return tensor.device
    return auto_device()


@dataclass
class OptimConfig:
    optim: str = "adam"         
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9  


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:

    optimizer_name = cfg.optim.lower()
    decay_params = []
    no_decay_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = n.endswith(".bias")
        is_gate = ("lam_u" in n) or ("lam_c" in n) or ("lam_t" in n) or (".lam_" in n)
        is_1d = p.ndim == 1
        if is_bias or is_gate or is_1d:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": cfg.weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    if optimizer_name == "adam":
        return torch.optim.Adam(param_groups, lr=cfg.lr)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=cfg.lr)
    if optimizer_name == "sgd":
        return torch.optim.SGD(param_groups, lr=cfg.lr, weight_decay=0.0, momentum=cfg.momentum)

    raise ValueError("Optimizer name must be one of: 'adam', 'adamw', 'sgd'.")

@dataclass
class AMPState:
    enabled: bool
    dtype: torch.dtype
    scaler: GradScaler


def make_amp(device: torch.device, mixed_precision: bool, prefer_bf16: bool = True) -> AMPState:
    use_cuda = (device.type == "cuda") and torch.cuda.is_available()
    enabled = bool(mixed_precision) and use_cuda

    if not enabled:
        return AMPState(enabled=False, dtype=torch.float32, scaler=GradScaler("cuda", enabled=False))

    if prefer_bf16 and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    scaler = GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    return AMPState(enabled=True, dtype=amp_dtype, scaler=scaler)


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    if isinstance(batch, (tuple, list)):
        return tuple(t.to(device) for t in batch) 
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    raise TypeError("Batch must be a tuple/list or a dict.")


@contextlib.contextmanager
def autocast_ctx(amp: AMPState):
    if amp.enabled:
        with autocast(device_type="cuda", dtype=amp.dtype, enabled=True):
            yield
    else:
        yield


def make_data_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def freeze_module(module: Union[nn.Module, object]) -> None:
    if isinstance(module, nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)

@dataclass
class ExportConfig:
    enabled: bool = False  
    model_path: Optional[str] = None
    state_dict_path: Optional[str] = None
    df: Optional[Any] = None
    feature_names: Optional[List[str]] = None
    ndigits: int = 4
    tol: float = 0.0

def export_artifacts(
    model: nn.Module,
    export_cfg: ExportConfig,
    result: Optional[Dict] = None,
) -> None:
    if not export_cfg.enabled:
        return

    if export_cfg.model_path is None:
        export_structure(
            model, 
            export_cfg.df, 
            export_cfg.feature_names, 
            ndigits=export_cfg.ndigits, 
            tol=export_cfg.tol
        )
    else:
        export_path = Path(export_cfg.model_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_structure(
            model, 
            export_cfg.df, 
            export_cfg.feature_names, 
            txt_path=str(export_path), 
            ndigits=export_cfg.ndigits, 
            tol=export_cfg.tol
        )
        
    if export_cfg.state_dict_path is not None:
        sd_path = Path(export_cfg.state_dict_path)
        sd_path.parent.mkdir(parents=True, exist_ok=True)

        if result is not None and "state_dict" in result:
            torch.save(result["state_dict"], sd_path)
        else:
            torch.save(model.state_dict(), sd_path)

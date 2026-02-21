from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch.nn as nn


def _as_logp_fn(model: Union[nn.Module, Callable]) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if callable(model) and not isinstance(model, nn.Module):
        return model

    if isinstance(model, nn.Module):
        return lambda x, y: model(x, y)

    raise TypeError("model must be a callable or an nn.Module.")


def _dsm_sample_loss(score_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if score_pred.dim() == 1:
        return 0.5 * (score_pred + target).pow(2).mean()
    return 0.5 * (score_pred + target).pow(2).sum(dim=1).mean()


class DSMLoss(nn.Module):
    def __init__(self, sigma: float = 1.0, n_noise: int = 1, temperature: float = 1.0) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")
        if n_noise <= 0:
            raise ValueError("n_noise must be >= 1.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.sigma = float(sigma)
        self.n_noise = int(n_noise)
        self.temperature = float(temperature)

    def forward(
        self,
        model: Union[nn.Module, Callable],
        xb: torch.Tensor,
        yb: torch.Tensor,
        sigma: Optional[float] = None,
        n_noise: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        logp_fn = _as_logp_fn(model)

        sig = float(self.sigma if sigma is None else sigma)
        k = int(self.n_noise if n_noise is None else n_noise)
        temp = float(self.temperature if temperature is None else temperature)

        device = xb.device
        total = torch.tensor(0.0, device=device)

        for _ in range(k):
            eps = torch.randn_like(yb) * sig
            y_tilde = (yb + eps).detach().requires_grad_(True)

            logp = logp_fn(xb, y_tilde)
            grad_logp = torch.autograd.grad(logp.sum(), y_tilde, create_graph=True)[0]

            score_pred = grad_logp / temp
            target = (y_tilde - yb) / (sig ** 2)

            total = total + _dsm_sample_loss(score_pred, target)

        return total / k
    

class CELoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, reduction: str = "mean", temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.temperature = float(temperature)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction)

    def forward(self, logits: torch.Tensor, y_disc: torch.Tensor) -> torch.Tensor:
        return self.ce(logits / self.temperature, y_disc)
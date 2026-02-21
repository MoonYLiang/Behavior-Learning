from __future__ import annotations

import torch
import torch.nn as nn

from ..training import utils as U  


@torch.no_grad()
def predict_continuous(
    predictor: nn.Module,
    x: torch.Tensor,
    *,
    device: str | torch.device | None = None,
    return_cpu: bool = True,
) -> torch.Tensor:

    predictor.eval()
    dev = U.resolve_device(model=predictor, tensor=x, device=device)
    y_hat = predictor(x.to(dev))
    return y_hat.detach().cpu() if return_cpu else y_hat.detach()
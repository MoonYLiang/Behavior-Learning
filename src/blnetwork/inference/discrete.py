import torch
import torch.nn.functional as F

from ..training import utils as U  


@torch.no_grad()
def predict_proba_discrete(
    model,
    x: torch.Tensor,
    *,
    temperature: float = 1.0,
    device: str | torch.device | None = None,
    return_cpu: bool = False,
) -> torch.Tensor:
    
    dev = U.resolve_device(model, x, device)
    x = x.to(dev)

    scores = model.logits(x) if hasattr(model, "logits") else model(x)
    logits = scores / float(temperature)
    probs = F.softmax(logits, dim=1)

    return probs.cpu() if return_cpu else probs


@torch.no_grad()
def predict_class_discrete(
    model,
    x: torch.Tensor,
    *,
    temperature: float = 1.0,
    device: str | torch.device | None = None,
    return_cpu: bool = True,
) -> torch.Tensor:
    probs = predict_proba_discrete(
        model, x, temperature=temperature, device=device, return_cpu=False
    )
    pred = probs.argmax(dim=1)
    return pred.cpu() if return_cpu else pred
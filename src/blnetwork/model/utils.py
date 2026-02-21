import torch
from typing import Callable, Tuple

__all__ = [
    "first_activation",
    "second_activation",
    "third_activation",
    "infer_num_classes",
    "onehot_candidates",
    "enumerate_onehot_logits",
]

def first_activation(z: torch.Tensor, first_act_func: str = "tanh") -> torch.Tensor:
    if first_act_func == "tanh":
        return torch.tanh(z)
    if first_act_func == "none":
        return z
    raise ValueError(f"Unknown first_act_func='{first_act_func}'. Use 'tanh' or 'none'.")


def second_activation(z: torch.Tensor, second_act_func: str, beta: float = 1.0) -> torch.Tensor:
    if second_act_func == "relu":
        return torch.relu(z)
    if second_act_func == "softplus":
        return torch.nn.functional.softplus(z, beta=beta)
    raise ValueError(f"Unknown second_act_func='{second_act_func}'. Use 'relu' or 'softplus'.")

def third_activation(z: torch.Tensor, third_act_func: str) -> torch.Tensor:
    if third_act_func == "abs":
        return torch.abs(z)
    if third_act_func == "square":
        return z ** 2
    raise ValueError(f"Unknown third_act_func='{third_act_func}'. Use 'abs' or 'square'.")

def infer_num_classes(y: torch.Tensor) -> Tuple[int, torch.Tensor]:

    if y.ndim == 2:
        K = int(y.shape[1])
        y_idx = torch.argmax(y, dim=1).long()
        return K, y_idx

    if y.ndim != 1:
        raise ValueError(f"y must be 1D class-index or 2D one-hot, got shape={tuple(y.shape)}")

    if y.dtype.is_floating_point:
        if not torch.allclose(y, y.round()):
            raise ValueError("Discrete y looks continuous (non-integer floats).")
        y = y.round()

    y = y.long()
    classes, y_idx = torch.unique(y, sorted=True, return_inverse=True)
    K = int(classes.numel())
    return K, y_idx


@torch.no_grad()
def onehot_candidates(m: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(int(m), device=device, dtype=dtype)


def enumerate_onehot_logits(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    m: int,
) -> torch.Tensor:

    B = x.shape[0]
    m = int(m)
    device, dtype = x.device, x.dtype

    X_rep = x.repeat_interleave(m, dim=0)      
    Y = onehot_candidates(m, device, dtype)    
    Y_rep = Y.repeat(B, 1)                     

    e = score_fn(X_rep, Y_rep)      
    e = e.view(B, m)
    return e

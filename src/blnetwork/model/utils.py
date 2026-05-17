from typing import Callable, Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = [
    "first_activation",
    "second_activation",
    "third_activation",
    "validate_x",
    "infer_continuous_y_dim",
    "validate_continuous_y",
    "validate_class_indices",
    "prepare_discrete_y",
    "prepare_continuous_y",
    "infer_num_classes",
    "enumerate_class_logits",
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


def validate_x(x: torch.Tensor, expected_x_dim: Optional[int] = None) -> int:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (B, x_dim), got shape {tuple(x.shape)}")

    x_dim = int(x.shape[1])
    if expected_x_dim is not None and x_dim != int(expected_x_dim):
        raise ValueError(
            f"x has wrong feature dimension: expected {expected_x_dim}, got {x_dim}. "
            f"Build a new model for inputs with a different x_dim."
        )
    return x_dim


def infer_continuous_y_dim(y: torch.Tensor) -> int:
    if y.ndim == 1:
        return 1
    if y.ndim == 2:
        return int(y.shape[1])
    raise ValueError(f"Continuous y must be 1D or 2D, got shape {tuple(y.shape)}")


def validate_continuous_y(y: torch.Tensor, expected_y_dim: Optional[int] = None) -> int:
    y_dim = infer_continuous_y_dim(y)
    if expected_y_dim is not None and y_dim != int(expected_y_dim):
        raise ValueError(
            f"y has wrong target dimension: expected {expected_y_dim}, got {y_dim}. "
            f"Build a new model for targets with a different y_dim."
        )
    return y_dim


def validate_class_indices(
    y: torch.Tensor,
    num_classes: Optional[int] = None,
    require_contiguous: bool = False,
) -> torch.Tensor:
    if y.ndim != 1:
        msg = f"For discrete task, y must be 1D class indices, got shape {tuple(y.shape)}."
        if y.ndim == 2:
            msg += " Convert one-hot labels with y.argmax(dim=1)."
        raise ValueError(msg)

    if y.dtype.is_floating_point:
        if not torch.allclose(y, y.round()):
            raise ValueError("Discrete labels must be integer class indices.")
        y = y.round()

    y_long = y.long()

    if num_classes is not None and y_long.numel() > 0:
        num_classes = int(num_classes)
        if (y_long.min() < 0) or (y_long.max() >= num_classes):
            raise ValueError(
                f"Discrete labels must be in range [0..{num_classes - 1}], "
                f"got values outside that range."
            )

    if require_contiguous:
        classes = torch.unique(y_long, sorted=True)
        expected = torch.arange(classes.numel(), device=classes.device, dtype=classes.dtype)
        if not torch.equal(classes, expected):
            raise ValueError(
                "Discrete labels must be contiguous class indices in [0, 1, 2, ...]. "
                "Please remap your labels first."
            )

    return y_long


def prepare_discrete_y(
    y: torch.Tensor,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y_long = validate_class_indices(y, num_classes=num_classes, require_contiguous=False)
    y_onehot = F.one_hot(y_long, num_classes=int(num_classes))
    return y_onehot.to(device=device, dtype=dtype)


def prepare_continuous_y(
    y: torch.Tensor,
    expected_y_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    validate_continuous_y(y, expected_y_dim=expected_y_dim)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    return y.to(device=device, dtype=dtype)


def infer_num_classes(y: torch.Tensor) -> Tuple[int, torch.Tensor]:
    y_long = validate_class_indices(y, require_contiguous=True)
    if y_long.numel() == 0:
        raise ValueError("Cannot infer num_classes from an empty y tensor.")

    num_classes = int(y_long.max().item()) + 1
    return num_classes, y_long


def enumerate_class_logits(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    m: int,
) -> torch.Tensor:
    m = int(m)
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")

    batch_size = x.shape[0]
    logits = []
    for class_idx in range(m):
        y_idx = torch.full((batch_size,), class_idx, device=x.device, dtype=torch.long)
        logits.append(score_fn(x, y_idx))
    return torch.cat(logits, dim=1)


def enumerate_onehot_logits(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    m: int,
) -> torch.Tensor:
    return enumerate_class_logits(score_fn, x, m=m)

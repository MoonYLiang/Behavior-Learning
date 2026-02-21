from typing import List, Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils as U


class BLUnit(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_basis: int,
        first_act_func: str = "tanh",
        second_act_func: str = "relu",
        third_act_func: str = "abs",
        eps: float = 1e-8,
        constrain_lambda: bool = True,
        init_lambda: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_basis = int(num_basis)
        self.first_act_func = str(first_act_func)
        self.second_act_func = str(second_act_func)
        self.third_act_func = str(third_act_func)
        self.eps = float(eps)
        self.constrain_lambda = bool(constrain_lambda)
        self.beta = float(beta)

        init_lambda = float(init_lambda)

        self.lin_u = nn.Linear(self.in_dim, self.num_basis, bias=True)
        self.lin_c = nn.Linear(self.in_dim, self.num_basis, bias=True)
        self.lin_t = nn.Linear(self.in_dim, self.num_basis, bias=True)

        self.lam = nn.Parameter(torch.full((3, self.num_basis), init_lambda))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        u = U.first_activation(self.lin_u(z), self.first_act_func)
        c = U.second_activation(self.lin_c(z), self.second_act_func, beta=self.beta)
        t = U.third_activation(self.lin_t(z), self.third_act_func)

        if self.constrain_lambda:
            lam = F.softplus(self.lam) + self.eps
        else:
            lam = self.lam

        lam_u = lam[0] 
        lam_c = lam[1]  
        lam_t = lam[2]  

        return lam_u * u - lam_c * c - lam_t * t


class BLBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_basis: int,
        first_act_func: str = "tanh",
        second_act_func: str = "relu",
        third_act_func: str = "abs",
        constrain_lambda: bool = True,  
        init_lambda: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.unit = BLUnit(
            in_dim=in_dim,
            num_basis=num_basis,
            first_act_func=first_act_func,
            second_act_func=second_act_func,
            third_act_func=third_act_func,
            constrain_lambda=constrain_lambda,  
            init_lambda=init_lambda,
            beta=beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unit(x)


class BLDeepBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        first_act_func: str = "tanh",
        second_act_func: str = "relu",
        third_act_func: str = "abs",
        constrain_lambda: bool = True,  
        init_lambda: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden_dims = list(hidden_dims)
        dims: List[int] = [int(in_dim)] + list(hidden_dims[:-1])

        self.blocks = nn.ModuleList(
            BLBlock(
                dims[i],
                num_basis=hidden_dims[i],
                first_act_func=first_act_func,
                second_act_func=second_act_func,
                third_act_func=third_act_func,
                constrain_lambda=constrain_lambda,  
                init_lambda=init_lambda,
                beta=beta,
            )
            for i in range(len(hidden_dims))
        )
        self.out_dim = int(hidden_dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class BLDeep(nn.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int],
        first_act_func: str = "tanh",
        second_act_func: str = "relu",
        third_act_func: str = "abs",
        head_bias: bool = True,
        num_classes: Optional[int] = None,
        task: str = "continuous",
        constrain_lambda: bool = True, 
        init_lambda: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dims = list(hidden_dims)
        self.first_act_func = str(first_act_func)
        self.second_act_func = str(second_act_func)
        self.third_act_func = str(third_act_func)
        self.head_bias = head_bias
        self.num_classes = num_classes

        if task not in {"continuous", "discrete"}:
            raise ValueError(f"task must be 'continuous' or 'discrete', got '{task}'")
        self.task = task
        self.constrain_lambda = bool(constrain_lambda)  
        self.init_lambda = float(init_lambda)
        self.beta = float(beta)

        self.x_dim: Optional[int] = None
        self.y_dim: Optional[int] = None
        self.backbone: Optional[nn.Module] = None
        self.head: Optional[nn.Module] = None

    def _build_architecture(self, x: torch.Tensor) -> None:
        self.backbone = BLDeepBackbone(
            in_dim=self.x_dim + self.y_dim,
            hidden_dims=self.hidden_dims,
            first_act_func=self.first_act_func,
            second_act_func=self.second_act_func,
            third_act_func=self.third_act_func,
            constrain_lambda=self.constrain_lambda,  
            init_lambda=self.init_lambda,
            beta=self.beta,
        )
        self.head = nn.Linear(self.backbone.out_dim, 1, bias=self.head_bias)

        device, dtype = x.device, x.dtype
        self.to(device=device, dtype=dtype)

    def build(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.x_dim = X.shape[1]

        if self.task == "discrete":
            m, y_idx = U.infer_num_classes(y)
            self.num_classes = int(m)
            self.y_dim = int(m)
            if not torch.equal(y_idx, y.long()):
                raise ValueError(
                    f"Discrete labels must be in range [0..K-1]. "
                    f"Got non-continuous labels. Please remap your labels to [0, 1, 2, ...] first."
                )
        else:
            self.y_dim = 1 if y.ndim == 1 else y.shape[1]

        self._build_architecture(X)
    
    def build_for_discrete_inference(self, x: torch.Tensor, num_classes: int) -> None:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (B, x_dim), got shape {tuple(x.shape)}")
        if self.task != "discrete":
            raise ValueError(f"This function only works with task='discrete', got task='{self.task}'")

        self.x_dim = int(x.shape[1])
        self.num_classes = int(num_classes)
        self.y_dim = int(self.num_classes)

        self._build_architecture(x)
        
    def score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            self.build(x, y)

        if y.ndim == 1:
            if self.task == "discrete":
                y = F.one_hot(y.long(), num_classes=self.num_classes).to(device=x.device, dtype=x.dtype)
            else:
                y = y.unsqueeze(1)
        elif y.ndim == 2:
            if self.task == "discrete" and y.shape[1] == 1:
                raise ValueError(
                    f"For discrete task, y should be 1D class indices (shape (B,)), "
                    f"but got shape {tuple(y.shape)}. Use y.squeeze(1) to convert (B,1) -> (B)."
                )

        z = torch.cat([x, y.to(device=x.device, dtype=x.dtype)], dim=1)
        feats = self.backbone(z)
        return self.head(feats)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.score(x, y)

    def logits(self, x: torch.Tensor, num_classes: int | None = None) -> torch.Tensor:
        if self.task != "discrete":
            raise RuntimeError("logits() is only available when task='discrete'.")

        m = int(num_classes or (self.num_classes or 0))
        if m <= 0:
            raise RuntimeError(
                "Unknown num_classes for discrete inference. "
                "Pass num_classes=K, or initialize BLDeep(..., num_classes=K), "
            )
        if self.backbone is None:
            self.build_for_discrete_inference(x, num_classes=m)

        return U.enumerate_onehot_logits(self.score, x, m=m)
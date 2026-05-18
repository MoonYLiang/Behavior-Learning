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
        num_u: int = 1,
        num_c: int = 1,
        num_t: int = 0,
        first_act_func: str = "none",
        second_act_func: str = "relu",
        third_act_func: str = "abs",
        eps: float = 1e-8,
        constrain_lambda: bool = True,
        init_lambda: float = 1.0,
        init_lambda_u: Optional[float] = None,
        init_lambda_c: Optional[float] = None,
        init_lambda_t: Optional[float] = None,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_basis = int(num_basis)
        self.num_u = int(num_u)
        self.num_c = int(num_c)
        self.num_t = int(num_t)
        self.first_act_func = str(first_act_func)
        self.second_act_func = str(second_act_func)
        self.third_act_func = str(third_act_func)
        self.eps = float(eps)
        self.constrain_lambda = bool(constrain_lambda)
        self.beta = float(beta)

        if self.num_u < 1:
            raise ValueError(f"num_u must be at least 1, got {self.num_u}")
        if self.num_c < 0:
            raise ValueError(f"num_c must be at least 0, got {self.num_c}")
        if self.num_t < 0:
            raise ValueError(f"num_t must be at least 0, got {self.num_t}")

        init_lambda = float(init_lambda)
        init_lambda_u = init_lambda if init_lambda_u is None else float(init_lambda_u)
        init_lambda_c = init_lambda if init_lambda_c is None else float(init_lambda_c)
        init_lambda_t = init_lambda if init_lambda_t is None else float(init_lambda_t)

        self.lin_u = nn.ModuleList(
            nn.Linear(self.in_dim, self.num_basis, bias=True) for _ in range(self.num_u)
        )
        self.lin_c = nn.ModuleList(
            nn.Linear(self.in_dim, self.num_basis, bias=True) for _ in range(self.num_c)
        )
        self.lin_t = nn.ModuleList(
            nn.Linear(self.in_dim, self.num_basis, bias=True) for _ in range(self.num_t)
        )

        self.lam_u = nn.Parameter(torch.full((self.num_u, self.num_basis), init_lambda_u))
        self.lam_c = nn.Parameter(torch.full((self.num_c, self.num_basis), init_lambda_c))
        self.lam_t = nn.Parameter(torch.full((self.num_t, self.num_basis), init_lambda_t))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.constrain_lambda:
            lam_u = F.softplus(self.lam_u) + self.eps
            lam_c = F.softplus(self.lam_c) + self.eps
            lam_t = F.softplus(self.lam_t) + self.eps
        else:
            lam_u = self.lam_u
            lam_c = self.lam_c
            lam_t = self.lam_t

        u = sum(
            lam_u[i] * U.first_activation(self.lin_u[i](z), self.first_act_func)
            for i in range(self.num_u)
        )
        c = sum(
            lam_c[i]
            * U.second_activation(self.lin_c[i](z), self.second_act_func, beta=self.beta)
            for i in range(self.num_c)
        )
        t = sum(
            lam_t[i] * U.third_activation(self.lin_t[i](z), self.third_act_func)
            for i in range(self.num_t)
        )

        return u - c - t


class BLBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_basis: int,
        num_u: int = 1,
        num_c: int = 1,
        num_t: int = 0,
        first_act_func: str = "none",
        second_act_func: str = "relu",
        third_act_func: str = "abs",
        constrain_lambda: bool = True,  
        init_lambda: float = 1.0,
        init_lambda_u: Optional[float] = None,
        init_lambda_c: Optional[float] = None,
        init_lambda_t: Optional[float] = None,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.unit = BLUnit(
            in_dim=in_dim,
            num_basis=num_basis,
            num_u=num_u,
            num_c=num_c,
            num_t=num_t,
            first_act_func=first_act_func,
            second_act_func=second_act_func,
            third_act_func=third_act_func,
            constrain_lambda=constrain_lambda,  
            init_lambda=init_lambda,
            init_lambda_u=init_lambda_u,
            init_lambda_c=init_lambda_c,
            init_lambda_t=init_lambda_t,
            beta=beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unit(x)


class BLDeepBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        num_u: int = 1,
        num_c: int = 1,
        num_t: int = 0,
        first_act_func: str = "none",
        second_act_func: str = "relu",
        third_act_func: str = "abs",
        constrain_lambda: bool = True,  
        init_lambda: float = 1.0,
        init_lambda_u: Optional[float] = None,
        init_lambda_c: Optional[float] = None,
        init_lambda_t: Optional[float] = None,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden_dims = [int(dim) for dim in hidden_dims]
        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one positive integer.")
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError(f"hidden_dims must contain only positive integers, got {self.hidden_dims}")
        dims: List[int] = [int(in_dim)] + self.hidden_dims[:-1]

        self.blocks = nn.ModuleList(
            BLBlock(
                dims[i],
                num_basis=self.hidden_dims[i],
                num_u=num_u,
                num_c=num_c,
                num_t=num_t,
                first_act_func=first_act_func,
                second_act_func=second_act_func,
                third_act_func=third_act_func,
                constrain_lambda=constrain_lambda,  
                init_lambda=init_lambda,
                init_lambda_u=init_lambda_u,
                init_lambda_c=init_lambda_c,
                init_lambda_t=init_lambda_t,
                beta=beta,
            )
            for i in range(len(self.hidden_dims))
        )
        self.out_dim = self.hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class BLDeep(nn.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int],
        num_u: int = 1,
        num_c: int = 1,
        num_t: int = 0,
        first_act_func: str = "none",
        second_act_func: str = "relu",
        third_act_func: str = "abs",
        head_bias: bool = True,
        num_classes: Optional[int] = None,
        task: str = "continuous",
        constrain_lambda: bool = True, 
        init_lambda: float = 1.0,
        init_lambda_u: Optional[float] = None,
        init_lambda_c: Optional[float] = None,
        init_lambda_t: Optional[float] = None,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dims = [int(dim) for dim in hidden_dims]
        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one positive integer.")
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError(f"hidden_dims must contain only positive integers, got {self.hidden_dims}")
        self.num_u = int(num_u)
        self.num_c = int(num_c)
        self.num_t = int(num_t)
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
        self.init_lambda_u = init_lambda_u
        self.init_lambda_c = init_lambda_c
        self.init_lambda_t = init_lambda_t
        self.beta = float(beta)

        self.x_dim: Optional[int] = None
        self.y_dim: Optional[int] = None
        self.backbone: Optional[nn.Module] = None
        self.head: Optional[nn.Module] = None

    def _build_architecture(self, x: torch.Tensor) -> None:
        self.backbone = BLDeepBackbone(
            in_dim=self.x_dim + self.y_dim,
            hidden_dims=self.hidden_dims,
            num_u=self.num_u,
            num_c=self.num_c,
            num_t=self.num_t,
            first_act_func=self.first_act_func,
            second_act_func=self.second_act_func,
            third_act_func=self.third_act_func,
            constrain_lambda=self.constrain_lambda,  
            init_lambda=self.init_lambda,
            init_lambda_u=self.init_lambda_u,
            init_lambda_c=self.init_lambda_c,
            init_lambda_t=self.init_lambda_t,
            beta=self.beta,
        )
        self.head = nn.Linear(self.backbone.out_dim, 1, bias=self.head_bias)

        device, dtype = x.device, x.dtype
        self.to(device=device, dtype=dtype)

    def build(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.x_dim = U.validate_x(X)

        if self.task == "discrete":
            m, _ = U.infer_num_classes(y)
            self.num_classes = int(m)
            self.y_dim = int(m)
        else:
            self.y_dim = U.infer_continuous_y_dim(y)

        self._build_architecture(X)
    
    def build_for_discrete_inference(self, x: torch.Tensor, num_classes: int) -> None:
        if self.task != "discrete":
            raise ValueError(f"This function only works with task='discrete', got task='{self.task}'")
        if int(num_classes) <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.x_dim = U.validate_x(x)
        self.num_classes = int(num_classes)
        self.y_dim = int(self.num_classes)

        self._build_architecture(x)
        
    def score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            self.build(x, y)
        U.validate_x(x, expected_x_dim=self.x_dim)

        if self.task == "discrete":
            y = U.format_discrete_y(y, self.num_classes, x.device, x.dtype)
        else:
            y = U.format_continuous_y(y, self.y_dim, x.device, x.dtype)

        z = torch.cat([x, y], dim=1)
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
        else:
            U.validate_x(x, expected_x_dim=self.x_dim)
            if m != self.num_classes:
                raise ValueError(
                    f"num_classes mismatch: model was built with num_classes={self.num_classes}, "
                    f"but got num_classes={m}. Build a new model for a different class count."
                )

        return U.enumerate_class_logits(self.score, x, m=m)

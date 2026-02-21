from __future__ import annotations

from .continuous import predict_continuous
from .discrete import predict_class_discrete, predict_proba_discrete

__all__ = [
	"predict_continuous",
	"predict_class_discrete",
	"predict_proba_discrete",
]

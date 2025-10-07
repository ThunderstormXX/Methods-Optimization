"""
Фреймворк для ограниченной оптимизации в PyTorch.

Этот пакет предоставляет:
- Множества ограничений (sets): L1-норма, L2-норма и т.д.
- Оптимизаторы (optim): Frank-Wolfe, Projected Gradient
- Модели (models): CNN и другие архитектуры

Пример использования:
    >>> from constraint_optimization.models import SmallCNN
    >>> from constraint_optimization.sets import L1ConstraintSet
    >>> from constraint_optimization.optim import FrankWolfe
    >>> 
    >>> model = SmallCNN()
    >>> tau = {name: 5.0 for name, _ in model.named_parameters()}
    >>> constraint = L1ConstraintSet(model, tau)
    >>> optimizer = FrankWolfe([{"params": model.parameters(), 
    ...                          "params_dict": dict(model.named_parameters())}], 
    ...                        constraint, lr=0.05)
"""

__version__ = "0.1.0"
__author__ = "Methods Optimization Team"

from . import sets
from . import optim
from . import models

__all__ = ['sets', 'optim', 'models']


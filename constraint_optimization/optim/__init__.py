"""
Модуль оптимизаторов для ограниченной оптимизации.
"""

from .frank_wolfe import FrankWolfe
from .projected_grad import ProjectedGradient

__all__ = ['FrankWolfe', 'ProjectedGradient']


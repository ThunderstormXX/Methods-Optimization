"""
Модуль constraint sets для ограниченной оптимизации.
"""

from .base import ConstraintSet
from .l1_constraint import L1ConstraintSet
from .l2_constraint import L2ConstraintSet
from .linf_constraint import LinfConstraintSet

__all__ = ['ConstraintSet', 'L1ConstraintSet', 'L2ConstraintSet', 'LinfConstraintSet']


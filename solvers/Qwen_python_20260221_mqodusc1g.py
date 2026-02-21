"""
Nonlinear Solver Implementations
================================
Production-ready solver implementations for nonlinear integral equations.
"""

from .picard import solve_picard
from .hybrid import solve_hybrid
from .nlcg import solve_nlcg
from .broyden import solve_broyden
from .akewe_hybrid import solve_akewe_hybrid

__all__ = [
    'solve_picard',
    'solve_hybrid',
    'solve_nlcg',
    'solve_broyden',
    'solve_akewe_hybrid'
]
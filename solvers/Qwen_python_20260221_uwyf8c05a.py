"""
Akewe Hybrid Iterative Scheme
==============================
Nigerian algorithmic contribution for singular kernel problems.
"""

import numpy as np
import time
import tracemalloc
from typing import Dict, Callable, Optional


def solve_akewe_hybrid(
    G: np.ndarray,
    h: float,
    u0: np.ndarray,
    phi_func: Callable,
    max_iter: int = 2000,
    tol: float = 1e-8,
    cond_G: Optional[float] = None,
    alpha: float = 0.5,
    beta: float = 0.5
) -> Dict:
    """
    Akewe Hybrid iterative scheme.
    
    Parameters
    ----------
    G : np.ndarray
        Green's matrix (n x n)
    h : float
        Grid spacing
    u0 : np.ndarray
        Initial guess (n,)
    phi_func : Callable
        Nonlinear function phi(u)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    cond_G : float, optional
        Condition number of G
    alpha : float
        First relaxation parameter
    beta : float
        Second relaxation parameter
    
    Returns
    -------
    Dict
        Solution dictionary with u, iterations, time, memory, converged
    """
    tracemalloc.start()
    start_time = time.perf_counter()
    
    u = u0.copy()
    errors = []
    converged = False
    
    if cond_G is not None and cond_G > 1e5:
        alpha, beta = 0.15, 0.15
    elif cond_G is not None and cond_G > 1e4:
        alpha, beta = 0.25, 0.25
    
    def T(x):
        return G @ phi_func(x) * h
    
    for k in range(max_iter):
        try:
            T_u = T(u)
            F_u = u - T_u
            err = np.linalg.norm(F_u, ord=np.inf)
        except Exception:
            err = np.inf
        
        errors.append(err)
        
        if err < tol:
            converged = True
            break
        if err > 1e12:
            break
        
        inner_z = (1 - beta) * u + beta * T_u
        z = T(inner_z)
        T_z = T(z)
        inner_y = (1 - alpha) * z + alpha * T_z
        y = T(inner_y)
        u = T(y)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'u': u,
        'errors': errors,
        'iterations': k + 1,
        'time': time.perf_counter() - start_time,
        'memory_peak': peak / 1024 / 1024,
        'converged': converged,
        'method': 'Akewe_Hybrid'
    }
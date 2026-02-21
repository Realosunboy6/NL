"""
Nonlinear Conjugate Gradient Solver
====================================
Memory-efficient solver for large-scale problems.
"""

import numpy as np
import time
import tracemalloc
from typing import Dict, Callable, Optional


def solve_nlcg(
    G: np.ndarray,
    h: float,
    u0: np.ndarray,
    phi_func: Callable,
    max_iter: int = 2000,
    tol: float = 1e-8,
    cond_G: Optional[float] = None
) -> Dict:
    """
    Nonlinear Conjugate Gradient solver.
    
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
    
    try:
        F_u = u - G @ phi_func(u) * h
        err = np.linalg.norm(F_u, ord=np.inf)
    except Exception:
        err = np.inf
    
    errors.append(err)
    d = -F_u.copy() if np.isfinite(err) else -u0.copy()
    
    for k in range(max_iter):
        if err < tol:
            converged = True
            break
        if err > 1e12:
            break
        
        alpha = 0.1
        u_new = u + alpha * d
        F_new = u_new - G @ phi_func(u_new) * h
        new_err = np.linalg.norm(F_new, ord=np.inf)
        
        if new_err < err:
            y = F_new - F_u
            beta = max(0, np.dot(F_new, y) / (np.dot(F_u, F_u) + 1e-15))
            d = -F_new + beta * d
            u, F_u, err = u_new, F_new, new_err
        else:
            d = -F_u
        
        errors.append(err)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'u': u,
        'errors': errors,
        'iterations': k + 1,
        'time': time.perf_counter() - start_time,
        'memory_peak': peak / 1024 / 1024,
        'converged': converged,
        'method': 'NLCG'
    }
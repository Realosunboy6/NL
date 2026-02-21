"""
Picard Iteration Solver
=======================
Universal baseline solver with condition-aware adaptive relaxation.
"""

import numpy as np
import time
import tracemalloc
from typing import Dict, Callable, Optional


def solve_picard(
    G: np.ndarray,
    h: float,
    u0: np.ndarray,
    phi_func: Callable,
    max_iter: int = 2000,
    tol: float = 1e-8,
    cond_G: Optional[float] = None
) -> Dict:
    """
    Picard iteration with adaptive relaxation.
    
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
    
    omega = 0.5 if cond_G is not None and cond_G > 1e4 else 1.0
    omega_min, omega_max = 0.05, 1.0
    error_history = []
    
    for k in range(max_iter):
        iter_start = time.perf_counter()
        
        try:
            u_new = G @ phi_func(u) * h
            err = np.linalg.norm(u_new - u, ord=np.inf)
        except Exception:
            break
        
        errors.append(err)
        
        if err < tol:
            converged = True
            break
        if err > 1e12:
            break
        
        if len(error_history) >= 2:
            if error_history[-1] > 2.0 * error_history[-2]:
                omega = max(omega_min, omega * 0.3)
            elif error_history[-1] < 0.3 * error_history[-2]:
                omega = min(omega_max, omega * 1.3)
        
        u = u_new
        error_history.append(err)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'u': u,
        'errors': errors,
        'iterations': k + 1,
        'time': time.perf_counter() - start_time,
        'memory_peak': peak / 1024 / 1024,
        'converged': converged,
        'method': 'Picard'
    }
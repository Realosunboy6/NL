"""
Broyden's Method Solver
=======================
Quasi-Newton method with O(N²) memory for ill-conditioned problems.
"""

import numpy as np
import time
import tracemalloc
from typing import Dict, Callable, Optional


def solve_broyden(
    G: np.ndarray,
    h: float,
    u0: np.ndarray,
    phi_func: Callable,
    max_iter: int = 2000,
    tol: float = 1e-8,
    cond_G: Optional[float] = None,
    timeout: float = 60.0
) -> Dict:
    """
    Broyden's quasi-Newton method with timeout protection.
    
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
    timeout : float
        Maximum time in seconds
    
    Returns
    -------
    Dict
        Solution dictionary with u, iterations, time, memory, converged, timeout
    """
    tracemalloc.start()
    start_time = time.perf_counter()
    
    n = len(u0)
    u = u0.copy()
    errors = []
    converged = False
    timed_out = False
    B = np.eye(n) * 0.5
    
    try:
        F_u = u - G @ phi_func(u) * h
        err = np.linalg.norm(F_u, ord=np.inf)
    except Exception:
        err = np.inf
    
    errors.append(err)
    
    for k in range(max_iter):
        elapsed = time.perf_counter() - start_time
        if elapsed > timeout:
            timed_out = True
            break
        
        if err < tol:
            converged = True
            break
        if err > 1e12:
            break
        
        try:
            delta = np.linalg.solve(B, -F_u)
        except Exception:
            delta = -0.1 * F_u
        
        u_new = u + 0.5 * delta
        F_new = u_new - G @ phi_func(u_new) * h
        new_err = np.linalg.norm(F_new, ord=np.inf)
        
        if new_err < err:
            s = u_new - u
            y = F_new - F_u
            if np.dot(s, s) > 1e-15:
                B = B + np.outer((y - B @ s), s) / np.dot(s, s)
            u, F_u, err = u_new, F_new, new_err
        
        errors.append(err)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'u': u,
        'errors': errors,
        'iterations': k + 1,
        'time': time.perf_counter() - start_time,
        'memory_peak': peak / 1024 / 1024,
        'converged': converged and not timed_out,
        'timeout': timed_out,
        'method': 'Broyden'
    }
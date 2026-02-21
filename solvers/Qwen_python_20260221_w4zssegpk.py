"""
Hybrid Picard-Newton Solver
============================
Combines Picard stability with Newton speed for robust convergence.
"""

import numpy as np
import time
import tracemalloc
from typing import Dict, Callable, Optional


def solve_hybrid(
    G: np.ndarray,
    h: float,
    u0: np.ndarray,
    phi_func: Callable,
    phi_derivative: Callable,
    max_iter: int = 2000,
    tol: float = 1e-8,
    cond_G: Optional[float] = None
) -> Dict:
    """
    Hybrid Picard-Newton solver.
    
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
    phi_derivative : Callable
        Derivative of phi(u)
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
    
    picard_max = 50 if cond_G is None or cond_G <= 1e4 else 100
    picard_tol = 1e-3
    
    for k in range(picard_max):
        u_new = G @ phi_func(u) * h
        err = np.linalg.norm(u_new - u, ord=np.inf)
        errors.append(err)
        u = u_new
        if err < picard_tol:
            break
    
    n = len(u)
    for j in range(max_iter - k):
        F_u = u - G @ phi_func(u) * h
        err = np.linalg.norm(F_u, ord=np.inf)
        errors.append(err)
        
        if err < tol:
            converged = True
            break
        if err > 1e12:
            break
        
        phi_prime = phi_derivative(u)
        J_diag = np.ones(n) - h * np.diag(G) * phi_prime
        J_diag = np.maximum(np.abs(J_diag), 1e-10) * np.sign(J_diag)
        delta = -F_u / J_diag
        u = u + 0.5 * delta
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'u': u,
        'errors': errors,
        'iterations': k + j + 1,
        'time': time.perf_counter() - start_time,
        'memory_peak': peak / 1024 / 1024,
        'converged': converged,
        'method': 'Hybrid'
    }
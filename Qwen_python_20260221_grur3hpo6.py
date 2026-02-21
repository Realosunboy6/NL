#!/usr/bin/env python3
"""
Nonlinear Solver Benchmark Runner
==================================
Main execution script for running benchmark suite.
"""

import numpy as np
import pandas as pd
import time
import tracemalloc
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solvers import (
    solve_picard,
    solve_hybrid,
    solve_nlcg,
    solve_broyden,
    solve_akewe_hybrid
)
from problems.benchmark_problems import get_benchmark_problems, ProblemConfig
from utils.visualization import generate_benchmark_plots
from config.solver_config import SOLVERS, DEFAULT_TOLERANCE, DEFAULT_SEED


def create_green_matrix(x: np.ndarray, kernel_type: str, alpha_sing: float):
    """Create Green's matrix for given kernel type."""
    n = len(x)
    X, S = np.meshgrid(x, x, indexing='ij')
    
    if kernel_type == 'standard':
        G = np.where(X <= S, X * (1 - S), S * (1 - X))
    elif kernel_type == 'modified':
        alpha = 1.0
        denom = np.sqrt(alpha)
        sinh_denom = np.sinh(denom)
        if abs(sinh_denom) < 1e-10:
            sinh_denom = 1e-10
        G = np.where(X <= S,
                     np.sinh(denom * X) * np.sinh(denom * (1 - S)) / (denom * sinh_denom),
                     np.sinh(denom * S) * np.sinh(denom * (1 - X)) / (denom * sinh_denom))
    elif kernel_type == 'stiff_kernel':
        alpha = 10.0
        denom = np.sqrt(alpha)
        sinh_denom = np.sinh(denom)
        if abs(sinh_denom) < 1e-10:
            sinh_denom = 1e-10
        G = np.where(X <= S,
                     np.sinh(denom * X) * np.sinh(denom * (1 - S)) / (denom * sinh_denom),
                     np.sinh(denom * S) * np.sinh(denom * (1 - X)) / (denom * sinh_denom))
    elif kernel_type == 'singular_weak':
        denom = np.sqrt(1.0 + alpha_sing * 10)
        sinh_denom = np.sinh(denom)
        if abs(sinh_denom) < 1e-10:
            sinh_denom = 1e-10
        term1 = np.sinh(denom * X) * np.sinh(denom * (1 - S))
        term2 = np.sinh(denom * S) * np.sinh(denom * (1 - X))
        G = np.where(X <= S, term1, term2) / (denom * sinh_denom)
        if alpha_sing > 0:
            dist = np.abs(X - S) + 1e-12
            G *= (1.0 + alpha_sing * np.exp(-dist * 50))
    else:
        G = np.where(X <= S, X * (1 - S), S * (1 - X))
    
    G_interior = G[1:-1, 1:-1].astype(np.float64)
    
    if n > 1000:
        step = max(1, n // 500)
        try:
            cond_G = np.linalg.cond(G_interior[::step, ::step])
        except Exception:
            cond_G = np.inf
    else:
        try:
            cond_G = np.linalg.cond(G_interior)
        except Exception:
            cond_G = np.inf
    
    return G_interior, cond_G


def phi_nonlinear(u: np.ndarray, config: ProblemConfig) -> np.ndarray:
    """Nonlinear source function."""
    lam, mu, nu = config.lambda_p, config.mu_p, config.nu_p
    u_clipped = np.clip(u, -25, 25)
    
    if config.nonlinearity_type == 'sin':
        return lam * np.sin(u_clipped)
    elif config.nonlinearity_type == 'cubic':
        return lam * u_clipped + mu * np.tanh(u_clipped**3)
    elif config.nonlinearity_type == 'hammerstein':
        return u_clipped**3 + u_clipped
    elif config.nonlinearity_type == 'chandrasekhar':
        return 0.5 * u_clipped / (1.0 + 0.5 * u_clipped)
    elif config.nonlinearity_type == 'love':
        return u_clipped
    elif config.nonlinearity_type == 'pro_extreme':
        return (10 * lam * np.sin(u_clipped) + 5 * mu * np.tanh(u_clipped**3) +
                0.5 * np.sign(u_clipped) * np.sqrt(np.abs(u_clipped)))
    elif config.nonlinearity_type == 'ultra_extreme':
        return (lam * u_clipped * np.sin(nu * u_clipped) +
                mu * np.tanh(u_clipped**5) +
                nu * np.exp(0.1 * u_clipped))
    else:
        return lam * np.sin(u_clipped) + mu * u_clipped**3


def phi_derivative(u: np.ndarray, config: ProblemConfig) -> np.ndarray:
    """Derivative of nonlinear function."""
    lam, mu, nu = config.lambda_p, config.mu_p, config.nu_p
    u_clipped = np.clip(u, -25, 25)
    
    if config.nonlinearity_type == 'sin':
        return lam * np.cos(u_clipped)
    elif config.nonlinearity_type == 'cubic':
        return lam + 3 * mu * u_clipped**2 * (1 - np.tanh(u_clipped**3)**2)
    elif config.nonlinearity_type == 'hammerstein':
        return 3 * u_clipped**2 + 1.0
    elif config.nonlinearity_type == 'chandrasekhar':
        return 0.5 / (1.0 + 0.5 * u_clipped)**2
    elif config.nonlinearity_type == 'love':
        return np.ones_like(u_clipped)
    elif config.nonlinearity_type == 'pro_extreme':
        return (10 * lam * np.cos(u_clipped) +
                15 * mu * u_clipped**2 * (1 - np.tanh(u_clipped**3)**2) +
                0.25 / np.sqrt(np.abs(u_clipped) + 1e-10))
    elif config.nonlinearity_type == 'ultra_extreme':
        return (lam * np.sin(nu * u_clipped) + lam * nu * u_clipped * np.cos(nu * u_clipped) +
                5 * mu * u_clipped**4 * (1 - np.tanh(u_clipped**5)**2) +
                0.1 * nu * np.exp(0.1 * u_clipped))
    else:
        return lam * np.cos(u_clipped) + 3 * mu * u_clipped**2


def setup_problem(config: ProblemConfig, seed: int = DEFAULT_SEED):
    """Setup problem from configuration."""
    np.random.seed(seed)
    x_full = np.linspace(0, 1, config.n_points)
    x = x_full[1:-1]
    h = x[1] - x[0]
    
    G, cond_G = create_green_matrix(x_full, config.kernel_type, config.singular_alpha)
    
    if config.difficulty == 'easy':
        u_true = 2 * np.sin(np.pi * x)
    elif config.difficulty == 'medium':
        u_true = 3 * np.sin(np.pi * x) * np.exp(-x)
    elif config.difficulty == 'hard':
        u_true = 5 * np.sin(2 * np.pi * x) * np.exp(-2 * x)
    else:
        u_true = (10 * np.sin(4 * np.pi * x) * np.exp(-4 * x) +
                  2 * np.sin(8 * np.pi * x))
    
    phi_true = phi_nonlinear(u_true, config)
    signal_norm = np.linalg.norm(phi_true, ord=np.inf)
    if signal_norm < 1e-10:
        signal_norm = 1.0
    noise = config.noise_level * signal_norm * np.random.normal(size=len(x))
    u_start = G @ (phi_true + noise) * h
    
    return {
        'x': x,
        'G': G,
        'h': h,
        'u_true': u_true,
        'u_start': u_start,
        'cond_G': cond_G
    }


def run_benchmark():
    """Execute benchmark suite."""
    problems = get_benchmark_problems()
    results_list = []
    
    print("\n" + "=" * 90)
    print(" NONLINEAR SOLVER BENCHMARK")
    print(" 10-Problem Suite: 4 Standard Literature + 6 Novel Extreme")
    print("=" * 90)
    
    tracemalloc.start()
    
    for config in problems:
        print(f"\n Configuration: {config.name} (N={config.n_points})")
        print(f"  λ={config.lambda_p}, μ={config.mu_p}, ν={config.nu_p}, α={config.singular_alpha}")
        print(f"  Kernel: {config.kernel_type} | Source: {config.problem_source}")
        print(f"  Time Limit: {config.time_limit_sec}s (Broyden: {config.broyden_time_limit}s)")
        
        data = setup_problem(config, DEFAULT_SEED)
        G, h, u_start, u_true, cond_G = (
            data['G'], data['h'], data['u_start'], data['u_true'], data['cond_G']
        )
        
        phi_func = lambda u: phi_nonlinear(u, config)
        phi_deriv_func = lambda u: phi_derivative(u, config)
        
        solvers = {
            'Picard': lambda: solve_picard(G, h, u_start, phi_func, config.max_iterations, DEFAULT_TOLERANCE, cond_G),
            'Hybrid': lambda: solve_hybrid(G, h, u_start, phi_func, phi_deriv_func, config.max_iterations, DEFAULT_TOLERANCE, cond_G),
            'NLCG': lambda: solve_nlcg(G, h, u_start, phi_func, config.max_iterations, DEFAULT_TOLERANCE, cond_G),
            'Akewe_Hybrid': lambda: solve_akewe_hybrid(G, h, u_start, phi_func, config.max_iterations, DEFAULT_TOLERANCE, cond_G),
        }
        
        if config.kernel_type == 'singular_weak' or config.n_points < 4000:
            solvers['Broyden'] = lambda: solve_broyden(G, h, u_start, phi_func, config.max_iterations, DEFAULT_TOLERANCE, cond_G, config.broyden_time_limit)
        
        for name, func in solvers.items():
            print(f"  Running {name}...", end='\r')
            try:
                res = func()
                rmse = np.sqrt(np.mean((res['u'] - u_true)**2)) if np.all(np.isfinite(res['u'])) else np.nan
                
                results_list.append({
                    'Problem': config.name,
                    'N_points': config.n_points,
                    'Method': name,
                    'Time_sec': round(res['time'], 3),
                    'Iterations': res['iterations'],
                    'Memory_MB': round(res['memory_peak'], 2),
                    'Converged': res['converged'],
                    'RMSE': rmse,
                    'Timeout': res.get('timeout', False),
                    'Problem_Source': config.problem_source,
                    'Reference': config.reference
                })
                
                status = "" if res['converged'] else ("⏱️" if res.get('timeout', False) else "⚠️")
                timeout_note = " (TIMEOUT)" if res.get('timeout', False) else ""
                print(f"  {status} {name:<15} | Time: {res['time']:<6.2f}s{timeout_note} | Iter: {res['iterations']:<5} | Mem: {res['memory_peak']:<6.1f}MB")
            except Exception as e:
                print(f"  ❌ {name} Crashed: {e}")
        
        print("-" * 90)
    
    tracemalloc.stop()
    
    df = pd.DataFrame(results_list)
    df.to_csv("benchmark_results.csv", index=False, encoding='utf-8')
    generate_benchmark_plots(df, 'benchmark_plots.png')
    
    print("\n" + "=" * 90)
    print(" RESULTS SUMMARY")
    print("=" * 90)
    summary = df.groupby(['Problem', 'Method']).agg(
        Avg_Time=('Time_sec', 'mean'),
        Avg_Iter=('Iterations', 'mean'),
        Success_Rate=('Converged', lambda x: x.mean() * 100),
        Peak_Mem_MB=('Memory_MB', 'max')
    ).round(2)
    print(summary.to_string())
    
    print("\n Files Saved:")
    print("  1. benchmark_results.csv")
    print("  2. benchmark_plots.png")
    print("=" * 90)
    
    return df


if __name__ == "__main__":
    run_benchmark()
"""
Benchmark Problem Configurations
================================
Define all benchmark problems here. Easy to add/modify without touching solver code.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ProblemConfig:
    """Problem configuration dataclass."""
    name: str
    n_points: int
    lambda_p: float
    mu_p: float
    nu_p: float
    noise_level: float
    nonlinearity_type: str
    difficulty: str
    kernel_type: str
    max_iterations: int
    singular_alpha: float
    time_limit_sec: float
    broyden_time_limit: float
    problem_source: str
    reference: str
    problem_type: str


def get_benchmark_problems() -> List[ProblemConfig]:
    """
    Return list of all benchmark problems.
    
    Modify this function to add/remove problems without changing solver code.
    
    Returns
    -------
    List[ProblemConfig]
        List of problem configurations
    """
    return [
        ProblemConfig(
            name='Hammerstein',
            n_points=1000,
            lambda_p=1.0,
            mu_p=0.5,
            nu_p=0.0,
            noise_level=0.1,
            nonlinearity_type='hammerstein',
            difficulty='medium',
            kernel_type='modified',
            max_iterations=800,
            singular_alpha=0.0,
            time_limit_sec=90.0,
            broyden_time_limit=60.0,
            problem_source='Literature',
            reference='Jerri (1999)',
            problem_type='standard'
        ),
        ProblemConfig(
            name='Chandrasekhar',
            n_points=1500,
            lambda_p=1.0,
            mu_p=0.0,
            nu_p=0.0,
            noise_level=0.1,
            nonlinearity_type='chandrasekhar',
            difficulty='hard',
            kernel_type='singular_weak',
            max_iterations=1000,
            singular_alpha=0.5,
            time_limit_sec=120.0,
            broyden_time_limit=60.0,
            problem_source='Literature',
            reference='Chandrasekhar (1960)',
            problem_type='standard'
        ),
        ProblemConfig(
            name='Fredholm',
            n_points=2000,
            lambda_p=2.0,
            mu_p=1.0,
            nu_p=0.0,
            noise_level=0.2,
            nonlinearity_type='cubic',
            difficulty='hard',
            kernel_type='stiff_kernel',
            max_iterations=1000,
            singular_alpha=0.3,
            time_limit_sec=120.0,
            broyden_time_limit=60.0,
            problem_source='Literature',
            reference='Atkinson (1997)',
            problem_type='standard'
        ),
        ProblemConfig(
            name='Love',
            n_points=1500,
            lambda_p=1.0,
            mu_p=0.0,
            nu_p=0.0,
            noise_level=0.1,
            nonlinearity_type='love',
            difficulty='medium',
            kernel_type='modified',
            max_iterations=1000,
            singular_alpha=0.0,
            time_limit_sec=120.0,
            broyden_time_limit=60.0,
            problem_source='Literature',
            reference='Love (1949)',
            problem_type='standard'
        ),
        ProblemConfig(
            name='Easy',
            n_points=500,
            lambda_p=1.0,
            mu_p=0.0,
            nu_p=0.0,
            noise_level=0.05,
            nonlinearity_type='sin',
            difficulty='easy',
            kernel_type='standard',
            max_iterations=500,
            singular_alpha=0.0,
            time_limit_sec=60.0,
            broyden_time_limit=60.0,
            problem_source='This Work',
            reference='',
            problem_type='novel'
        ),
        ProblemConfig(
            name='Medium',
            n_points=1000,
            lambda_p=2.0,
            mu_p=0.5,
            nu_p=0.0,
            noise_level=0.1,
            nonlinearity_type='cubic',
            difficulty='medium',
            kernel_type='modified',
            max_iterations=800,
            singular_alpha=0.0,
            time_limit_sec=90.0,
            broyden_time_limit=60.0,
            problem_source='This Work',
            reference='',
            problem_type='novel'
        ),
        ProblemConfig(
            name='Hard',
            n_points=2000,
            lambda_p=3.0,
            mu_p=1.0,
            nu_p=0.0,
            noise_level=0.2,
            nonlinearity_type='stiff',
            difficulty='hard',
            kernel_type='stiff_kernel',
            max_iterations=1000,
            singular_alpha=0.3,
            time_limit_sec=120.0,
            broyden_time_limit=60.0,
            problem_source='This Work',
            reference='',
            problem_type='novel'
        ),
        ProblemConfig(
            name='PRO EXTREME',
            n_points=3000,
            lambda_p=5.0,
            mu_p=2.0,
            nu_p=0.5,
            noise_level=0.3,
            nonlinearity_type='pro_extreme',
            difficulty='pro_extreme',
            kernel_type='stiff_kernel',
            max_iterations=2000,
            singular_alpha=0.0,
            time_limit_sec=120.0,
            broyden_time_limit=60.0,
            problem_source='This Work',
            reference='',
            problem_type='novel'
        ),
        ProblemConfig(
            name='ULTRA EXTREME',
            n_points=5000,
            lambda_p=10.0,
            mu_p=5.0,
            nu_p=2.0,
            noise_level=0.5,
            nonlinearity_type='ultra_extreme',
            difficulty='ultra_extreme',
            kernel_type='singular_weak',
            max_iterations=2000,
            singular_alpha=0.7,
            time_limit_sec=180.0,
            broyden_time_limit=60.0,
            problem_source='This Work',
            reference='',
            problem_type='novel'
        ),
        ProblemConfig(
            name='GOD TIER',
            n_points=6000,
            lambda_p=20.0,
            mu_p=10.0,
            nu_p=5.0,
            noise_level=0.7,
            nonlinearity_type='ultra_extreme',
            difficulty='god_tier',
            kernel_type='singular_weak',
            max_iterations=2000,
            singular_alpha=0.9,
            time_limit_sec=300.0,
            broyden_time_limit=90.0,
            problem_source='This Work',
            reference='',
            problem_type='novel'
        ),
    ]
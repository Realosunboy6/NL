"""
Visualization Utilities
=======================
Generate publication-quality plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List


def generate_benchmark_plots(df: pd.DataFrame, output_path: str = 'benchmark_plots.png'):
    """
    Generate benchmark visualization plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results DataFrame
    output_path : str
        Output file path
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(3, 2, figsize=(18, 16))
    
    methods = ['Picard', 'Hybrid', 'NLCG', 'Broyden', 'Akewe_Hybrid']
    colors = {
        'Picard': '#2E86AB',
        'Hybrid': '#06A77D',
        'NLCG': '#6A994E',
        'Broyden': '#F18F01',
        'Akewe_Hybrid': '#A23B72'
    }
    
    configs = df['Problem'].unique()
    x = np.arange(len(configs))
    width = 0.15
    
    ax = axs[0, 0]
    for i, m in enumerate(methods):
        subset = df[df['Method'] == m]
        times = [subset[subset['Problem'] == c]['Time_sec'].mean() for c in configs]
        ax.bar(x + i*width, times, width, label=m, color=colors.get(m, 'gray'), alpha=0.8)
    ax.set_xlabel('Problem', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Computation Time', fontsize=13, fontweight='bold')
    ax.set_xticks(x + 2*width)
    ax.set_xticklabels(configs, rotation=25, ha='right')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axs[0, 1]
    for i, m in enumerate(methods):
        subset = df[df['Method'] == m]
        mems = [subset[subset['Problem'] == c]['Memory_MB'].max() for c in configs]
        ax.bar(x + i*width, mems, width, label=m, color=colors.get(m, 'gray'), alpha=0.8)
    ax.set_xlabel('Problem', fontsize=11, fontweight='bold')
    ax.set_ylabel('Memory (MB)', fontsize=11, fontweight='bold')
    ax.set_title('Peak Memory Usage', fontsize=13, fontweight='bold')
    ax.set_xticks(x + 2*width)
    ax.set_xticklabels(configs, rotation=25, ha='right')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axs[1, 0]
    for i, m in enumerate(methods):
        subset = df[df['Method'] == m]
        iters = [subset[subset['Problem'] == c]['Iterations'].mean() for c in configs]
        ax.bar(x + i*width, iters, width, label=m, color=colors.get(m, 'gray'), alpha=0.8)
    ax.set_xlabel('Problem', fontsize=11, fontweight='bold')
    ax.set_ylabel('Iterations', fontsize=11, fontweight='bold')
    ax.set_title('Iterations to Converge', fontsize=13, fontweight='bold')
    ax.set_xticks(x + 2*width)
    ax.set_xticklabels(configs, rotation=25, ha='right')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axs[1, 1]
    for i, m in enumerate(methods):
        subset = df[df['Method'] == m]
        avg_iters = [subset[subset['Problem'] == c]['Avg_Iter_Time'].mean() for c in configs]
        ax.bar(x + i*width, avg_iters, width, label=m, color=colors.get(m, 'gray'), alpha=0.8)
    ax.set_xlabel('Problem', fontsize=11, fontweight='bold')
    ax.set_ylabel('Avg Iteration Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Per-Iteration Performance', fontsize=13, fontweight='bold')
    ax.set_xticks(x + 2*width)
    ax.set_xticklabels(configs, rotation=25, ha='right')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axs[2, 0]
    for i, m in enumerate(methods):
        subset = df[df['Method'] == m]
        conv_rates = [subset[subset['Problem'] == c]['Converged'].mean() * 100 for c in configs]
        ax.bar(x + i*width, conv_rates, width, label=m, color=colors.get(m, 'gray'), alpha=0.8)
    ax.set_xlabel('Problem', fontsize=11, fontweight='bold')
    ax.set_ylabel('Convergence Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Convergence Success Rate', fontsize=13, fontweight='bold')
    ax.set_xticks(x + 2*width)
    ax.set_xticklabels(configs, rotation=25, ha='right')
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axs[2, 1]
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
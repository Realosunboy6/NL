"""
Profiling Utilities
===================
Memory and timing profiling utilities.
"""

import time
import tracemalloc
from typing import Dict


class Profiler:
    """Simple profiler for solver performance tracking."""
    
    def __init__(self):
        self.start_time = None
        self.start_mem = 0
        self.peak_mem = 0
    
    def start(self):
        """Start profiling."""
        self.start_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        self.start_mem = current
    
    def stop(self):
        """Stop profiling."""
        current, peak = tracemalloc.get_traced_memory()
        self.peak_mem = peak
        return {
            'time': time.perf_counter() - self.start_time,
            'memory_peak': self.peak_mem / 1024 / 1024
        }
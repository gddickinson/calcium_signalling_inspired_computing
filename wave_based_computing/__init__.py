"""Wave-based computing inspired by calcium waves.

Submodules:
- wave_medium: WaveComputingMedium (core 2D wave simulation)
- wave_computing: WaveComputing (pattern recognition, logic gates, memory, filtering)
- wave_based_computing: Original combined module (preserved for backward compatibility)
"""
from .wave_medium import WaveComputingMedium
from .wave_computing import WaveComputing

__all__ = ['WaveComputingMedium', 'WaveComputing']

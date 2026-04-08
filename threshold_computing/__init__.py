"""Analog computing with calcium-like thresholds.

Contains:
- ThresholdComputingUnit: Single threshold unit
- ThresholdComputingNetwork: Network of threshold units
- ThresholdLogicComputer: Applications (temporal integration, detection, memory)
"""
from .threshold_computing import ThresholdComputingUnit, ThresholdComputingNetwork, ThresholdLogicComputer

__all__ = ['ThresholdComputingUnit', 'ThresholdComputingNetwork', 'ThresholdLogicComputer']

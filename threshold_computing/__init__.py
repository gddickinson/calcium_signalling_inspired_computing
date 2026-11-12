"""Analog computing with calcium-like thresholds.

Contains:
- ThresholdComputingUnit: Single threshold unit
- ThresholdComputingNetwork: Network of threshold units
- ThresholdLogicComputer (threshold_logic.py): Applications (temporal integration, detection, memory)
"""
from .threshold_computing import ThresholdComputingUnit, ThresholdComputingNetwork
from .threshold_logic import ThresholdLogicComputer

__all__ = ['ThresholdComputingUnit', 'ThresholdComputingNetwork', 'ThresholdLogicComputer']

"""Multi-store hybrid computing inspired by calcium stores.

Submodules:
- store: CalciumStore (individual memory compartment)
- computer: MultiStoreComputer (multi-store orchestration and visualization)
- hybrid: HybridComputing (hierarchical memory, cross-store, specialized tasks)
"""
from .store import CalciumStore
from .computer import MultiStoreComputer
from .hybrid import HybridComputing

__all__ = ['CalciumStore', 'MultiStoreComputer', 'HybridComputing']

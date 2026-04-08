"""Coordinated recruitment computing inspired by calcium channel dynamics.

Submodules:
- element: CoordinatedRecruitmentElement (single computing unit)
- network: CoordinatedRecruitmentNetwork (2D grid with coupling and clustering)
- quorum: QuorumComputer (fault tolerance, amplification, self-organization)
"""
from .element import CoordinatedRecruitmentElement
from .network import CoordinatedRecruitmentNetwork
from .quorum import QuorumComputer

__all__ = ['CoordinatedRecruitmentElement', 'CoordinatedRecruitmentNetwork', 'QuorumComputer']

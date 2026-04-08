# Calcium Signalling Inspired Computing -- Interface Map

## Project Structure

```
calcium_signalling_inspired_computing/
  __init__.py                      # Package root (version info)
  requirements.txt                 # Pinned dependencies
  ROADMAP.md                       # Development roadmap
  INTERFACE.md                     # This file
  tests/                           # Shared test suite
    test_probabilistic.py          # Tests for probabilistic_computing
    test_multistore.py             # Tests for multistore_computing
    test_coordinated.py            # Tests for coordinated_recruitment
    test_wave.py                   # Tests for wave_based_computing
    test_threshold.py              # Tests for threshold_computing
    test_spatiotemporal.py         # Tests for spatiotemporal_computing
  probabilistic_computing/
    __init__.py                    # Exports CalciumPuffComputing, StochasticComputing
    probabilistic_computing.py     # Grid-based probabilistic computing + stochastic arithmetic
  spatiotemporal_computing/
    __init__.py                    # Exports SpatiotemporalComputing, PatternRecognitionComputer
    spatiotemporal_computing.py    # Reaction-diffusion computing + pattern recognition
  threshold_computing/
    __init__.py                    # Exports ThresholdComputingUnit, ThresholdComputingNetwork, ThresholdLogicComputer
    threshold_computing.py         # Analog threshold computing (single-file, 510 lines)
  coordinated_recruitment/
    __init__.py                    # Exports CoordinatedRecruitmentElement, CoordinatedRecruitmentNetwork, QuorumComputer
    element.py                     # CoordinatedRecruitmentElement (single unit)
    network.py                     # CoordinatedRecruitmentNetwork (2D grid + DBSCAN clustering)
    quorum.py                      # QuorumComputer (fault tolerance, amplification, self-organization)
    coordinated_recruitment.py     # Original combined file (preserved, has __main__ demo)
  multistore_computing/
    __init__.py                    # Exports CalciumStore, MultiStoreComputer, HybridComputing
    store.py                       # CalciumStore (single memory compartment)
    computer.py                    # MultiStoreComputer (orchestration, cross-store comms, visualization)
    hybrid.py                      # HybridComputing (hierarchical memory, cross-store processing)
    multistore_computing.py        # Original combined file (preserved, has __main__ demo)
  wave_based_computing/
    __init__.py                    # Exports WaveComputingMedium, WaveComputing
    wave_medium.py                 # WaveComputingMedium (2D wave equation solver)
    wave_computing.py              # WaveComputing (pattern recognition, logic gates, memory, filtering)
    wave_based_computing.py        # Original combined file (preserved, has __main__ demo)
```

## Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| CalciumPuffComputing | probabilistic_computing | Grid-based probabilistic computing model |
| StochasticComputing | probabilistic_computing | Stochastic multiply/add/subtract/divide |
| SpatiotemporalComputing | spatiotemporal_computing | Multi-channel reaction-diffusion system |
| PatternRecognitionComputer | spatiotemporal_computing | Graph-based pattern recognition |
| ThresholdComputingUnit | threshold_computing | Single sigmoid threshold unit |
| ThresholdComputingNetwork | threshold_computing | 2D network of threshold units |
| ThresholdLogicComputer | threshold_computing | Temporal integration, detection, analog memory |
| CoordinatedRecruitmentElement | coordinated_recruitment.element | Single quorum-sensing unit |
| CoordinatedRecruitmentNetwork | coordinated_recruitment.network | 2D grid with coupling + DBSCAN clustering |
| QuorumComputer | coordinated_recruitment.quorum | Fault tolerance, amplification analysis |
| CalciumStore | multistore_computing.store | Read/write memory compartment |
| MultiStoreComputer | multistore_computing.computer | Multi-store orchestration + visualization |
| HybridComputing | multistore_computing.hybrid | Hierarchical memory, cross-store processing |
| WaveComputingMedium | wave_based_computing.wave_medium | 2D wave equation with sources, obstacles, resonance |
| WaveComputing | wave_based_computing.wave_computing | Pattern recognition, logic gates, memory, filtering |

## Module Dependencies

All six modules are independent -- they share no code between them.
Each module depends on: numpy, scipy, matplotlib.
Some additionally use: scikit-learn (PCA, DBSCAN, KMeans), networkx.

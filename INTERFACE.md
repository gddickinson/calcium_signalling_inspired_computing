# Calcium Signalling Inspired Computing -- Interface Map

## Project Structure

```
calcium_signalling_inspired_computing/
  __init__.py                      # Package root (version info)
  requirements.txt                 # Pinned dependencies
  ROADMAP.md                       # Development roadmap
  INTERFACE.md                     # This file
  SESSION_LOG.md                   # Work log (bug fixes, web interface, open decisions)
  run_web.py                       # Starts the web app on 127.0.0.1:8144
  requirements-web.txt             # Pinned web/test dependencies
  .github/workflows/pages.yml      # Builds and publishes the static site on push to main
  docs/img/                        # README screenshots of the web interface
  tests/                           # Shared test suite
    test_probabilistic.py          # Tests for probabilistic_computing
    test_multistore.py             # Tests for multistore_computing
    test_coordinated.py            # Tests for coordinated_recruitment
    test_wave.py                   # Tests for wave_based_computing
    test_threshold.py              # Tests for threshold_computing
    test_spatiotemporal.py         # Tests for spatiotemporal_computing
    test_regressions.py            # One regression test per fixed bug (see SESSION_LOG.md)
    test_wave_physics.py           # Wave equation: CFL substeps, energy conservation/decay, speed, stability
    test_wave_gates.py             # AND/OR/XOR truth tables of the wave gates
    test_pattern_recognition.py    # Stored patterns recognised, unstored ones rejected (3 seeds)
    test_web.py                    # Web API, static build, JS engine node tests, fixtures still current
    fixtures/*.json                # Seeded Python reference runs for the JS engines
  probabilistic_computing/
    __init__.py                    # Exports CalciumPuffComputing, StochasticComputing
    probabilistic_computing.py     # Grid-based probabilistic computing + stochastic arithmetic
  spatiotemporal_computing/
    __init__.py                    # Exports SpatiotemporalComputing, PatternRecognitionComputer
    spatiotemporal_computing.py    # SpatiotemporalComputing (reaction-diffusion-style model) + demo
    pattern_recognition.py         # PatternRecognitionComputer (early-dynamics signatures, accept radius)
  threshold_computing/
    __init__.py                    # Exports ThresholdComputingUnit, ThresholdComputingNetwork, ThresholdLogicComputer
    threshold_computing.py         # ThresholdComputingUnit, ThresholdComputingNetwork (+ reset), demo
    threshold_logic.py             # ThresholdLogicComputer (integration, detection, memory; trials start from rest)
  coordinated_recruitment/
    __init__.py                    # Exports CoordinatedRecruitmentElement, CoordinatedRecruitmentNetwork, QuorumComputer
    element.py                     # CoordinatedRecruitmentElement (single unit)
    network.py                     # CoordinatedRecruitmentNetwork (2D grid + DBSCAN clustering)
    quorum.py                      # QuorumComputer (fault tolerance, amplification, self-organization)
    coordinated_recruitment.py     # Demo script; re-exports the classes from the modules above
  multistore_computing/
    __init__.py                    # Exports CalciumStore, MultiStoreComputer, HybridComputing
    store.py                       # CalciumStore (single memory compartment)
    computer.py                    # MultiStoreComputer (orchestration, cross-store comms, visualization)
    hybrid.py                      # HybridComputing (hierarchical memory, cross-store processing)
    multistore_computing.py        # Demo script; re-exports the classes from the modules above
  wave_based_computing/
    __init__.py                    # Exports WaveComputingMedium, WaveComputing
    wave_medium.py                 # WaveComputingMedium (damped wave equation, CFL substeps, sources, sponge)
    wave_computing.py              # WaveComputing (logic gates from inputs, memory, filtering, PCA pattern matching)
    wave_based_computing.py        # Demo script; re-exports the classes from the modules above
```

## Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| CalciumPuffComputing | probabilistic_computing | Grid-based probabilistic computing model |
| StochasticComputing | probabilistic_computing | Stochastic multiply/add/subtract/divide |
| SpatiotemporalComputing | spatiotemporal_computing | Multi-channel reaction-diffusion system |
| PatternRecognitionComputer | spatiotemporal_computing.pattern_recognition | Recognise/reject patterns by their early dynamics |
| ThresholdComputingUnit | threshold_computing | Single sigmoid threshold unit |
| ThresholdComputingNetwork | threshold_computing | 2D network of threshold units |
| ThresholdLogicComputer | threshold_computing.threshold_logic | Temporal integration, detection, analog memory |
| CoordinatedRecruitmentElement | coordinated_recruitment.element | Single quorum-sensing unit |
| CoordinatedRecruitmentNetwork | coordinated_recruitment.network | 2D grid with coupling + DBSCAN clustering |
| QuorumComputer | coordinated_recruitment.quorum | Fault tolerance, amplification analysis |
| CalciumStore | multistore_computing.store | Read/write memory compartment |
| MultiStoreComputer | multistore_computing.computer | Multi-store orchestration + visualization |
| HybridComputing | multistore_computing.hybrid | Hierarchical memory, cross-store processing |
| WaveComputingMedium | wave_based_computing.wave_medium | Damped 2D wave equation with sources, obstacles, resonance |
| WaveComputing | wave_based_computing.wave_computing | Logic gates (AND/OR/XOR), memory, filtering, pattern matching |

## Module Dependencies

All six modules are independent -- they share no code between them.
Each module depends on: numpy, scipy, matplotlib.
Some additionally use: scikit-learn (PCA, DBSCAN, KMeans), networkx.

## Web interface (`web/`, `scripts/`)

```
web/
  app.py                 # FastAPI app: /api/info, /models, /overview, /checks, /recognition; STATIC_ENDPOINTS
  readme.py              # README sections, demo-script docstrings, class defaults (inspect) -> site text
  recognition.py         # Seeded PatternRecognitionComputer run (Python only, precomputed for the site)
  static/
    index.html, style.css (shared kit), style_app.css, core.js (kit router/helpers), main.js, static_api.js
    lab.js               # Shared model-page layout: controls with units, run/step/reset loop, README panel
    chart.js, heat.js, grid.js, bg.js   # Line/scatter charts, grid colour maps, convolution/statistics helpers
    np_random.js         # numpy legacy RandomState (MT19937, randn/normal/randint/choice) in JS
    puff_engine.js       # <- probabilistic_computing.py      (page: puffs.js)
    st_engine.js         # <- spatiotemporal_computing.py     (page: patterns.js)
    threshold_engine.js  # <- threshold_computing.py + threshold_logic.py (page: threshold.js)
    recruit_engine.js    # <- coordinated_recruitment/*.py incl. DBSCAN labelling (page: recruitment.js)
    store_engine.js      # <- multistore_computing/*.py       (page: stores.js)
    wave_engine.js       # <- wave_based_computing/*.py       (page: waves.js)
    *_engine.test.mjs, np_random.test.mjs, util.test.mjs   # node tests against tests/fixtures/*.json
    home.js, about.js    # Overview and About pages
scripts/
  build_static.py        # Export the static site (site/) via the FastAPI TestClient
  screenshots.py         # Playwright smoke test (+ --shots to docs/img)
  fixture_common.py      # Shared helpers for the fixture scripts (to_json, close, main)
  make_<name>_fixture.py # Seeded Python reference runs -> tests/fixtures/<name>.json
```

The web layer imports the model packages read-only; the models do not import the web layer.

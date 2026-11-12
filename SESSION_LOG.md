# Session log

## 2026-11-11 (later) — wave physics, wave gates, pattern recognition, README (branch `web-interface`)

George asked to fix the open problems. Each in its own "Fix ..." commit with tests:
- **Wave medium is now a real damped wave equation** (`wave_medium.py`): leapfrog with a 9-point Laplacian scaled to
  approximate ∇², internal substeps chosen from the CFL condition (c·dt/dx within 90% of the stencil limit),
  `wave_speed` parameter (cells per time unit, default 10; `diffusion` kept as an alias for c²), damping as velocity
  kept per step, hard sources and memory elements, 5-cell sponge for absorbing edges, real mirror/wrap edges,
  resonant regions lossless when tuned (was an unbounded x1.1 gain). `wave_energy()` is the conserved discrete energy.
  Tests: `tests/test_wave_physics.py` (energy conserved to 1e-9 without damping, monotone decay with it, pulse
  arrives at ~d/c, stable at the defaults with sources over 400 steps). Frequency/phase estimate vectorised (demo 60 s
  -> 5 s).
- **Wave logic gates use their inputs** (`wave_computing.py`): bit = pattern mean > 0.5 switches a line of sources;
  output = centre amplitude vs the one-input amplitude (AND > 1.5x, OR > 0.5x, XOR > 0.5x with B in antiphase).
  100 x 100: 0 / 0.365 / 0.365 / 0.734, XOR(1,1) = 0. `tests/test_wave_gates.py` checks all truth tables.
- **Pattern recognition** (`pattern_recognition.py`, moved out of spatiotemporal_computing.py): signature = first 5
  frames of the run, 5 x 5 block means per channel, channel mean removed; accept radius = 2x the largest distance of
  5 noisy training copies. Recognises all 3 stored patterns from noisy copies and rejects a corner pulse and a
  diagonal stripe, for 3 training seeds (`tests/test_pattern_recognition.py`). Site: 5 of 5 correct.
- **README puff claim** corrected to the measured response (slope 0.02-0.03 at decay 0.95; ~0.7 at decay 0.5);
  model unchanged.
- **numpy pin** relaxed to <3.0.
- Web: wave engine re-ported (fixture regenerated, gates included in the JS-vs-Python check), waves page has gate
  inputs A/B, wave-speed slider, CFL readout; recognition table updated.

Then (same day) the last two:
- **Multi-store sequence writes** (`computer.py`): sequences went to address 0 every time and overwrote each other;
  now the first run of quiet cells long enough (else the least active window); `write_log` records each write.
  Regression test stores three sequences and reads each back; store_engine.js and its fixture updated.
- **Wave pattern matching** (`wave_computing.py`): the encoding (pixels strictly above the 80th percentile) selected
  no pixels for the demo's clean references, so all reference media were empty; and the PCA-statistics signature
  scored ~0.46 for every pair. Now: sources at pixels above the middle of the value range; signature = RMS amplitude
  map (5 x 5 blocks, last 20 frames, mean removed, unit length); match = best correlation >= 0.9, else no match.
  Noisy references score >= 0.98, a cross and a corner block <= 0.68 (`tests/test_wave_recognition.py`, 50 and 100
  steps). Not ported to the web engine (Python only, like before).

## 2026-11-11 — web interface, static site, bug fixes (branch `web-interface`, Claude Code)

### Bug fixes (each in its own commit, regression tests in `tests/test_regressions.py`)
- **Stochastic subtraction/division** (`probabilistic_computing.py`): `a AND NOT b` and `P(a|b)` were computed on
  independent bitstreams, giving a·(1−b) and a instead of max(0, a−b) and min(1, a/b). Now one shared random stream
  (correlated bitstreams).
- **Wave medium on non-square grids** (`wave_medium.py`): arrays were shape (width, height) but indexed [y, x];
  IndexError whenever width != height. Arrays are now (height, width).
- **Threshold unit with refractory_period=0** (`threshold_computing.py`): ZeroDivisionError on the first update.
- **Threshold detection / analog memory trials** (`threshold_logic.py`, new): each trial inherited the state left by
  the previous trial; units and network now have `reset()` and every trial starts from rest. ThresholdLogicComputer
  moved to `threshold_logic.py` (re-exported) to keep files under 500 lines; the two affected PNGs regenerated.
- **Pattern recognition history** (`spatiotemporal_computing.py`): train()/recognize() appended to one shared history,
  so later patterns' graphs included earlier runs. History is now cleared per run.
- **Multi-store numpy scalars** (`store.py`, `computer.py`): np.int64/float32 raised TypeError on write and were
  silently replaced by [0.5, 0.4, 0.3] in query.
- Docs: README run commands, import paths (`calcium_computing.*` does not exist), clone/citation URLs, missing logo;
  wave docstrings now describe the update and logic gates the code actually has.
- Refactor: the three combined demo scripts (wave/multistore/coordinated) held full duplicate copies of the classes;
  they now import them from the split modules (checked method by method to be identical first).

### Open decisions for George (the first four were resolved later the same day, see above)
- **Wave medium physics**: the update is written as a leapfrog wave equation but sets `previous = current` before use
  and uses the neighbour mean as the Laplacian, so it is a first-order damped smoothing of a driven field. A direct fix
  (true leapfrog + mean − centre) was tried and blows up at the defaults (sources add amplitude per step to the
  displacement; explicit −0.1·u³ term), and with D·dt² = 0.002 waves would travel ~0.025 cells/step anyway. Needs a
  choice of source scaling and D/dt. Left as is; docs corrected; the web page explains it.
- **wave_logic_gates ignores its input patterns**; at the defaults the centre readout is ~1e-9 or smaller for AND/OR/XOR.
- **Pattern recognition method**: k-means labels are arbitrary per run, so comparing transition-graph edges across runs
  is weak; the seeded demo matches 1 of 3 noisy patterns (shown honestly on the site).
- **Puff model**: with the defaults the measured firing probability is a few percent and only weakly density-dependent
  (slope ~0.03), because the refractory threshold blocks most cells; the README's "linear relationship" holds in
  correlation but not as probability ≈ density. Text on the site says so.
- `requirements.txt` pins numpy<2.0, but the code and tests run on numpy 2.5.4 (requirements-web.txt pins that).
- `MultiStoreComputer.write_data` writes every sequence from address 0, so two sequence writes to one store overwrite
  each other (not changed).

### Web interface
- FastAPI app (`web/app.py`, port 8144, `run_web.py`) + static GitHub Pages site (`scripts/build_static.py`,
  `.github/workflows/pages.yml`). One page per model, each running a JS port node-tested against seeded Python
  (`tests/fixtures/*.json`, `scripts/make_*_fixture.py`), plus Overview and About.
- No Mac mini use.

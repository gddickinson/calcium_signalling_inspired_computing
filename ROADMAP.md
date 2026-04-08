# Calcium Signalling Inspired Computing -- Roadmap

## Current State
A collection of six independent computational models inspired by calcium signaling dynamics, each in its own subdirectory: `probabilistic_computing/`, `spatiotemporal_computing/`, `threshold_computing/`, `coordinated_recruitment/`, `multistore_computing/`, and `wave_based_computing/`. Each contains a single self-contained Python script. No shared utilities, no package structure, no tests, no `requirements.txt`. The models demonstrate novel computing paradigms but exist as standalone demonstrations.

## Short-term Improvements
- [x] Add `requirements.txt` at the project root (numpy, scipy, matplotlib, scikit-learn, networkx)
- [x] Add a top-level `__init__.py` and per-module `__init__.py` to make this importable as a package
- [ ] Extract common utilities (grid creation, visualization helpers, simulation loops) into a shared `utils/` module
- [x] Add unit tests for each model's core computation (e.g., stochastic multiply accuracy, wave interference patterns)
- [ ] Add docstrings and type hints to all six scripts
- [ ] Add a `run_all.py` script that demonstrates each model sequentially with consistent output
- [ ] Standardize the interface across models -- each should have `create()`, `simulate()`, `visualize()` methods

## Feature Enhancements
- [ ] Add interactive parameter exploration via Jupyter notebooks for each model
- [ ] Implement performance benchmarks comparing calcium-inspired computing vs. traditional approaches
- [ ] Add a combined model that integrates multiple paradigms (e.g., wave-based + threshold computing)
- [ ] Create animated visualizations (matplotlib animation or Manim) for each computing model
- [ ] Add noise robustness analysis -- how do models perform with noisy inputs?
- [ ] Implement a hardware simulation mode mapping models to neuromorphic chip constraints
- [ ] Add classification benchmarks (MNIST, simple datasets) for the pattern recognition models

## Long-term Vision
- [ ] Publish as a pip-installable package (`calcium-computing`) with proper API
- [ ] Write companion academic paper describing the computational frameworks
- [ ] Create a web-based demo (Streamlit/Gradio) for interactive exploration
- [ ] Implement real-time FPGA synthesis targets for the threshold and wave computing models
- [ ] Add integration with Brian2 or NEST for neuroscience simulation frameworks
- [ ] Develop a unified "calcium computing VM" that composes all six paradigms

## Technical Debt
- [x] Six completely independent scripts with no code sharing -- significant duplication of grid setup, plotting, and simulation loops (files split into modular submodules)
- [ ] No top-level entry point or CLI interface
- [x] Each script likely runs its full demo on import -- separate demo code from library code using `if __name__ == "__main__"` (already present in all scripts)
- [x] No `.gitignore` for generated plots and `__pycache__/`
- [ ] Missing `LICENSE` file referenced in README
- [ ] The README references import paths like `calcium_computing.probabilistic` that do not exist in the actual file structure

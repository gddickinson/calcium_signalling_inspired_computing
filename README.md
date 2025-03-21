# CalciumComputing

## Novel Computational Architectures Inspired by Calcium Signaling

This repository contains a collection of computational models inspired by calcium signaling dynamics in biological systems. These models demonstrate how principles from cellular calcium dynamics can inform novel computing paradigms with unique capabilities beyond traditional computing approaches.

![CalciumComputing Logo](assets/calcium-computing-logo.png)

## Overview

Calcium signaling represents one of the most versatile and widespread signaling systems in biology. The spatial and temporal organization of calcium dynamics enables cells to encode complex information processing tasks. This project explores how these principles can be adapted into computational frameworks with applications in machine learning, signal processing, fault-tolerant computing, and more.

## Models

### 1. Probabilistic Computing Based on Calcium Puff Dynamics

This model demonstrates how the probability of signal activation can scale linearly with the number of processing elements, similar to how calcium puff probability scales with IP3R receptor cluster size.

**Key Features:**
- Stochastic computing with predictable probabilities
- Linear relationship between element density and activation probability
- Implementation of probabilistic logic operations

```bash
python probabilistic_computing.py
```

### 2. Spatiotemporal Computing Inspired by Calcium Wave Propagation

This model implements a reaction-diffusion system where information is encoded in wave patterns and processing occurs through their spatiotemporal evolution.

**Key Features:**
- Multiple channels with different diffusion constraints
- Pattern recognition through spatiotemporal dynamics
- Information encoding in spatial configurations

```bash
python spatiotemporal_computing.py
```

### 3. Analog Computing with Calcium-Like Thresholds

This model creates threshold logic units with calcium-like activation properties, demonstrating how computation can emerge from threshold-based dynamics.

**Key Features:**
- Temporal integration of weak signals
- Threshold-based decision making
- Analog memory capability

```bash
python threshold_computing.py
```

### 4. Coordinated Recruitment Computing

This model demonstrates quorum sensing-like computation where elements remain dormant until enough neighboring elements are activated, similar to coordinated calcium channel opening.

**Key Features:**
- Self-organizing functional groups
- Fault tolerance through redundancy
- Signal amplification through recruitment

```bash
python coordinated_recruitment.py
```

### 5. Multi-Store Hybrid Computing Architecture

This model implements a hierarchical memory system with different store types inspired by distinct calcium stores (ER, lysosomes, etc.) in cells.

**Key Features:**
- Specialized stores for different types of computation
- Cross-store communication
- Hybrid analog-digital processing

```bash
python multistore_computing.py
```

### 6. Wave-Based Processing

This model demonstrates how information can be encoded in wave properties (amplitude, frequency, phase) and computation can emerge through wave interactions.

**Key Features:**
- Pattern recognition through wave dynamics
- Logic operations via wave interference
- Signal filtering through resonant wave interactions
- Information storage in persistent wave patterns

```bash
python wave_computing.py
```

## Installation

```bash
# Clone this repository
git clone https://github.com/username/CalciumComputing.git

# Navigate to the repository directory
cd CalciumComputing

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- networkx

## Usage Examples

### Probabilistic Computing

```python
from calcium_computing.probabilistic import CalciumPuffComputing, StochasticComputing

# Initialize the model
model = CalciumPuffComputing(grid_size=50)

# Run simulation
results = model.run_simulation(steps=500)

# Visualize results
model.visualize(results)

# Perform stochastic multiplication
stoch_comp = StochasticComputing(model)
result = stoch_comp.stochastic_multiply(0.3, 0.7)
print(f"0.3 × 0.7 ≈ {result:.4f}")
```

### Wave-Based Computing

```python
from calcium_computing.wave import WaveComputingMedium, WaveComputing

# Initialize medium
medium = WaveComputingMedium(size=(100, 100))

# Add wave sources
medium.add_source((20, 50), amplitude=0.5, frequency=0.8)
medium.add_source((80, 50), amplitude=0.5, frequency=0.8, phase=np.pi)

# Run simulation
frames = medium.run_simulation(steps=100, visualize=True)

# Initialize wave computer
computer = WaveComputing(medium)

# Filter a signal
signal = np.sin(np.linspace(0, 2*np.pi, 50)) + 0.3*np.sin(5*np.linspace(0, 2*np.pi, 50))
result = computer.wave_filter(signal, filter_type='lowpass', cutoff_frequency=0.3)
```

## Applications

These computational models have potential applications in:

- **Fault-Tolerant Computing**: Systems resilient to component failures
- **Energy-Efficient Computing**: Lower power requirements through analog and stochastic methods
- **Pattern Recognition**: Novel approaches to classification problems
- **Signal Processing**: Filtering and transformation through wave dynamics
- **Hybrid Computing**: Architectures combining digital and analog processing
- **Self-Organizing Systems**: Adaptable computing without explicit programming

## Research Background

This project is inspired by research on calcium signaling dynamics, particularly:

- IP3-mediated calcium puffs and waves
- NAADP-sensitive calcium stores
- Coordinated opening of calcium channels
- Cross-talk between different calcium stores
- Spatial and temporal calcium encoding

## Citation

If you use these models in your research, please cite:

```
@misc{CalciumComputing2023,
  author = {Dickinson, George},
  title = {CalciumComputing: Novel Computational Architectures Inspired by Calcium Signaling},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/username/CalciumComputing}}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work draws inspiration from research on calcium signaling in various cell types
- Special thanks to contributors in the fields of computational neuroscience and biologically-inspired computing

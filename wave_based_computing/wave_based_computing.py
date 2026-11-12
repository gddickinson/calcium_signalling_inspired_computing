"""
Wave-Based Processing Inspired by Calcium Waves

This simulation models a computational architecture inspired by calcium waves
where information is encoded in wave properties (amplitude, frequency, phase)
and computation emerges from wave interactions.

The model demonstrates:
1. Information encoding in wave properties
2. Computation through wave interference and interaction
3. Pattern recognition and signal filtering using wave dynamics
4. Memory implementation as persistent wave patterns

The classes live in the package modules (wave_medium.py, wave_computing.py); this file
re-exports them and keeps the original end-to-end demo: python wave_based_computing/wave_based_computing.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if __package__:
    from .wave_medium import WaveComputingMedium
    from .wave_computing import WaveComputing
else:  # run as a script: import the package from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from wave_based_computing.wave_medium import WaveComputingMedium
    from wave_based_computing.wave_computing import WaveComputing

__all__ = ['WaveComputingMedium', 'WaveComputing']

if __name__ == "__main__":
    # Initialize the wave computing medium
    np.random.seed(42)  # For reproducibility
    medium = WaveComputingMedium(size=(100, 100), boundary='absorbing')

    print("Setting up wave computing medium...")

    # Add some sources
    medium.add_source((20, 50), amplitude=0.5, frequency=0.8)
    medium.add_source((80, 50), amplitude=0.5, frequency=0.8, phase=np.pi)  # Out of phase

    # Add an obstacle
    medium.add_obstacle(45, 55, 40, 60)

    # Add a resonant region
    medium.add_resonant_region(70, 90, 70, 90, frequency=0.8)

    # Run the simulation
    print("Running wave simulation...")
    frames = medium.run_simulation(steps=100, visualize=True)

    # Visualize the final state
    fig = medium.visualize_state(mode='amplitude')
    plt.savefig('wave_computing_state.png')
    plt.close()

    # Create animation
    if frames:
        ani = medium.create_animation(frames=frames)
        # Uncomment to save animation:
        # ani.save('wave_computing_dynamics.mp4', writer='ffmpeg')

    # Initialize a wave computer
    wave_computer = WaveComputing(medium)

    # Demo: Pattern Recognition
    print("\nDemonstrating pattern recognition...")

    # Create test patterns
    pattern_size = (20, 20)

    # Pattern 1: Circle
    circle = np.zeros(pattern_size)
    center = (pattern_size[0] // 2, pattern_size[1] // 2)
    radius = min(pattern_size) // 3

    for i in range(pattern_size[0]):
        for j in range(pattern_size[1]):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            if dist < radius:
                circle[i, j] = 1.0

    # Pattern 2: Square
    square = np.zeros(pattern_size)
    margin = pattern_size[0] // 4
    square[margin:-margin, margin:-margin] = 1.0

    # Pattern 3: Diagonal line
    diagonal = np.zeros(pattern_size)
    for i in range(pattern_size[0]):
        j = int(i * pattern_size[1] / pattern_size[0])
        if 0 <= j < pattern_size[1]:
            diagonal[i, j] = 1.0
            # Make line thicker
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < pattern_size[0] and 0 <= nj < pattern_size[1]:
                        diagonal[ni, nj] = 0.8

    # Add noise to create test patterns
    test_circle = circle + np.random.normal(0, 0.1, pattern_size)
    test_square = square + np.random.normal(0, 0.1, pattern_size)
    test_diagonal = diagonal + np.random.normal(0, 0.1, pattern_size)

    # Reference patterns
    reference_patterns = [circle, square, diagonal]

    # Test patterns
    test_patterns = [test_circle, test_square, test_diagonal]

    # Perform pattern recognition
    recognition_results = wave_computer.pattern_recognition(
        test_patterns=test_patterns,
        reference_patterns=reference_patterns,
        simulation_steps=50
    )

    # Display results
    for i, result in enumerate(recognition_results):
        best_match = result['matches'][0]
        print(f"Test pattern {i} best matches reference {best_match['reference_index']} with similarity {best_match['similarity']:.3f}")

    # Demo: Wave Logic Gates
    print("\nDemonstrating wave logic gates...")

    # Test AND gate
    and_result = wave_computer.wave_logic_gates(
        input_patterns=[np.ones((10, 10)), np.ones((10, 10))],
        operation='and'
    )

    if and_result:
        print(f"AND gate result: {and_result['result_value']:.3f}")

    # Test OR gate
    or_result = wave_computer.wave_logic_gates(
        input_patterns=[np.ones((10, 10)), np.zeros((10, 10))],
        operation='or'
    )

    if or_result:
        print(f"OR gate result: {or_result['result_value']:.3f}")

    # Demo: Wave Memory
    print("\nDemonstrating wave memory...")

    # Create test data
    test_data = [0.8, -0.5, 0.3, 0.0, -0.2, 0.6]

    # Store and retrieve
    memory_result = wave_computer.wave_memory(
        input_data=test_data,
        storage_time=30,
        retrieval_time=20
    )

    # Compare input and retrieved data
    if memory_result:
        print("Original data:", test_data)
        print("Retrieved data:", [f"{val:.3f}" for val in memory_result['retrieved_data']])

    # Demo: Wave Filtering
    print("\nDemonstrating wave filtering...")

    # Create test signal
    t = np.linspace(0, 2*np.pi, 50)
    low_freq = 0.5 * np.sin(t)
    high_freq = 0.3 * np.sin(5*t)
    mixed_signal = low_freq + high_freq

    # Apply lowpass filter
    filter_result = wave_computer.wave_filter(
        input_signal=mixed_signal,
        filter_type='lowpass',
        cutoff_frequency=0.3
    )

    if filter_result:
        plt.figure(figsize=(10, 6))
        plt.plot(mixed_signal, label='Original Signal')
        plt.plot(filter_result['filtered_signal'], label='Filtered Signal')
        plt.legend()
        plt.title('Wave-Based Signal Filtering')
        plt.savefig('wave_filter.png')
        plt.close()

    print("\nThis wave-based computing model demonstrates how:")
    print("1. Information can be encoded in wave properties")
    print("2. Computation can emerge from wave interactions")
    print("3. Wave dynamics can perform signal processing")
    print("4. Memory can be implemented using persistent wave patterns")

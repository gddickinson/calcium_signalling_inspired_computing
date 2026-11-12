"""
Coordinated Recruitment Computing Based on Calcium Channel Dynamics

This simulation models a computational architecture inspired by how calcium channels
coordinate during calcium puffs. It demonstrates quorum sensing-like computation where
processing elements remain dormant until enough neighboring elements are activated.

The model implements:
1. Computing elements that self-organize into functional groups
2. Quorum-based activation thresholds
3. Coordinated recruitment of additional elements
4. Fault-tolerant signal processing

The classes live in the package modules (element.py, network.py, quorum.py); this file
re-exports them and keeps the original end-to-end demo: python coordinated_recruitment/coordinated_recruitment.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if __package__:
    from .element import CoordinatedRecruitmentElement
    from .network import CoordinatedRecruitmentNetwork
    from .quorum import QuorumComputer
else:  # run as a script: import the package from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from coordinated_recruitment.element import CoordinatedRecruitmentElement
    from coordinated_recruitment.network import CoordinatedRecruitmentNetwork
    from coordinated_recruitment.quorum import QuorumComputer

__all__ = ['CoordinatedRecruitmentElement', 'CoordinatedRecruitmentNetwork', 'QuorumComputer']

if __name__ == "__main__":
    # Initialize the model
    np.random.seed(42)  # For reproducibility
    network = CoordinatedRecruitmentNetwork(grid_size=(30, 30), coupling_radius=3, fault_probability=0.05)

    print("Running coordinated recruitment computing simulation...")

    # Create a simple input pattern (circle in the center)
    center = (network.height // 2, network.width // 2)
    network.set_input_region(center, radius=5, strength=0.7)

    # Run simulation
    network.run_simulation(steps=30)

    # Visualize state
    fig = network.visualize_state()
    plt.savefig('coordinated_recruitment_state.png')
    plt.close()

    # Create animation
    ani = network.create_animation(frames=30)
    # Uncomment to save animation:
    # ani.save('coordinated_recruitment_dynamics.mp4', writer='ffmpeg')

    # Test quorum computing applications
    quorum_computer = QuorumComputer(network)

    # Test 1: Fault tolerance
    print("\nTesting fault tolerance...")
    # Create test pattern
    test_pattern = np.zeros((network.height, network.width))
    test_pattern[10:20, 10:20] = 0.8

    fault_prob_range = np.linspace(0, 0.5, 6)
    fault_tolerance_results = quorum_computer.fault_tolerant_computation(
        input_patterns=[test_pattern],
        fault_prob_range=fault_prob_range
    )

    # Plot fault tolerance results
    plt.figure(figsize=(10, 5))
    fault_probs = [r['fault_prob'] for r in fault_tolerance_results]
    activations = [r['avg_activation'] for r in fault_tolerance_results]
    cluster_sizes = [r['avg_max_cluster'] for r in fault_tolerance_results]

    plt.subplot(1, 2, 1)
    plt.plot(fault_probs, activations, 'b-o')
    plt.xlabel('Fault Probability')
    plt.ylabel('Average Total Activation')
    plt.title('Fault Tolerance: Activation')

    plt.subplot(1, 2, 2)
    plt.plot(fault_probs, cluster_sizes, 'r-o')
    plt.xlabel('Fault Probability')
    plt.ylabel('Average Max Cluster Size')
    plt.title('Fault Tolerance: Clustering')

    plt.tight_layout()
    plt.savefig('fault_tolerance.png')
    plt.close()

    # Test 2: Signal amplification
    print("\nTesting signal amplification...")
    input_strengths = np.linspace(0.1, 1.0, 5)
    amplification_results = quorum_computer.signal_amplification(input_strengths)

    # Plot amplification results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for result in amplification_results:
        plt.plot(result['time_series'], label=f"Input={result['input_strength']:.1f}")

    plt.xlabel('Time Steps')
    plt.ylabel('Total Activation')
    plt.title('Signal Amplification Over Time')
    plt.legend()

    plt.subplot(1, 2, 2)
    strengths = [r['input_strength'] for r in amplification_results]
    ratios = [r['amplification_ratio'] for r in amplification_results]
    plt.plot(strengths, ratios, 'g-o')
    plt.axhline(y=1, color='k', linestyle='--', label='No Amplification')
    plt.xlabel('Input Strength')
    plt.ylabel('Amplification Ratio')
    plt.title('Signal Amplification by Input Strength')
    plt.legend()

    plt.tight_layout()
    plt.savefig('signal_amplification.png')
    plt.close()

    # Test 3: Self-organizing computation
    print("\nTesting self-organizing computation...")
    # Create two different patterns
    pattern1 = np.zeros((network.height, network.width))
    pattern1[5:15, 5:15] = 0.8

    pattern2 = np.zeros((network.height, network.width))
    pattern2[15:25, 15:25] = 0.8

    organization_results = quorum_computer.self_organizing_computation(pattern1, pattern2)

    # Plot self-organization results
    plt.figure(figsize=(10, 5))

    transitions = organization_results['cluster_transitions']
    times = [t['time'] for t in transitions]
    persisting = [t['persisting'] for t in transitions]
    new_elements = [t['new'] for t in transitions]
    disbanded = [t['disbanded'] for t in transitions]

    plt.stackplot(times, persisting, new_elements, disbanded,
                 labels=['Persisting', 'New', 'Disbanded'],
                 colors=['lightblue', 'lightgreen', 'salmon'])

    plt.axvline(x=10, color='k', linestyle='--', label='Start Transition')
    plt.axvline(x=10+30, color='k', linestyle=':', label='End Transition')

    plt.xlabel('Time Steps')
    plt.ylabel('Number of Elements')
    plt.title('Self-Organizing Computation During Transition')
    plt.legend()

    plt.savefig('self_organization.png')
    plt.close()

    print("\nThis coordinated recruitment computing model demonstrates how:")
    print("1. Computing elements can self-organize into functional groups")
    print("2. The network exhibits fault tolerance through redundancy")
    print("3. Weak signals can be amplified through coordinated recruitment")
    print("4. The system can adapt to changing inputs through self-organization")

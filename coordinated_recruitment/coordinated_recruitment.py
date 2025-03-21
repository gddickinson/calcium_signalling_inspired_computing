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
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from scipy.ndimage import convolve
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

class CoordinatedRecruitmentElement:
    def __init__(self, sensitivity=0.5, recruitment_threshold=0.3, coupling_strength=0.2,
                 recovery_time=5, max_activation=1.0):
        """
        Initialize a coordinated recruitment computing element.

        Parameters:
        - sensitivity: Sensitivity to input signals
        - recruitment_threshold: Threshold for recruiting neighboring elements
        - coupling_strength: Strength of coupling between elements
        - recovery_time: Time it takes to recover after activation
        - max_activation: Maximum activation level
        """
        self.sensitivity = sensitivity
        self.recruitment_threshold = recruitment_threshold
        self.coupling_strength = coupling_strength
        self.recovery_time = recovery_time
        self.max_activation = max_activation

        # State variables
        self.activation = 0.0  # Current activation level
        self.recovery_countdown = 0  # Time until recovery
        self.neighborhood_activation = 0.0  # Activation from neighbors
        self.fatigue = 0.0  # Fatigue level (increases with activation)

    def update(self, direct_input, neighborhood_activation):
        """
        Update element state based on direct input and neighborhood activation.

        Parameters:
        - direct_input: External input signal
        - neighborhood_activation: Activation level from neighbors

        Returns:
        - recruitment_signal: Signal to recruit neighbors
        """
        # Store neighborhood activation
        self.neighborhood_activation = neighborhood_activation

        # Calculate effective sensitivity (decreases with fatigue)
        effective_sensitivity = self.sensitivity * (1.0 - 0.5 * self.fatigue)

        # Update recovery countdown
        if self.recovery_countdown > 0:
            self.recovery_countdown -= 1
            # Reduce sensitivity during recovery
            effective_sensitivity *= 0.5

        # Calculate total input
        total_input = (direct_input * effective_sensitivity +
                      neighborhood_activation * self.coupling_strength)

        # Check for recruitment threshold
        recruitment_signal = 0.0

        if total_input > self.recruitment_threshold and self.recovery_countdown == 0:
            # Element gets recruited
            self.activation = min(self.max_activation, total_input)

            # Generate recruitment signal
            recruitment_signal = self.activation * 0.8

            # Update fatigue
            self.fatigue = min(1.0, self.fatigue + 0.1)

            # Set recovery countdown
            self.recovery_countdown = self.recovery_time
        else:
            # Decay activation
            self.activation *= 0.8

            # Recover from fatigue
            self.fatigue = max(0.0, self.fatigue - 0.01)

        return recruitment_signal

class CoordinatedRecruitmentNetwork:
    def __init__(self, grid_size=(30, 30), coupling_radius=3, fault_probability=0.0):
        """
        Initialize a network of coordinated recruitment elements.

        Parameters:
        - grid_size: Size of the 2D grid
        - coupling_radius: Radius for element coupling
        - fault_probability: Probability of element failure
        """
        self.height, self.width = grid_size
        self.coupling_radius = coupling_radius
        self.fault_probability = fault_probability

        # Create elements with varying properties
        self.elements = np.empty(grid_size, dtype=object)
        self.faulty_elements = set()  # Track faulty elements

        for i in range(self.height):
            for j in range(self.width):
                # Vary sensitivity across the grid
                sensitivity = 0.4 + 0.2 * np.random.randn()
                sensitivity = max(0.2, min(0.8, sensitivity))

                # Vary recruitment threshold
                threshold = 0.3 + 0.1 * np.random.randn()
                threshold = max(0.1, min(0.5, threshold))

                self.elements[i, j] = CoordinatedRecruitmentElement(
                    sensitivity=sensitivity,
                    recruitment_threshold=threshold
                )

                # Randomly mark elements as faulty
                if np.random.random() < fault_probability:
                    self.faulty_elements.add((i, j))

        # Create coupling kernel
        self.coupling_kernel = self._create_coupling_kernel(coupling_radius)

        # Keep track of activations and clusters
        self.activation_history = []
        self.cluster_history = []

        # Initialize direct input grid
        self.direct_input = np.zeros(grid_size)

    def _create_coupling_kernel(self, radius):
        """Create a kernel for element coupling based on distance"""
        size = 2 * radius + 1
        kernel = np.zeros((size, size))
        center = radius

        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist <= radius:
                    # Exponential decay with distance
                    kernel[i, j] = np.exp(-dist / (radius / 2))

        # Set center to 0 (no self-coupling)
        kernel[center, center] = 0

        # Normalize
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()

        return kernel

    def set_input_region(self, center, radius, strength):
        """
        Set a circular input region.

        Parameters:
        - center: (i, j) coordinates of the center
        - radius: Radius of the input region
        - strength: Strength of the input signal
        """
        center_i, center_j = center

        for i in range(self.height):
            for j in range(self.width):
                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                if dist <= radius:
                    # Reduce strength with distance
                    self.direct_input[i, j] = strength * (1 - dist/radius)

    def set_input_pattern(self, pattern):
        """
        Set an arbitrary input pattern.

        Parameters:
        - pattern: 2D array of input values (must match grid size)
        """
        if pattern.shape != (self.height, self.width):
            raise ValueError(f"Pattern shape {pattern.shape} doesn't match grid size ({self.height}, {self.width})")

        self.direct_input = pattern.copy()

    def update(self):
        """Update all elements in the network for one time step"""
        # Calculate current activation grid
        activation_grid = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.faulty_elements:
                    activation_grid[i, j] = self.elements[i, j].activation

        # Compute neighborhood activation using convolution
        neighborhood_grid = convolve(activation_grid, self.coupling_kernel, mode='constant', cval=0.0)

        # Update elements and collect new recruitment signals
        recruitment_grid = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.faulty_elements:
                    recruitment_grid[i, j] = self.elements[i, j].update(
                        self.direct_input[i, j],
                        neighborhood_grid[i, j]
                    )

        # Store current activation for history
        current_activation = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.faulty_elements:
                    current_activation[i, j] = self.elements[i, j].activation

        self.activation_history.append(current_activation)

        # Identify clusters of active elements
        clusters = self._identify_clusters(current_activation)
        self.cluster_history.append(clusters)

        # Limit history length
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)
            self.cluster_history.pop(0)

    def _identify_clusters(self, activation_grid, activation_threshold=0.3):
        """Identify clusters of activated elements"""
        # Extract positions of active elements
        active_positions = []
        for i in range(self.height):
            for j in range(self.width):
                if activation_grid[i, j] > activation_threshold:
                    active_positions.append((i, j))

        if not active_positions:
            return []

        # Convert to numpy array
        active_positions = np.array(active_positions)

        # Use DBSCAN to cluster active elements
        clustering = DBSCAN(eps=1.5, min_samples=3).fit(active_positions)

        # Group elements by cluster
        clusters = {}
        for pos, label in zip(active_positions, clustering.labels_):
            if label != -1:  # Skip noise points
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(tuple(pos))

        return list(clusters.values())

    def run_simulation(self, steps=100, input_pattern_func=None):
        """
        Run the simulation for a specified number of steps.

        Parameters:
        - steps: Number of simulation steps
        - input_pattern_func: Function to generate input pattern at each step (optional)

        Returns:
        - Dictionary of results
        """
        for step in range(steps):
            # Update input if function provided
            if input_pattern_func is not None:
                input_pattern = input_pattern_func(step)
                self.set_input_pattern(input_pattern)

            # Update network
            self.update()

        return self._analyze_results()

    def _analyze_results(self):
        """Analyze simulation results"""
        if not self.activation_history:
            return {}

        # Analyze cluster formation and stability
        cluster_sizes = []
        for clusters in self.cluster_history:
            sizes = [len(cluster) for cluster in clusters]
            cluster_sizes.append(sizes if sizes else [0])

        # Calculate average cluster size over time
        avg_cluster_sizes = [np.mean(sizes) if sizes else 0 for sizes in cluster_sizes]

        # Calculate number of clusters over time
        num_clusters = [len(clusters) for clusters in self.cluster_history]

        # Calculate total activation over time
        total_activation = [np.sum(act) for act in self.activation_history]

        return {
            'avg_cluster_sizes': avg_cluster_sizes,
            'num_clusters': num_clusters,
            'total_activation': total_activation,
            'cluster_history': self.cluster_history,
            'activation_history': self.activation_history
        }

    def visualize_state(self, time_step=-1):
        """
        Visualize network state at a specified time step.

        Parameters:
        - time_step: Time step to visualize (-1 for latest)
        """
        if not self.activation_history:
            return None

        # Ensure valid time step
        if time_step >= len(self.activation_history) or time_step < -len(self.activation_history):
            time_step = -1

        # Get state at the specified time step
        activation = self.activation_history[time_step]
        clusters = self.cluster_history[time_step]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot activation levels
        im = axes[0].imshow(activation, cmap='hot', vmin=0, vmax=1)
        axes[0].set_title('Element Activation Levels')
        plt.colorbar(im, ax=axes[0])

        # Mark faulty elements
        for i, j in self.faulty_elements:
            axes[0].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='blue', linewidth=1))

        # Plot clusters with different colors
        cluster_map = np.zeros((self.height, self.width))
        for cluster_idx, cluster in enumerate(clusters, 1):
            for i, j in cluster:
                cluster_map[i, j] = cluster_idx

        im2 = axes[1].imshow(cluster_map, cmap='tab20', interpolation='nearest')
        axes[1].set_title(f'Element Clusters (t={time_step})')

        if clusters:
            plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        return fig

    def create_animation(self, frames=50):
        """Create animation of network dynamics"""
        if len(self.activation_history) < frames:
            print("Not enough history. Running simulation...")
            self.run_simulation(steps=frames)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Initialize with first frame
        im = ax.imshow(self.activation_history[0], cmap='hot', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_title('Element Activation Dynamics')

        # Mark faulty elements
        for i, j in self.faulty_elements:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='blue', linewidth=1))

        def update(frame):
            im.set_array(self.activation_history[frame])
            ax.set_title(f'Element Activation Dynamics (t={frame})')
            return [im]

        ani = FuncAnimation(fig, update, frames=min(frames, len(self.activation_history)),
                           interval=150, blit=True)

        return ani

    def visualize_clusters_3d(self):
        """Visualize cluster formation over time in 3D"""
        if len(self.cluster_history) < 2:
            return None

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot clusters over time
        for t, clusters in enumerate(self.cluster_history):
            for cluster_idx, cluster in enumerate(clusters):
                # Get cluster coordinates
                xs = [j for i, j in cluster]  # x-coordinates are j values (columns)
                ys = [i for i, j in cluster]  # y-coordinates are i values (rows)
                zs = [t] * len(cluster)  # z-coordinate is time step

                # Plot with unique color based on cluster index
                color = plt.cm.tab20(cluster_idx % 20)
                ax.scatter(xs, ys, zs, c=[color], s=30, alpha=0.7)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Time')
        ax.set_title('Cluster Formation Over Time')

        return fig


# Implementation of coordinated recruitment computing applications
class QuorumComputer:
    def __init__(self, network):
        self.network = network

    def fault_tolerant_computation(self, input_patterns, fault_prob_range, trials=5):
        """
        Test fault tolerance by increasing fault probability.

        Parameters:
        - input_patterns: List of input patterns to test
        - fault_prob_range: Range of fault probabilities to test
        - trials: Number of trials per fault probability

        Returns:
        - Results for analysis
        """
        results = []

        for fault_prob in fault_prob_range:
            trial_results = []

            for _ in range(trials):
                # Create a new network with the specified fault probability
                test_network = CoordinatedRecruitmentNetwork(
                    grid_size=(self.network.height, self.network.width),
                    coupling_radius=self.network.coupling_radius,
                    fault_probability=fault_prob
                )

                # Run simulation for each input pattern
                for pattern in input_patterns:
                    test_network.set_input_pattern(pattern)
                    test_network.run_simulation(steps=20)

                    # Extract results
                    analysis = test_network._analyze_results()

                    # Track key metrics
                    final_activation = analysis['total_activation'][-1] if analysis['total_activation'] else 0
                    max_cluster_size = max(analysis['avg_cluster_sizes']) if analysis['avg_cluster_sizes'] else 0

                    trial_results.append({
                        'fault_prob': fault_prob,
                        'final_activation': final_activation,
                        'max_cluster_size': max_cluster_size,
                        'num_clusters': analysis['num_clusters'][-1] if analysis['num_clusters'] else 0
                    })

            # Average results across trials
            avg_activation = np.mean([r['final_activation'] for r in trial_results])
            avg_max_cluster = np.mean([r['max_cluster_size'] for r in trial_results])
            avg_num_clusters = np.mean([r['num_clusters'] for r in trial_results])

            results.append({
                'fault_prob': fault_prob,
                'avg_activation': avg_activation,
                'avg_max_cluster': avg_max_cluster,
                'avg_num_clusters': avg_num_clusters
            })

        return results

    def signal_amplification(self, input_strengths, duration=20):
        """
        Demonstrate signal amplification for weak inputs.

        Parameters:
        - input_strengths: List of input strengths to test
        - duration: Duration of the simulation

        Returns:
        - Amplification results for each input strength
        """
        results = []

        for strength in input_strengths:
            # Create input pattern (central circle)
            pattern = np.zeros((self.network.height, self.network.width))
            center_i, center_j = self.network.height // 2, self.network.width // 2
            radius = min(self.network.height, self.network.width) // 6

            for i in range(self.network.height):
                for j in range(self.network.width):
                    dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                    if dist <= radius:
                        pattern[i, j] = strength

            # Reset network and run simulation
            self.network = CoordinatedRecruitmentNetwork(
                grid_size=(self.network.height, self.network.width),
                coupling_radius=self.network.coupling_radius
            )

            self.network.set_input_pattern(pattern)
            self.network.run_simulation(steps=duration)

            # Calculate amplification (ratio of final total activation to initial input)
            initial_input = pattern.sum()
            final_activation = self.network.activation_history[-1].sum() if self.network.activation_history else 0

            amplification_ratio = final_activation / initial_input if initial_input > 0 else 0

            results.append({
                'input_strength': strength,
                'initial_input': initial_input,
                'final_activation': final_activation,
                'amplification_ratio': amplification_ratio,
                'time_series': [act.sum() for act in self.network.activation_history]
            })

        return results

    def self_organizing_computation(self, input_pattern1, input_pattern2, blend_steps=30):
        """
        Demonstrate how computation self-organizes when transitioning between inputs.

        Parameters:
        - input_pattern1: First input pattern
        - input_pattern2: Second input pattern
        - blend_steps: Number of steps to blend from pattern1 to pattern2

        Returns:
        - Analysis of emergent computation
        """
        # Define blending function for inputs
        def blend_patterns(step):
            if step < 10:
                return input_pattern1
            elif step >= 10 + blend_steps:
                return input_pattern2
            else:
                # Linear blend
                alpha = (step - 10) / blend_steps
                return (1 - alpha) * input_pattern1 + alpha * input_pattern2

        # Reset network
        self.network = CoordinatedRecruitmentNetwork(
            grid_size=(self.network.height, self.network.width),
            coupling_radius=self.network.coupling_radius
        )

        # Run simulation with blending inputs
        total_steps = 10 + blend_steps + 20  # Initial, blend, final
        self.network.run_simulation(steps=total_steps, input_pattern_func=blend_patterns)

        # Track cluster reorganization
        cluster_transitions = []
        for t in range(1, len(self.network.cluster_history)):
            prev_clusters = self.network.cluster_history[t-1]
            curr_clusters = self.network.cluster_history[t]

            # Count new, persisting and disbanded clusters
            prev_elements = set()
            for cluster in prev_clusters:
                prev_elements.update(cluster)

            curr_elements = set()
            for cluster in curr_clusters:
                curr_elements.update(cluster)

            persisting = len(prev_elements.intersection(curr_elements))
            new_elements = len(curr_elements - prev_elements)
            disbanded = len(prev_elements - curr_elements)

            cluster_transitions.append({
                'time': t,
                'persisting': persisting,
                'new': new_elements,
                'disbanded': disbanded,
                'num_clusters': len(curr_clusters)
            })

        return {
            'cluster_transitions': cluster_transitions,
            'activation_history': self.network.activation_history
        }


# Example usage
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

"""
Analog Computing with Calcium-Like Thresholds

This simulation models a computational architecture inspired by calcium signaling
thresholds and channel opening dynamics. It demonstrates how analog computing
can emerge from threshold-based activation similar to calcium channel recruitment
during puffs and waves.

The model implements:
1. Threshold logic units with calcium-like activation
2. Analog memory elements (variable states) rather than binary values
3. Integration of small signals over time until reaching threshold
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from scipy.ndimage import gaussian_filter

class ThresholdComputingUnit:
    def __init__(self, threshold=0.7, activation_slope=10, refractory_period=5, 
                 integration_rate=0.1, recovery_rate=0.05):
        """
        Initialize a computing unit with threshold behavior inspired by calcium channels.
        
        Parameters:
        - threshold: Activation threshold (analogous to calcium channel opening threshold)
        - activation_slope: Steepness of the activation curve
        - refractory_period: Time steps where unit is less responsive after activation
        - integration_rate: Rate at which input signals are integrated
        - recovery_rate: Rate of recovery from refractory state
        """
        self.threshold = threshold
        self.activation_slope = activation_slope
        self.refractory_period = refractory_period
        self.integration_rate = integration_rate
        self.recovery_rate = recovery_rate
        
        # State variables
        self.membrane_potential = 0.0  # Analogous to local calcium concentration
        self.activation = 0.0  # Output activation level
        self.refractory_countdown = 0  # Refractory period counter
        self.internal_state = 0.0  # Internal memory (analogous to slow calcium buffers)
    
    def update(self, input_signal):
        """
        Update the unit's state based on input signal.
        
        Parameters:
        - input_signal: Input value (could be a single value or array)
        
        Returns:
        - activation: Current activation level
        """
        # Update refractory period
        if self.refractory_countdown > 0:
            self.refractory_countdown -= 1
        
        # Calculate effective threshold (higher during refractory period)
        effective_threshold = self.threshold
        if self.refractory_countdown > 0:
            effective_threshold += 0.2 * (self.refractory_countdown / self.refractory_period)
        
        # Integrate input signal (with influence from internal state)
        integration_factor = self.integration_rate * (1.0 - 0.8 * (self.refractory_countdown / self.refractory_period))
        self.membrane_potential += integration_factor * input_signal + 0.01 * self.internal_state
        
        # Apply leakage
        self.membrane_potential *= 0.95
        
        # Calculate activation using sigmoid function
        x = self.activation_slope * (self.membrane_potential - effective_threshold)
        self.activation = 1.0 / (1.0 + np.exp(-x))
        
        # Update internal state (slow dynamics)
        self.internal_state = 0.99 * self.internal_state + 0.01 * self.activation
        
        # Check if threshold crossed for refractory period
        if self.membrane_potential > effective_threshold and self.refractory_countdown == 0:
            self.refractory_countdown = self.refractory_period
        
        return self.activation

class ThresholdComputingNetwork:
    def __init__(self, size=(10, 10), connectivity_radius=2, connectivity_prob=0.3):
        """
        Initialize a network of threshold computing units.
        
        Parameters:
        - size: Tuple (height, width) specifying grid dimensions
        - connectivity_radius: Max distance for connections between units
        - connectivity_prob: Probability of connection within radius
        """
        self.height, self.width = size
        self.units = np.empty(size, dtype=object)
        
        # Create units with slightly varying parameters
        for i in range(self.height):
            for j in range(self.width):
                threshold = 0.7 + 0.1 * np.random.randn()
                activation_slope = 10 + np.random.randn()
                self.units[i, j] = ThresholdComputingUnit(
                    threshold=max(0.4, threshold),
                    activation_slope=max(5, activation_slope)
                )
        
        # Create connectivity graph
        self.connections = self._create_connections(connectivity_radius, connectivity_prob)
        
        # Create input and output regions
        self.input_region = [(i, j) for i in range(self.height) for j in range(self.width) 
                             if i < 2]  # Top rows are input
        
        self.output_region = [(i, j) for i in range(self.height) for j in range(self.width) 
                              if i >= self.height-2]  # Bottom rows are output
        
        # State history
        self.activation_history = []
        self.membrane_potential_history = []
    
    def _create_connections(self, radius, probability):
        """Create connections between units based on radius and probability"""
        connections = {}
        
        for i1 in range(self.height):
            for j1 in range(self.width):
                connections[(i1, j1)] = []
                
                # Consider connections within radius
                for i2 in range(max(0, i1-radius), min(self.height, i1+radius+1)):
                    for j2 in range(max(0, j1-radius), min(self.width, j1+radius+1)):
                        if (i1, j1) != (i2, j2):  # No self-connections
                            dist = np.sqrt((i1-i2)**2 + (j1-j2)**2)
                            if dist <= radius and np.random.random() < probability:
                                # Create connection with random weight
                                weight = 0.5 + 0.5 * np.random.randn()
                                connections[(i1, j1)].append(((i2, j2), weight))
        
        return connections
    
    def set_input(self, input_pattern):
        """Set input to the network"""
        # Validate input pattern shape
        if len(input_pattern) != len(self.input_region):
            raise ValueError(f"Input pattern length ({len(input_pattern)}) must match input region size ({len(self.input_region)})")
        
        # Apply inputs
        for (i, j), value in zip(self.input_region, input_pattern):
            self.units[i, j].membrane_potential = value
    
    def get_output(self):
        """Get current output from the network"""
        return [self.units[i, j].activation for (i, j) in self.output_region]
    
    def update(self):
        """Update all units in the network for one time step"""
        # Calculate new inputs based on current activations
        new_inputs = np.zeros((self.height, self.width))
        
        for i in range(self.height):
            for j in range(self.width):
                unit_pos = (i, j)
                
                # Sum weighted inputs from connected units
                for (ni, nj), weight in self.connections[unit_pos]:
                    new_inputs[i, j] += weight * self.units[ni, nj].activation
        
        # Update all units with their new inputs
        activations = np.zeros((self.height, self.width))
        potentials = np.zeros((self.height, self.width))
        
        for i in range(self.height):
            for j in range(self.width):
                activations[i, j] = self.units[i, j].update(new_inputs[i, j])
                potentials[i, j] = self.units[i, j].membrane_potential
        
        # Store history
        self.activation_history.append(activations.copy())
        self.membrane_potential_history.append(potentials.copy())
        
        # Limit history length
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)
            self.membrane_potential_history.pop(0)
    
    def run_simulation(self, steps=100, input_patterns=None):
        """
        Run simulation for specified number of steps.
        
        Parameters:
        - steps: Number of simulation steps
        - input_patterns: List of input patterns to present (optional)
        """
        outputs = []
        
        for step in range(steps):
            # Apply input if provided
            if input_patterns is not None and step < len(input_patterns):
                self.set_input(input_patterns[step])
            
            # Update network
            self.update()
            
            # Collect output
            outputs.append(self.get_output())
        
        return outputs
    
    def visualize_state(self):
        """Visualize current network state"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract activations and membrane potentials
        activations = np.zeros((self.height, self.width))
        potentials = np.zeros((self.height, self.width))
        
        for i in range(self.height):
            for j in range(self.width):
                activations[i, j] = self.units[i, j].activation
                potentials[i, j] = self.units[i, j].membrane_potential
        
        # Plot activations
        im1 = axes[0].imshow(activations, cmap='hot', vmin=0, vmax=1)
        axes[0].set_title('Unit Activations')
        plt.colorbar(im1, ax=axes[0])
        
        # Mark input and output regions
        for i, j in self.input_region:
            axes[0].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2))
        
        for i, j in self.output_region:
            axes[0].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='green', linewidth=2))
        
        # Plot membrane potentials
        im2 = axes[1].imshow(potentials, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Membrane Potentials')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        return fig
    
    def visualize_connectivity(self):
        """Visualize network connectivity"""
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(self.height):
            for j in range(self.width):
                node_id = f"{i},{j}"
                
                # Determine node type (input, output, or hidden)
                if (i, j) in self.input_region:
                    node_type = 'input'
                elif (i, j) in self.output_region:
                    node_type = 'output'
                else:
                    node_type = 'hidden'
                
                G.add_node(node_id, pos=(j, -i), type=node_type)
        
        # Add edges
        for (i1, j1), connections in self.connections.items():
            source = f"{i1},{j1}"
            for (i2, j2), weight in connections:
                target = f"{i2},{j2}"
                G.add_edge(source, target, weight=weight)
        
        # Get positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node]['type']
            if node_type == 'input':
                node_colors.append('skyblue')
            elif node_type == 'output':
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
        
        # Draw edges with colors based on weights
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        # Normalize weights for coloring
        norm_weights = [(w - min(weights)) / (max(weights) - min(weights) + 1e-6) for w in weights]
        
        # Generate colors from weights (red for negative, blue for positive)
        edge_colors = []
        for w in weights:
            if w >= 0:
                edge_colors.append((0, 0, 1, min(1, abs(w))))  # Blue
            else:
                edge_colors.append((1, 0, 0, min(1, abs(w))))  # Red
        
        nx.draw_networkx_edges(G, pos, width=1, alpha=0.6, edge_color=edge_colors)
        
        plt.title('Network Connectivity')
        plt.axis('off')
        
        return plt.gcf()
    
    def create_animation(self, frames=100):
        """Create animation of network dynamics"""
        # Run simulation first to collect history
        if len(self.activation_history) < frames:
            self.run_simulation(steps=frames)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Initialize with first frame
        im = ax.imshow(self.activation_history[0], cmap='hot', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_title('Unit Activations Over Time')
        
        def update(frame):
            im.set_array(self.activation_history[frame])
            return [im]
        
        ani = FuncAnimation(fig, update, frames=min(frames, len(self.activation_history)), 
                           interval=100, blit=True)
        
        return ani

# Example application: Signal integration and threshold-based computation
class ThresholdLogicComputer:
    def __init__(self, network):
        self.network = network
    
    def temporal_integration(self, input_sequence, duration=50):
        """
        Demonstrate temporal integration of weak signals.
        
        Parameters:
        - input_sequence: Sequence of input patterns
        - duration: Duration to run the simulation
        
        Returns:
        - output_sequence: Sequence of outputs
        """
        # Pad input sequence if needed
        if len(input_sequence) < duration:
            padded_sequence = input_sequence + [np.zeros(len(self.network.input_region))] * (duration - len(input_sequence))
        else:
            padded_sequence = input_sequence[:duration]
        
        # Run simulation
        outputs = self.network.run_simulation(steps=duration, input_patterns=padded_sequence)
        
        return outputs
    
    def threshold_detection(self, signal_strengths, duration_per_signal=20):
        """
        Demonstrate threshold detection with varying signal strengths.
        
        Parameters:
        - signal_strengths: List of signal strengths to test
        - duration_per_signal: Duration to apply each signal
        
        Returns:
        - detection_results: Whether each signal was detected
        """
        detection_results = []
        
        for strength in signal_strengths:
            # Create input pattern with given strength
            input_pattern = [strength] * len(self.network.input_region)
            input_sequence = [input_pattern] * duration_per_signal
            
            # Run simulation
            outputs = self.network.run_simulation(steps=duration_per_signal, input_patterns=input_sequence)
            
            # Check if output exceeded detection threshold (0.5)
            max_output = max([max(output) for output in outputs])
            detected = max_output > 0.5
            
            detection_results.append((strength, max_output, detected))
        
        return detection_results
    
    def analog_memory(self, input_values, retention_time=30):
        """
        Demonstrate analog memory capabilities.
        
        Parameters:
        - input_values: Input values to store
        - retention_time: How long to check retention
        
        Returns:
        - memory_trace: How values are retained over time
        """
        memory_traces = []
        
        for value in input_values:
            # Set input
            input_pattern = [value] * len(self.network.input_region)
            
            # Apply input for 5 steps then remove
            input_sequence = [input_pattern] * 5 + [np.zeros(len(self.network.input_region))] * (retention_time - 5)
            
            # Run simulation
            outputs = self.network.run_simulation(steps=retention_time, input_patterns=input_sequence)
            
            # Track output over time
            output_trace = [max(output) for output in outputs]
            memory_traces.append((value, output_trace))
        
        return memory_traces

# Example usage
if __name__ == "__main__":
    # Initialize the model
    np.random.seed(42)  # For reproducibility
    network = ThresholdComputingNetwork(size=(10, 10), connectivity_radius=3, connectivity_prob=0.3)
    
    # Visualize the network
    print("Visualizing threshold computing network...")
    network.visualize_connectivity()
    plt.savefig('threshold_network_connectivity.png')
    plt.close()
    
    # Run a basic simulation
    print("Running basic simulation...")
    input_pattern = np.random.random(len(network.input_region)) * 0.8
    network.set_input(input_pattern)
    network.run_simulation(steps=30)
    
    # Visualize state
    network.visualize_state()
    plt.savefig('threshold_network_state.png')
    plt.close()
    
    # Create an animation
    ani = network.create_animation(frames=30)
    # Uncomment to save animation:
    # ani.save('threshold_network_dynamics.mp4', writer='ffmpeg')
    
    # Test threshold-based computing applications
    threshold_computer = ThresholdLogicComputer(network)
    
    # Test 1: Temporal Integration
    print("\nTesting temporal integration...")
    weak_inputs = [np.random.random(len(network.input_region)) * 0.3 for _ in range(20)]
    output_sequence = threshold_computer.temporal_integration(weak_inputs, duration=40)
    
    # Plot integration results
    plt.figure(figsize=(10, 4))
    plt.plot([max(output) for output in output_sequence])
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Time Steps')
    plt.ylabel('Maximum Output Activation')
    plt.title('Temporal Integration of Weak Signals')
    plt.legend()
    plt.savefig('temporal_integration.png')
    plt.close()
    
    # Test 2: Threshold Detection
    print("\nTesting threshold detection...")
    signal_strengths = np.linspace(0.1, 1.0, 10)
    detection_results = threshold_computer.threshold_detection(signal_strengths)
    
    # Plot detection results
    strengths, max_outputs, detected = zip(*detection_results)
    plt.figure(figsize=(10, 4))
    plt.plot(strengths, max_outputs, 'bo-')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Detection Threshold')
    
    # Mark detected signals
    for strength, output, detect in detection_results:
        if detect:
            plt.plot(strength, output, 'go', markersize=10)
        else:
            plt.plot(strength, output, 'ro', markersize=10)
    
    plt.xlabel('Signal Strength')
    plt.ylabel('Maximum Output Activation')
    plt.title('Threshold Detection of Signals')
    plt.legend()
    plt.savefig('threshold_detection.png')
    plt.close()
    
    # Test 3: Analog Memory
    print("\nTesting analog memory capabilities...")
    input_values = [0.2, 0.5, 0.8]
    memory_traces = threshold_computer.analog_memory(input_values)
    
    # Plot memory traces
    plt.figure(figsize=(10, 4))
    for value, trace in memory_traces:
        plt.plot(trace, label=f'Input = {value:.1f}')
    
    plt.axvline(x=5, color='k', linestyle='--', label='Input Removed')
    plt.xlabel('Time Steps')
    plt.ylabel('Output Activation')
    plt.title('Analog Memory Retention')
    plt.legend()
    plt.savefig('analog_memory.png')
    plt.close()
    
    print("\nThis threshold computing model demonstrates how:")
    print("1. Computation can emerge from threshold-based activation")
    print("2. Weak signals can be integrated over time")
    print("3. Analog values can be stored in system state")
    print("4. Threshold detection creates decision boundaries")
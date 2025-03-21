"""
Spatiotemporal Computing Inspired by Calcium Wave Propagation

This simulation models a computational architecture inspired by calcium waves
and IP3 diffusion constraints. It demonstrates how computation can emerge from
spatial patterns and temporal evolution of signals, similar to how information 
is processed through calcium wave propagation in biological systems.

The model implements a reaction-diffusion system where:
1. Information is encoded in wave patterns
2. Processing occurs through wave interactions
3. Restricted diffusion creates computational constraints similar to IP3 diffusion
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class SpatiotemporalComputing:
    def __init__(self, grid_size=100, n_channels=3, diffusion_rates=None, reaction_params=None):
        """
        Initialize a spatiotemporal computing model based on calcium wave dynamics.
        
        Parameters:
        - grid_size: Size of the 2D grid
        - n_channels: Number of chemical species (analogous to calcium, IP3, etc.)
        - diffusion_rates: Diffusion rates for each channel
        - reaction_params: Parameters governing interactions between channels
        """
        self.grid_size = grid_size
        self.n_channels = n_channels
        
        # Initialize state grid for each channel
        self.state = np.zeros((n_channels, grid_size, grid_size))
        
        # Set diffusion rates (different for each channel)
        if diffusion_rates is None:
            # Default diffusion rates - channel 0 could represent calcium, channel 1 IP3, etc.
            self.diffusion_rates = np.array([0.6, 0.2, 0.05])  # Fast, medium, slow diffusion
        else:
            self.diffusion_rates = np.array(diffusion_rates)
        
        # Set reaction parameters (how channels interact)
        if reaction_params is None:
            # Simple default interaction matrix
            self.reaction_params = np.array([
                [0.0, 0.2, -0.1],   # Channel 0 effects on all channels
                [0.3, -0.1, 0.2],   # Channel 1 effects on all channels
                [-0.2, 0.1, 0.0]    # Channel 2 effects on all channels
            ])
        else:
            self.reaction_params = np.array(reaction_params)
        
        # Create different diffusion kernels for each channel
        self.diffusion_kernels = [self._create_diffusion_kernel(rate) for rate in self.diffusion_rates]
        
        # Create spatial heterogeneity (analogous to different cellular regions)
        self.spatial_regions = self._create_spatial_regions()
        
        # State history for analysis
        self.history = []
        
        # Initialize with some random patterns
        self._initialize_patterns()
    
    def _create_spatial_regions(self):
        """Create spatial regions with different properties (analogous to ER, cytosol, etc.)"""
        regions = np.zeros((self.grid_size, self.grid_size))
        
        # Create circular regions
        for _ in range(5):
            center_x = np.random.randint(0, self.grid_size)
            center_y = np.random.randint(0, self.grid_size)
            radius = np.random.randint(5, 20)
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist < radius:
                        regions[i, j] = np.random.randint(1, 4)  # Different region types
        
        return regions
    
    def _create_diffusion_kernel(self, diffusion_rate):
        """Create a diffusion kernel based on the diffusion rate"""
        size = 5  # Kernel size
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                kernel[i, j] = np.exp(-dist**2 / diffusion_rate)
        
        # Normalize the kernel
        return kernel / kernel.sum()
    
    def _initialize_patterns(self):
        """Initialize the system with some patterns as inputs"""
        # Add some random noise
        self.state += np.random.normal(0, 0.01, self.state.shape)
        
        # Add specific patterns in different regions
        # Pattern 1: A small square pulse in channel 0
        self.state[0, 20:30, 20:30] = 1.0
        
        # Pattern 2: A line in channel 1
        self.state[1, 40:45, 10:90] = 0.8
        
        # Pattern 3: A gradient in channel 2
        for i in range(self.grid_size):
            self.state[2, i, :] += i / self.grid_size
    
    def update(self):
        """Update the state for one time step"""
        # Save current state to history
        self.history.append(self.state.copy())
        if len(self.history) > 100:  # Limit history size
            self.history.pop(0)
        
        # Apply diffusion to each channel
        new_state = self.state.copy()
        for i in range(self.n_channels):
            new_state[i] = convolve2d(self.state[i], self.diffusion_kernels[i], mode='same', boundary='wrap')
        
        # Apply reaction dynamics (interactions between channels)
        for i in range(self.n_channels):
            reaction_term = np.zeros((self.grid_size, self.grid_size))
            for j in range(self.n_channels):
                reaction_term += self.reaction_params[j, i] * self.state[j]
            
            # Apply the reaction term
            new_state[i] += reaction_term
            
            # Modify reaction based on spatial regions
            region_effect = 0.1 * (self.spatial_regions == i+1)
            new_state[i] += region_effect
        
        # Apply nonlinearities and constraints
        new_state = np.tanh(new_state)  # Nonlinear activation
        
        # Update state
        self.state = new_state
    
    def run_simulation(self, steps=100):
        """Run the simulation for a specified number of steps"""
        for _ in range(steps):
            self.update()
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze the spatiotemporal patterns that emerged"""
        if not self.history:
            return {}
        
        # Extract the last several states for analysis
        recent_states = self.history[-20:]
        
        # Reshape for analysis
        flattened_states = [state.reshape(self.n_channels, -1) for state in recent_states]
        
        # Analyze spatial patterns
        spatial_analysis = self._analyze_spatial_patterns(recent_states[-1])
        
        # Analyze temporal patterns
        temporal_analysis = self._analyze_temporal_patterns(flattened_states)
        
        return {
            'spatial_analysis': spatial_analysis,
            'temporal_analysis': temporal_analysis
        }
    
    def _analyze_spatial_patterns(self, state):
        """Analyze spatial patterns in the current state"""
        # Calculate gradient magnitude for each channel
        gradients = []
        for i in range(self.n_channels):
            dx = np.gradient(state[i], axis=0)
            dy = np.gradient(state[i], axis=1)
            gradients.append(np.sqrt(dx**2 + dy**2))
        
        # Detect wave fronts (high gradient areas)
        wave_fronts = []
        for grad in gradients:
            threshold = np.percentile(grad, 80)  # Top 20% of gradient values
            wave_fronts.append(grad > threshold)
        
        # Calculate correlations between channels
        correlations = np.corrcoef([state[i].flatten() for i in range(self.n_channels)])
        
        return {
            'gradients': gradients,
            'wave_fronts': wave_fronts,
            'correlations': correlations
        }
    
    def _analyze_temporal_patterns(self, states):
        """Analyze temporal patterns in the state history"""
        # Analyze each channel separately
        results = []
        for channel in range(self.n_channels):
            # Extract channel data across time
            channel_data = np.array([state[channel] for state in states])
            
            # Reshape to 2D array: (timesteps, spatial_points)
            reshaped_data = channel_data.reshape(len(states), -1)
            
            # Apply PCA to find dominant temporal patterns
            if reshaped_data.shape[0] > 1:  # Need at least 2 timesteps for PCA
                pca = PCA(n_components=min(5, reshaped_data.shape[0]-1))
                pca.fit(reshaped_data)
                
                results.append({
                    'variance_explained': pca.explained_variance_ratio_,
                    'components': pca.components_
                })
            else:
                results.append({})
        
        return results
    
    def visualize(self, results=None):
        """Visualize the current state and analysis results"""
        if not self.history:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot current state for each channel
        for i in range(self.n_channels):
            im = axes[0, i].imshow(self.state[i], cmap='viridis', 
                                   vmin=-1, vmax=1, 
                                   interpolation='nearest')
            axes[0, i].set_title(f'Channel {i} State')
            plt.colorbar(im, ax=axes[0, i])
        
        # Plot additional analysis if available
        if results and 'spatial_analysis' in results:
            # Plot wave fronts (overlay on gradient)
            for i in range(min(self.n_channels, 3)):
                gradients = results['spatial_analysis']['gradients'][i]
                wave_fronts = results['spatial_analysis']['wave_fronts'][i]
                
                axes[1, i].imshow(gradients, cmap='gray')
                axes[1, i].imshow(wave_fronts, cmap='autumn', alpha=0.5)
                axes[1, i].set_title(f'Channel {i} Wave Fronts')
        
        plt.tight_layout()
        return fig
    
    def create_animation(self, frames=100):
        """Create an animation of the spatiotemporal dynamics"""
        fig, axes = plt.subplots(1, self.n_channels, figsize=(5*self.n_channels, 4))
        
        if self.n_channels == 1:
            axes = [axes]  # Make iterable if only one channel
        
        images = []
        for i, ax in enumerate(axes):
            im = ax.imshow(self.state[i], cmap='viridis', vmin=-1, vmax=1)
            ax.set_title(f'Channel {i}')
            plt.colorbar(im, ax=ax)
            images.append(im)
        
        def update(frame):
            self.update()
            for i, im in enumerate(images):
                im.set_array(self.state[i])
            return images
        
        ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        return ani

# Implementation of a spatiotemporal computing task
class PatternRecognitionComputer:
    def __init__(self, model):
        self.model = model
        self.reference_patterns = {}
        self.pattern_graphs = {}
    
    def train(self, patterns):
        """
        Train the system to recognize patterns.
        
        Parameters:
        - patterns: Dictionary of named patterns, each a numpy array matching state dimensions
        """
        self.reference_patterns = patterns
        
        # For each pattern, compute its graph representation
        for name, pattern in patterns.items():
            # Initialize model with the pattern
            self.model.state = pattern.copy()
            
            # Run simulation to see how pattern evolves
            self.model.run_simulation(steps=20)
            
            # Extract the pattern's dynamic signature
            graph = self._compute_pattern_graph(self.model.history)
            self.pattern_graphs[name] = graph
    
    def _compute_pattern_graph(self, history):
        """Compute a graph representation of the pattern dynamics"""
        # Create a graph where nodes are states and edges represent transitions
        G = nx.DiGraph()
        
        # Extract key states by clustering
        states = [state.flatten() for state in history]
        
        if len(states) > 5:  # Need enough states for clustering
            # Reduce dimensionality for clustering
            pca = PCA(n_components=min(10, len(states)-1))
            reduced_states = pca.fit_transform(states)
            
            # Cluster into 5 representative states
            kmeans = KMeans(n_clusters=min(5, len(states)), random_state=42)
            clusters = kmeans.fit_predict(reduced_states)
            
            # Add nodes and edges to graph
            for i in range(len(states)-1):
                G.add_edge(clusters[i], clusters[i+1])
        
        return G
    
    def recognize(self, input_pattern, steps=20):
        """
        Recognize an input pattern by comparing its dynamics to reference patterns.
        
        Parameters:
        - input_pattern: Pattern to recognize
        - steps: Number of simulation steps
        
        Returns:
        - Best matching pattern name and similarity score
        """
        # Initialize model with input pattern
        self.model.state = input_pattern.copy()
        
        # Run simulation
        self.model.run_simulation(steps=steps)
        
        # Compute graph representation
        input_graph = self._compute_pattern_graph(self.model.history)
        
        # Compare with reference patterns
        best_match = None
        best_score = -float('inf')
        
        for name, ref_graph in self.pattern_graphs.items():
            # Calculate graph similarity
            similarity = self._graph_similarity(input_graph, ref_graph)
            
            if similarity > best_score:
                best_score = similarity
                best_match = name
        
        return best_match, best_score
    
    def _graph_similarity(self, graph1, graph2):
        """Calculate similarity between two graphs"""
        # Simple measure: common edges ratio
        edges1 = set(graph1.edges())
        edges2 = set(graph2.edges())
        
        if not edges1 or not edges2:
            return 0.0
        
        common_edges = len(edges1.intersection(edges2))
        total_edges = len(edges1.union(edges2))
        
        if total_edges == 0:
            return 0.0
        
        return common_edges / total_edges

# Example usage
if __name__ == "__main__":
    # Initialize the model
    np.random.seed(42)  # For reproducibility
    model = SpatiotemporalComputing(grid_size=50, n_channels=3)
    
    # Run simulation
    print("Running spatiotemporal computing simulation...")
    results = model.run_simulation(steps=100)
    
    # Visualize results
    fig = model.visualize(results)
    plt.show()
    
    # Create an animation
    ani = model.create_animation(frames=50)
    # Uncomment to save animation:
    # ani.save('spatiotemporal_dynamics.mp4', writer='ffmpeg')
    
    # Demonstrate pattern recognition
    pattern_recognizer = PatternRecognitionComputer(model)
    
    # Create some reference patterns
    patterns = {}
    
    # Pattern 1: Central pulse
    pattern1 = np.zeros((3, 50, 50))
    pattern1[0, 20:30, 20:30] = 1.0
    patterns['central_pulse'] = pattern1
    
    # Pattern 2: Horizontal stripe
    pattern2 = np.zeros((3, 50, 50))
    pattern2[1, 20:30, :] = 0.8
    patterns['horizontal_stripe'] = pattern2
    
    # Pattern 3: Vertical stripe
    pattern3 = np.zeros((3, 50, 50))
    pattern3[2, :, 20:30] = 0.8
    patterns['vertical_stripe'] = pattern3
    
    # Train the pattern recognizer
    print("\nTraining pattern recognition system...")
    pattern_recognizer.train(patterns)
    
    # Test with a new pattern (noisy version of pattern1)
    test_pattern = pattern1.copy()
    test_pattern += np.random.normal(0, 0.2, test_pattern.shape)
    
    print("\nTesting pattern recognition...")
    matched_pattern, score = pattern_recognizer.recognize(test_pattern)
    print(f"Test pattern matched with: {matched_pattern}, similarity score: {score:.3f}")
    
    # Test with another pattern
    test_pattern2 = pattern2.copy()
    test_pattern2 += np.random.normal(0, 0.2, test_pattern2.shape)
    
    matched_pattern, score = pattern_recognizer.recognize(test_pattern2)
    print(f"Test pattern matched with: {matched_pattern}, similarity score: {score:.3f}")
    
    print("\nThis spatiotemporal computing model demonstrates how:")
    print("1. Information can be encoded in spatial wave patterns")
    print("2. Computation emerges from wave interactions over time")
    print("3. Different diffusion constraints create processing dynamics")
    print("4. Pattern recognition can be performed through spatiotemporal evolution")
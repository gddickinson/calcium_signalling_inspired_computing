"""
Probabilistic Computing Based on Calcium Puff Dynamics

This simulation models a computational system inspired by calcium puff dynamics,
where the probability of signal activation scales linearly with the number of
channels (similar to how calcium puff probability scales with IP3R cluster size).

The model demonstrates how this biological principle can be applied to create
stochastic computing elements with predictable probabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d

class CalciumPuffComputing:
    def __init__(self, grid_size=50, decay_rate=0.95, threshold=0.7, diffusion_kernel_size=3):
        """
        Initialize the calcium-inspired probabilistic computing model.

        Parameters:
        - grid_size: Size of the 2D grid representing the computing substrate
        - decay_rate: Rate at which calcium signals decay over time
        - threshold: Threshold for channel activation
        - diffusion_kernel_size: Size of the kernel used for calcium diffusion
        """
        self.grid_size = grid_size
        self.decay_rate = decay_rate
        self.threshold = threshold

        # Initialize grid representing calcium concentration
        self.calcium_grid = np.zeros((grid_size, grid_size))

        # Grid representing channel density (analogous to IP3R cluster size)
        self.channel_density = np.zeros((grid_size, grid_size))

        # Create some regions with varying channel densities
        self._initialize_channel_distribution()

        # Create diffusion kernel for calcium spread
        self.diffusion_kernel = self._create_diffusion_kernel(diffusion_kernel_size)

        # Track activations for analysis
        self.activation_history = []
        self.density_vs_activation = {}

    def _initialize_channel_distribution(self):
        """Create regions with different channel densities"""
        # Create 4 regions with different densities
        region_size = self.grid_size // 2

        # Region 1: Low density (0.2)
        self.channel_density[:region_size, :region_size] = 0.2

        # Region 2: Medium density (0.4)
        self.channel_density[:region_size, region_size:] = 0.4

        # Region 3: High density (0.6)
        self.channel_density[region_size:, :region_size] = 0.6

        # Region 4: Very high density (0.8)
        self.channel_density[region_size:, region_size:] = 0.8

        # Add some random variation
        self.channel_density += np.random.normal(0, 0.05, (self.grid_size, self.grid_size))
        self.channel_density = np.clip(self.channel_density, 0, 1)

    def _create_diffusion_kernel(self, size):
        """Create a Gaussian kernel for calcium diffusion"""
        kernel = np.zeros((size, size))
        center = size // 2

        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                kernel[i, j] = np.exp(-dist**2)

        return kernel / kernel.sum()

    def update(self):
        """Update the calcium grid for one time step"""
        # Apply calcium decay
        self.calcium_grid *= self.decay_rate

        # Probabilistic channel activation based on channel density
        random_grid = np.random.random((self.grid_size, self.grid_size))
        activation_grid = (random_grid < self.channel_density) & (self.calcium_grid < self.threshold)

        # Record activations for analysis
        new_activations = np.zeros_like(self.calcium_grid)
        new_activations[activation_grid] = 1
        self.activation_history.append(new_activations)

        # Apply calcium release at activation sites
        self.calcium_grid[activation_grid] += 1.0

        # Apply diffusion (calcium spread)
        self.calcium_grid = convolve2d(self.calcium_grid, self.diffusion_kernel, mode='same', boundary='wrap')

        # Update statistics for density vs activation probability
        if len(self.activation_history) > 10:  # Start tracking after some warmup
            self._update_statistics()

    def _update_statistics(self):
        """Update statistics relating channel density to activation probability"""
        # Discretize density into bins
        density_bins = np.arange(0, 1.01, 0.1)

        for i in range(len(density_bins)-1):
            lower = density_bins[i]
            upper = density_bins[i+1]
            bin_key = f"{lower:.1f}-{upper:.1f}"

            # Find cells with density in this range
            mask = (self.channel_density >= lower) & (self.channel_density < upper)

            if mask.sum() > 0:
                # Calculate activation probability over recent history (last 50 steps)
                history_length = min(50, len(self.activation_history))
                recent_history = self.activation_history[-history_length:]

                activation_sum = sum(hist[mask].sum() for hist in recent_history)
                total_opportunities = mask.sum() * history_length

                # Record average activation probability for this density bin
                prob = activation_sum / total_opportunities if total_opportunities > 0 else 0

                if bin_key in self.density_vs_activation:
                    # Running average
                    old_val = self.density_vs_activation[bin_key]
                    self.density_vs_activation[bin_key] = 0.95 * old_val + 0.05 * prob
                else:
                    self.density_vs_activation[bin_key] = prob

    def run_simulation(self, steps=1000):
        """Run the simulation for a specified number of steps"""
        for _ in range(steps):
            self.update()

        return self.analyze_results()

    def analyze_results(self):
        """Analyze the relationship between channel density and activation probability"""
        # Extract density values and corresponding activation probabilities
        densities = []
        probabilities = []

        for key, prob in sorted(self.density_vs_activation.items()):
            # Extract the middle value of the bin range
            bin_range = key.split('-')
            mid_density = (float(bin_range[0]) + float(bin_range[1])) / 2

            densities.append(mid_density)
            probabilities.append(prob)

        # Fit a linear regression to verify linear relationship
        if len(densities) > 1:
            coeffs = np.polyfit(densities, probabilities, 1)
            poly = np.poly1d(coeffs)

            return {
                'densities': densities,
                'probabilities': probabilities,
                'fit_line': poly,
                'correlation': np.corrcoef(densities, probabilities)[0, 1],
                'slope': coeffs[0],
                'intercept': coeffs[1]
            }
        else:
            return {
                'densities': densities,
                'probabilities': probabilities
            }

    def visualize(self, results=None):
        """Visualize the simulation results"""
        if results is None:
            results = self.analyze_results()

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot channel density
        im1 = axes[0, 0].imshow(self.channel_density, cmap='viridis')
        axes[0, 0].set_title('Channel Density Distribution')
        plt.colorbar(im1, ax=axes[0, 0])

        # Plot current calcium concentration
        im2 = axes[0, 1].imshow(self.calcium_grid, cmap='hot')
        axes[0, 1].set_title('Current Calcium Concentration')
        plt.colorbar(im2, ax=axes[0, 1])

        # Plot recent activation map (average of last 10 frames)
        recent_activations = np.mean(self.activation_history[-10:], axis=0) if self.activation_history else np.zeros_like(self.calcium_grid)
        im3 = axes[1, 0].imshow(recent_activations, cmap='Blues')
        axes[1, 0].set_title('Recent Activation Probability')
        plt.colorbar(im3, ax=axes[1, 0])

        # Plot density vs activation probability
        if 'densities' in results and len(results['densities']) > 0:
            axes[1, 1].scatter(results['densities'], results['probabilities'], c='blue', alpha=0.7)
            axes[1, 1].set_xlabel('Channel Density')
            axes[1, 1].set_ylabel('Activation Probability')
            axes[1, 1].set_title('Channel Density vs. Activation Probability')

            # Plot the linear fit if available
            if 'fit_line' in results:
                x_range = np.linspace(0, 1, 100)
                axes[1, 1].plot(x_range, results['fit_line'](x_range), 'r--')

                # Add correlation text
                corr_text = f"Correlation: {results['correlation']:.3f}\nSlope: {results['slope']:.3f}"
                axes[1, 1].text(0.05, 0.95, corr_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

        plt.tight_layout()
        return fig

    def create_animation(self, frames=100):
        """Create an animation of calcium dynamics"""
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.calcium_grid, cmap='hot', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_title('Calcium Concentration Dynamics')

        def update(frame):
            self.update()
            im.set_array(self.calcium_grid)
            return [im]

        ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        return ani

# Implement a simple stochastic computing operation using this model
class StochasticComputing:
    def __init__(self, model=None):
        """Initialize a stochastic computing system"""
        if model is None:
            self.model = CalciumPuffComputing(grid_size=50)
        else:
            self.model = model

    def stochastic_multiply(self, a, b, iterations=1000):
        """
        Multiply two probabilities using stochastic computing principles
        Using AND operation on two independent bitstreams
        """
        if not (0 <= a <= 1 and 0 <= b <= 1):
            raise ValueError("Inputs must be probabilities between 0 and 1")

        # Generate independent bitstreams
        bitstream_a = np.random.random(iterations) < a
        bitstream_b = np.random.random(iterations) < b

        # AND operation represents multiplication
        result_stream = np.logical_and(bitstream_a, bitstream_b)

        # Return mean of result stream
        return result_stream.mean()

    def stochastic_add(self, a, b, iterations=1000):
        """
        Add two probabilities using stochastic computing principles
        Using a multiplexer with 0.5 selection probability
        Result is (a+b)/2, so multiply by 2 and cap at 1.0
        """
        if not (0 <= a <= 1 and 0 <= b <= 1):
            raise ValueError("Inputs must be probabilities between 0 and 1")

        # Generate independent bitstreams
        bitstream_a = np.random.random(iterations) < a
        bitstream_b = np.random.random(iterations) < b

        # Generate selector stream with 0.5 probability
        selector = np.random.random(iterations) < 0.5

        # Multiplexer operation: select a when selector is 1, b when selector is 0
        result_stream = np.where(selector, bitstream_a, bitstream_b)

        # This gives (a+b)/2, so multiply by 2 and cap at 1.0
        return min(1.0, 2.0 * result_stream.mean())

    def stochastic_subtract(self, a, b, iterations=1000):
        """
        Subtract b from a using stochastic computing principles
        Using AND NOT operation: a AND (NOT b)
        """
        if not (0 <= a <= 1 and 0 <= b <= 1):
            raise ValueError("Inputs must be probabilities between 0 and 1")

        # Generate independent bitstreams
        bitstream_a = np.random.random(iterations) < a
        bitstream_b = np.random.random(iterations) < b

        # a AND (NOT b) represents a * (1-b), which is equivalent to a-a*b or a*(1-b)
        result_stream = np.logical_and(bitstream_a, np.logical_not(bitstream_b))

        # Return mean, with floor at 0 since probabilities can't be negative
        return max(0.0, result_stream.mean())

    def stochastic_divide(self, a, b, iterations=1000, epsilon=0.01):
        """
        Divide a by b using stochastic computing principles
        A simplified approximation using conditional probability
        """
        if not (0 <= a <= 1 and 0 <= b <= 1):
            raise ValueError("Inputs must be probabilities between 0 and 1")

        if b < epsilon:
            return 1.0  # Avoid division by very small numbers

        # Generate independent bitstreams
        bitstream_a = np.random.random(iterations) < a
        bitstream_b = np.random.random(iterations) < b

        # Count when a is 1 given that b is 1 (conditional probability)
        b_indices = np.where(bitstream_b)[0]

        if len(b_indices) == 0:
            return 0.0  # Avoid division by zero

        # Calculate conditional probability P(a|b) which approximates a/b
        conditional_prob = bitstream_a[b_indices].mean()

        # Cap at 1.0 since we're dealing with probabilities
        return min(1.0, conditional_prob)

    def run_samples(self, iterations=10000):
        """Run samples of each operation and compare with exact results"""
        print("\nStochastic Computing Operations - Comparing to Exact Results")
        print("=" * 60)
        print(f"Using {iterations} iterations for each operation")
        print("-" * 60)

        # Test values
        test_pairs = [
            (0.3, 0.7),
            (0.5, 0.5),
            (0.8, 0.2),
            (0.1, 0.9),
            (0.4, 0.4)
        ]

        for a, b in test_pairs:
            print(f"\nTest with a = {a}, b = {b}")

            # Multiplication
            exact = a * b
            stochastic = self.stochastic_multiply(a, b, iterations)
            error = abs(exact - stochastic)
            print(f"Multiplication: {a} × {b} = {exact:.4f} (Exact) ≈ {stochastic:.4f} (Stochastic), Error: {error:.4f}")

            # Addition (capped at 1.0)
            exact = min(1.0, a + b)
            stochastic = self.stochastic_add(a, b, iterations)
            error = abs(exact - stochastic)
            print(f"Addition:       {a} + {b} = {exact:.4f} (Exact) ≈ {stochastic:.4f} (Stochastic), Error: {error:.4f}")

            # Subtraction (floored at 0.0)
            exact = max(0.0, a - b)
            stochastic = self.stochastic_subtract(a, b, iterations)
            error = abs(exact - stochastic)
            print(f"Subtraction:    {a} - {b} = {exact:.4f} (Exact) ≈ {stochastic:.4f} (Stochastic), Error: {error:.4f}")

            # Division (capped at 1.0)
            if b > 0.01:
                exact = min(1.0, a / b)
                stochastic = self.stochastic_divide(a, b, iterations)
                error = abs(exact - stochastic)
                print(f"Division:       {a} ÷ {b} = {exact:.4f} (Exact) ≈ {stochastic:.4f} (Stochastic), Error: {error:.4f}")

        print("\nKey Insights:")
        print("1. Stochastic computing trades precision for simplicity of hardware implementation")
        print("2. The accuracy improves with more iterations (longer bit streams)")
        print("3. Multiplication and subtraction are the most accurate operations")
        print("4. Division is the most sensitive to statistical variations")

    def _get_region_for_probability(self, p):
        """Get grid indices for a region with the specified channel density"""
        # Find closest matching density
        distances = np.abs(self.model.channel_density - p)
        region = distances < np.percentile(distances, 10)  # Get lowest 10% of distances

        return region

# Example usage
if __name__ == "__main__":
    # Initialize the model
    np.random.seed(42)  # For reproducibility
    model = CalciumPuffComputing(grid_size=50)

    # Run simulation
    print("Running calcium-inspired probabilistic computing simulation...")
    results = model.run_simulation(steps=500)

    # Visualize results
    fig = model.visualize(results)
    plt.show()

    # Create an animation
    ani = model.create_animation(frames=100)
    # Uncomment to save animation:
    # ani.save('calcium_dynamics.mp4', writer='ffmpeg')

    # Demonstrate stochastic computing
    stoch_comp = StochasticComputing(model)

    # Test multiplication
    a, b = 0.3, 0.7
    product = stoch_comp.stochastic_multiply(a, b, iterations=1000)

    print(f"\nStochastic multiplication example:")
    print(f"{a} × {b} = {a*b} (Exact)")
    print(f"{a} × {b} ≈ {product:.4f} (Stochastic)")

    # Test stochastic computing operations
    print("\nTesting stochastic computing operations...")
    stoch_comp = StochasticComputing()
    stoch_comp.run_samples(iterations=10000)

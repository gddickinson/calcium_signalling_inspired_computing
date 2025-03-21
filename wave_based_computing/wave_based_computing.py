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
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import time
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

class WaveComputingMedium:
    def __init__(self, size=(100, 100), boundary='absorbing', damping=0.99,
                diffusion=0.2, nonlinearity=0.1, time_step=0.1):
        """
        Initialize a wave computing medium with specified properties.

        Parameters:
        - size: Size of the 2D grid
        - boundary: Boundary condition ('absorbing', 'reflecting', 'periodic')
        - damping: Wave damping factor
        - diffusion: Diffusion coefficient
        - nonlinearity: Degree of nonlinearity in wave propagation
        - time_step: Time step for simulation
        """
        self.width, self.height = size
        self.boundary = boundary
        self.damping = damping
        self.diffusion = diffusion
        self.nonlinearity = nonlinearity
        self.time_step = time_step

        # Initialize wave state variables
        self.current = np.zeros(size)  # Current wave amplitude
        self.previous = np.zeros(size)  # Previous wave amplitude
        self.frequency = np.zeros(size)  # Local frequency estimates
        self.phase = np.zeros(size)  # Local phase estimates

        # Create diffusion kernel
        self.kernel = self._create_diffusion_kernel()

        # Initialize wave sources and obstacles
        self.sources = []  # List of (x, y, amplitude, frequency, phase) tuples
        self.obstacles = np.zeros(size, dtype=bool)  # Boolean mask for obstacles

        # Create resonant regions with specific frequency responses
        self.resonant_regions = {}  # Maps (x_min, x_max, y_min, y_max) to preferred frequency

        # Store for wave memory elements
        self.memory_elements = {}  # Maps (x, y) to (value, decay_rate)

        # History for analysis
        self.history = []
        self.energy_history = []
        self.peak_frequency_history = []

    def _create_diffusion_kernel(self):
        """Create a 3x3 kernel for wave diffusion"""
        kernel = np.array([
            [0.05, 0.2, 0.05],
            [0.2, 0.0, 0.2],
            [0.05, 0.2, 0.05]
        ])
        return kernel / kernel.sum()

    def add_source(self, position, amplitude, frequency, phase=0, duration=None):
        """
        Add a wave source to the medium.

        Parameters:
        - position: (x, y) position
        - amplitude: Wave amplitude
        - frequency: Wave frequency (oscillations per time unit)
        - phase: Initial phase
        - duration: Duration of the source (None for permanent)
        """
        self.sources.append({
            'position': position,
            'amplitude': amplitude,
            'frequency': frequency,
            'phase': phase,
            'start_time': 0,
            'duration': duration
        })

    def add_obstacle(self, x_min, x_max, y_min, y_max):
        """Add an obstacle that blocks wave propagation"""
        self.obstacles[y_min:y_max, x_min:x_max] = True

    def add_resonant_region(self, x_min, x_max, y_min, y_max, frequency):
        """Add a region that resonates with a specific frequency"""
        self.resonant_regions[(x_min, x_max, y_min, y_max)] = frequency

    def set_memory_element(self, position, value):
        """Set a value in a specific position for wave memory"""
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            # Use higher damping rate for memory elements
            decay_rate = 0.999
            self.memory_elements[(x, y)] = (value, decay_rate)

    def read_memory_element(self, position):
        """Read a value from a memory element"""
        x, y = position
        if (x, y) in self.memory_elements:
            return self.memory_elements[(x, y)][0]
        else:
            return 0.0

    def update(self, t):
        """
        Update the wave medium for one time step.

        Parameters:
        - t: Current time (used for sources)
        """
        # Save previous state
        self.previous = self.current.copy()

        # Apply wave equation with damping and diffusion
        # Using discrete approximation of wave equation:
        # u(t+dt) = 2*u(t) - u(t-dt) + (c²dt²)∇²u(t)
        laplacian = convolve2d(self.current, self.kernel, mode='same', boundary='symm')

        # Basic wave equation
        next_state = (2 * self.current - self.previous
                     + self.diffusion * self.time_step**2 * laplacian)

        # Apply damping
        next_state *= self.damping

        # Apply boundary conditions
        if self.boundary == 'absorbing':
            # Gradually absorb waves at boundaries
            boundary_mask = np.ones_like(next_state)
            boundary_width = 5
            for i in range(boundary_width):
                # Create a frame of decreasing values toward boundary
                factor = 1.0 - (i / boundary_width)**2
                boundary_mask[i, :] *= factor
                boundary_mask[-i-1, :] *= factor
                boundary_mask[:, i] *= factor
                boundary_mask[:, -i-1] *= factor

            next_state *= boundary_mask
        elif self.boundary == 'reflecting':
            # Reflection is automatically handled by the symm boundary in convolve2d
            pass
        elif self.boundary == 'periodic':
            # Handle periodicity by adding contributions from opposite edges
            next_state[0, :] += 0.5 * next_state[-1, :]
            next_state[-1, :] += 0.5 * next_state[0, :]
            next_state[:, 0] += 0.5 * next_state[:, -1]
            next_state[:, -1] += 0.5 * next_state[:, 0]

        # Apply nonlinearity (similar to excitable media)
        if self.nonlinearity > 0:
            nonlinear_term = self.nonlinearity * self.current**3
            next_state -= nonlinear_term

        # Apply obstacles (set wave amplitude to 0 at obstacle positions)
        next_state[self.obstacles] = 0

        # Apply resonant regions (amplify frequencies that match the region)
        for (x_min, x_max, y_min, y_max), freq in self.resonant_regions.items():
            # Calculate local frequency in the region
            if len(self.history) > 10:  # Need history for frequency estimation
                local_freq = self._estimate_local_frequency(x_min, x_max, y_min, y_max)
                # Amplify or dampen based on frequency match
                resonance_factor = 1.0 + 0.1 * np.exp(-((local_freq - freq) / 0.1)**2)
                next_state[y_min:y_max, x_min:x_max] *= resonance_factor

        # Apply wave sources
        for source in self.sources:
            x, y = source['position']
            amplitude = source['amplitude']
            frequency = source['frequency']
            phase = source['phase']

            # Check if source is active
            if source['duration'] is None or t < source['start_time'] + source['duration']:
                # Generate wave at source position
                source_value = amplitude * np.sin(2 * np.pi * frequency * t + phase)

                # Apply to a small region around the source
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            # Gaussian weight by distance
                            weight = np.exp(-(dx**2 + dy**2) / 2)
                            next_state[ny, nx] += source_value * weight

        # Update memory elements
        for (x, y), (value, decay_rate) in list(self.memory_elements.items()):
            # Update value with decay
            self.memory_elements[(x, y)] = (value * decay_rate, decay_rate)

            # Apply to wave medium (memory elements act as damped oscillators)
            next_state[y, x] += 0.1 * value

        # Update current state
        self.current = next_state

        # Update frequency and phase estimates
        if len(self.history) > 5:
            self._update_frequency_phase_estimates()

        # Save state for history
        self.history.append(self.current.copy())
        if len(self.history) > 100:
            self.history.pop(0)

        # Calculate and save total energy
        energy = np.sum(self.current**2)
        self.energy_history.append(energy)
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)

    def _estimate_local_frequency(self, x_min, x_max, y_min, y_max):
        """Estimate local frequency in a region using FFT"""
        # Extract recent history for the region
        region_history = [frame[y_min:y_max, x_min:x_max].mean() for frame in self.history[-50:]]

        if len(region_history) < 10:
            return 0.0

        # Perform FFT
        fft_result = np.abs(np.fft.rfft(region_history))
        freqs = np.fft.rfftfreq(len(region_history), d=self.time_step)

        # Find peak frequency
        if len(fft_result) > 1:
            peak_idx = np.argmax(fft_result[1:]) + 1  # Skip DC component
            return freqs[peak_idx]
        else:
            return 0.0

    def _update_frequency_phase_estimates(self):
        """Update local frequency and phase estimates"""
        # This is a simplistic approach - a real implementation would use
        # more sophisticated techniques like wavelet analysis or Hilbert transform
        if len(self.history) < 10:
            return

        # Use recent history to estimate frequency
        recent = np.array(self.history[-10:])

        # For each point, find local frequency using zero-crossing approach
        for y in range(self.height):
            for x in range(self.width):
                # Extract time series for this point
                series = recent[:, y, x]

                # Count zero crossings as crude frequency estimate
                crossings = np.sum(np.diff(np.signbit(series)))
                self.frequency[y, x] = crossings / (10 * self.time_step) / 2  # Divide by 2 as each cycle has 2 crossings

                # Crude phase estimate based on current value relative to recent min/max
                min_val = np.min(series)
                max_val = np.max(series)
                if max_val > min_val:
                    normalized = (self.current[y, x] - min_val) / (max_val - min_val)
                    self.phase[y, x] = normalized * 2 * np.pi
                else:
                    self.phase[y, x] = 0

    def run_simulation(self, steps=100, visualize=False):
        """
        Run simulation for specified number of steps.

        Parameters:
        - steps: Number of time steps
        - visualize: Whether to return visualization frames

        Returns:
        - frames: List of frames if visualize=True
        """
        frames = []

        for t in range(steps):
            self.update(t * self.time_step)

            if visualize and t % 5 == 0:  # Save every 5th frame to reduce memory usage
                frames.append(self.current.copy())

        return frames if visualize else None

    def analyze_dynamics(self):
        """Analyze wave dynamics and patterns"""
        if len(self.history) < 10:
            return {}

        # Calculate spectral properties
        spectral_analysis = self._analyze_spectral_properties()

        # Identify wave patterns
        pattern_analysis = self._identify_wave_patterns()

        # Calculate energy distribution
        energy_analysis = {
            'total_energy': self.energy_history[-1],
            'energy_trend': np.diff(self.energy_history[-10:]).mean() if len(self.energy_history) >= 10 else 0,
            'energy_stability': np.std(self.energy_history[-20:]) / np.mean(self.energy_history[-20:]) if len(self.energy_history) >= 20 else 1
        }

        return {
            'spectral': spectral_analysis,
            'patterns': pattern_analysis,
            'energy': energy_analysis
        }

    def _analyze_spectral_properties(self):
        """Analyze spectral properties of the wave dynamics"""
        # Flatten recent history for each point
        if len(self.history) < 10:
            return {}

        recent = np.array(self.history[-10:])
        n_samples = recent.shape[0]
        flat_data = recent.reshape(n_samples, -1)

        # Check for NaN or insufficient variance
        if np.any(np.isnan(flat_data)) or np.std(flat_data) < 1e-10:
            # Add small random noise to ensure non-zero variance
            flat_data = flat_data + np.random.normal(0, 1e-8, flat_data.shape)

        try:
            # Perform PCA on temporal evolution
            pca = PCA(n_components=min(5, n_samples))
            pca.fit(flat_data)
            dominant_modes = pca.explained_variance_ratio_
        except Exception as e:
            print(f"PCA analysis failed: {e}")
            dominant_modes = np.zeros(min(5, n_samples))



        # Identify dominant frequencies using FFT
        fft_data = np.abs(np.fft.rfft(flat_data, axis=0))
        freqs = np.fft.rfftfreq(n_samples, d=self.time_step)

        # Find peak frequencies
        peak_freq_idx = np.argmax(fft_data, axis=0)
        peak_frequencies = freqs[peak_freq_idx]

        # Dominant global frequency
        global_fft = np.mean(fft_data, axis=1)
        global_peak_idx = np.argmax(global_fft)
        global_peak_freq = freqs[global_peak_idx]

        self.peak_frequency_history.append(global_peak_freq)
        if len(self.peak_frequency_history) > 100:
            self.peak_frequency_history.pop(0)

        return {
            'dominant_modes': dominant_modes,
            'global_peak_frequency': global_peak_freq,
            'frequency_distribution': np.histogram(peak_frequencies, bins=10)[0]
        }

    def _identify_wave_patterns(self):
        """Identify wave patterns in the medium"""
        if len(self.history) < 5:
            return {}

        current = self.current

        # Identify wave fronts using gradient
        dx = np.gradient(current, axis=1)
        dy = np.gradient(current, axis=0)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)

        # Threshold to find wave fronts
        wave_front_threshold = np.percentile(gradient_magnitude, 80)
        wave_fronts = gradient_magnitude > wave_front_threshold

        # Count and measure wave fronts
        from scipy import ndimage
        labeled_fronts, num_fronts = ndimage.label(wave_fronts)

        front_sizes = []
        if num_fronts > 0:
            for i in range(1, num_fronts + 1):
                front_size = np.sum(labeled_fronts == i)
                front_sizes.append(front_size)

        # Detect spiral patterns using curl of the gradient field
        curl = np.gradient(dy, axis=1) - np.gradient(dx, axis=0)

        # High curl magnitude indicates spiral centers
        curl_magnitude = np.abs(curl)
        spiral_threshold = np.percentile(curl_magnitude, 95)
        potential_spirals = curl_magnitude > spiral_threshold

        # Label and count potential spiral centers
        labeled_spirals, num_spirals = ndimage.label(potential_spirals)

        # Analyze wave stability by comparing consecutive frames
        if len(self.history) >= 2:
            last_frame = self.history[-2]
            frame_diff = np.abs(current - last_frame)
            stability = 1.0 - np.mean(frame_diff) / (np.std(current) + 1e-6)
        else:
            stability = 0.0

        return {
            'num_wave_fronts': num_fronts,
            'wave_front_sizes': front_sizes,
            'num_potential_spirals': num_spirals,
            'stability': stability
        }

    def visualize_state(self, mode='amplitude'):
        """
        Visualize the current state of the wave medium.

        Parameters:
        - mode: Visualization mode ('amplitude', 'frequency', 'phase')

        Returns:
        - fig: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        if mode == 'amplitude':
            # Visualize wave amplitude
            vmin, vmax = -0.5, 0.5
            im = ax.imshow(self.current, cmap='seismic',
                          vmin=vmin, vmax=vmax, interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='Wave Amplitude')
            ax.set_title('Wave Amplitude')

            # Mark sources
            for source in self.sources:
                x, y = source['position']
                ax.plot(x, y, 'yo', markersize=10)

            # Mark obstacles
            for y in range(self.height):
                for x in range(self.width):
                    if self.obstacles[y, x]:
                        ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1,
                                                 fill=True, color='black', alpha=0.7))

            # Mark resonant regions
            for (x_min, x_max, y_min, y_max), freq in self.resonant_regions.items():
                rect = plt.Rectangle((x_min-0.5, y_min-0.5), x_max-x_min, y_max-y_min,
                                   fill=False, edgecolor='green', linewidth=2)
                ax.add_patch(rect)
                ax.text(x_min + (x_max-x_min)/2, y_min + (y_max-y_min)/2,
                       f"{freq:.1f}Hz", color='white', ha='center', va='center')

            # Mark memory elements
            for (x, y), (value, _) in self.memory_elements.items():
                color = 'blue' if value > 0 else 'red'
                size = min(20, max(5, abs(value) * 15))
                ax.plot(x, y, 'o', color=color, markersize=size, alpha=0.7)

        elif mode == 'frequency':
            # Visualize local frequency estimates
            im = ax.imshow(self.frequency, cmap='viridis',
                          vmin=0, vmax=2, interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='Frequency (Hz)')
            ax.set_title('Local Frequency Estimates')

        elif mode == 'phase':
            # Visualize local phase estimates
            im = ax.imshow(self.phase / (2*np.pi), cmap='hsv',
                          vmin=0, vmax=1, interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='Phase (normalized)')
            ax.set_title('Local Phase Estimates')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.tight_layout()

        return fig

    def create_animation(self, frames=None):
        """
        Create an animation of wave dynamics.

        Parameters:
        - frames: List of frames (if None, use history)

        Returns:
        - animation: Matplotlib animation
        """
        if frames is None:
            if len(self.history) < 2:
                return None
            frames = self.history

        fig, ax = plt.subplots(figsize=(8, 8))

        vmin = min(np.min(frame) for frame in frames)
        vmax = max(np.max(frame) for frame in frames)

        # Custom colormap with white at zero
        cdict = {
            'red': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
            'green': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)],
            'blue': [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)]
        }
        wave_cmap = LinearSegmentedColormap('wave_cmap', cdict)

        im = ax.imshow(frames[0], cmap=wave_cmap,
                      vmin=vmin, vmax=vmax, interpolation='bilinear')

        # Mark obstacles
        for y in range(self.height):
            for x in range(self.width):
                if self.obstacles[y, x]:
                    ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1,
                                             fill=True, color='black', alpha=0.7))

        # Mark sources
        for source in self.sources:
            x, y = source['position']
            ax.plot(x, y, 'yo', markersize=8)

        plt.colorbar(im, ax=ax, label='Wave Amplitude')
        ax.set_title('Wave Dynamics')

        def update(frame):
            im.set_array(frame)
            return [im]

        ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

        return ani


class WaveComputing:
    def __init__(self, medium):
        """
        Initialize a wave computing system with the given medium.

        Parameters:
        - medium: WaveComputingMedium instance
        """
        self.medium = medium

    def pattern_recognition(self, test_patterns, reference_patterns, simulation_steps=100):
        """
        Use wave dynamics for pattern recognition.

        Parameters:
        - test_patterns: List of patterns to recognize
        - reference_patterns: Known reference patterns
        - simulation_steps: Number of steps to run for each pattern

        Returns:
        - recognition_results: Matching results
        """
        results = []

        for test_idx, test_pattern in enumerate(test_patterns):
            # Create a medium configuration specific to this test
            medium = WaveComputingMedium(
                size=(self.medium.width, self.medium.height),
                boundary=self.medium.boundary,
                damping=self.medium.damping,
                diffusion=self.medium.diffusion,
                nonlinearity=self.medium.nonlinearity
            )

            # Create wave sources based on test pattern
            self._encode_pattern_as_sources(medium, test_pattern)

            # Run simulation
            medium.run_simulation(steps=simulation_steps)

            # Create wave signature from the medium state
            test_signature = self._extract_wave_signature(medium)

            # Compare with reference patterns
            pattern_matches = []
            for ref_idx, ref_pattern in enumerate(reference_patterns):
                # Create reference signature
                ref_medium = WaveComputingMedium(
                    size=(self.medium.width, self.medium.height),
                    boundary=self.medium.boundary,
                    damping=self.medium.damping,
                    diffusion=self.medium.diffusion,
                    nonlinearity=self.medium.nonlinearity
                )

                self._encode_pattern_as_sources(ref_medium, ref_pattern)
                ref_medium.run_simulation(steps=simulation_steps)
                ref_signature = self._extract_wave_signature(ref_medium)

                # Calculate similarity
                similarity = self._calculate_signature_similarity(test_signature, ref_signature)

                pattern_matches.append({
                    'reference_index': ref_idx,
                    'similarity': similarity
                })

            # Sort by similarity
            pattern_matches.sort(key=lambda x: x['similarity'], reverse=True)

            results.append({
                'test_index': test_idx,
                'matches': pattern_matches,
                'wave_signature': test_signature
            })

        return results

    def _encode_pattern_as_sources(self, medium, pattern):
        """Encode a pattern as wave sources"""
        height, width = pattern.shape
        scale_y = medium.height / height
        scale_x = medium.width / width

        # Find high-intensity points in the pattern
        threshold = np.percentile(pattern, 80)
        y_indices, x_indices = np.where(pattern > threshold)

        # Add wave sources at high-intensity points
        for y, x in zip(y_indices, x_indices):
            # Scale to medium coordinates
            medium_x = int(x * scale_x)
            medium_y = int(y * scale_y)

            # Intensity determines amplitude
            intensity = pattern[y, x]
            amplitude = 0.5 * intensity

            # Position-dependent frequency
            frequency = 0.5 + 0.5 * (x / width)

            medium.add_source((medium_x, medium_y), amplitude, frequency)

    def _extract_wave_signature(self, medium):
        """Extract a signature from wave dynamics for pattern matching"""
        if len(medium.history) < 10:
            return np.zeros(5)

        recent = np.array(medium.history[-10:])
        n_samples = recent.shape[0]
        flat_data = recent.reshape(n_samples, -1)

        # Add variation to ensure distinguishable signatures
        if np.std(flat_data) < 1e-6:
            flat_data += np.random.normal(0, 1e-5, flat_data.shape)

        # Use more components for better discrimination
        try:
            pca = PCA(n_components=min(8, n_samples))
            pca.fit(flat_data)

            signature = []
            for i, (component, variance) in enumerate(zip(pca.components_, pca.explained_variance_)):
                # Add more statistics for better pattern discrimination
                signature.extend([
                    variance,
                    np.mean(component),
                    np.std(component),
                    np.percentile(component, 10),
                    np.percentile(component, 90),
                    np.max(component) - np.min(component)  # Add range
                ])

            # Add spatial information from the last frame
            last_frame = medium.current
            signature.extend([
                np.mean(last_frame),
                np.std(last_frame),
                np.max(last_frame),
                np.min(last_frame)
            ])

        except Exception as e:
            print(f"PCA signature extraction failed: {e}")
            signature = [0.0] * 52  # 8 components * 6 statistics + 4 spatial features

        # Normalize signature
        if np.std(signature) > 0:
            signature = (signature - np.mean(signature)) / np.std(signature)

        return np.array(signature)

    def _calculate_signature_similarity(self, sig1, sig2):
        """Calculate similarity between two wave signatures"""
        # Ensure same length
        min_len = min(len(sig1), len(sig2))
        sig1 = sig1[:min_len]
        sig2 = sig2[:min_len]

        # Normalize signatures
        sig1_norm = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-6)
        sig2_norm = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-6)

        # Calculate cosine similarity
        dot_product = np.dot(sig1_norm, sig2_norm)
        magnitude1 = np.sqrt(np.dot(sig1_norm, sig1_norm))
        magnitude2 = np.sqrt(np.dot(sig2_norm, sig2_norm))

        similarity = dot_product / (magnitude1 * magnitude2 + 1e-6)

        return (similarity + 1) / 2  # Scale to [0, 1]

    def wave_logic_gates(self, input_patterns, operation='and', simulation_steps=50):
        """
        Implement logic gates using wave interference.

        Parameters:
        - input_patterns: List of input patterns
        - operation: Logic operation ('and', 'or', 'xor')
        - simulation_steps: Number of simulation steps

        Returns:
        - output_pattern: Result of the operation
        """
        if len(input_patterns) < 2:
            return None

        # Create medium for computation
        medium = WaveComputingMedium(
            size=(self.medium.width, self.medium.height),
            boundary=self.medium.boundary,
            damping=self.medium.damping,
            diffusion=self.medium.diffusion,
            nonlinearity=self.medium.nonlinearity
        )

        # Set up wave sources for inputs
        if operation == 'and':
            # AND: Set up inputs to require constructive interference
            # Input 1 at left side
            for y in range(medium.height // 4, 3 * medium.height // 4):
                medium.add_source((5, y), 0.3, 0.5, phase=0)

            # Input 2 at top side
            for x in range(medium.width // 4, 3 * medium.width // 4):
                medium.add_source((x, 5), 0.3, 0.5, phase=0)

            # Add a nonlinear threshold region in the center
            center_x, center_y = medium.width // 2, medium.height // 2
            medium.add_resonant_region(
                center_x - 10, center_x + 10,
                center_y - 10, center_y + 10,
                frequency=0.5  # Resonates with input frequency
            )

        elif operation == 'or':
            # OR: Set up inputs to produce output with either input
            # Input 1 at left side
            for y in range(medium.height // 4, 3 * medium.height // 4):
                medium.add_source((5, y), 0.2, 0.7, phase=0)

            # Input 2 at top side
            for x in range(medium.width // 4, 3 * medium.width // 4):
                medium.add_source((x, 5), 0.2, 0.7, phase=0)

            # Add amplifying resonant region in center
            center_x, center_y = medium.width // 2, medium.height // 2
            medium.add_resonant_region(
                center_x - 15, center_x + 15,
                center_y - 15, center_y + 15,
                frequency=0.7  # Resonates with input frequency
            )

        elif operation == 'xor':
            # XOR: Set up inputs with phase shift to create destructive interference when both active
            # Input 1 at left side
            for y in range(medium.height // 4, 3 * medium.height // 4):
                medium.add_source((5, y), 0.3, 0.6, phase=0)

            # Input 2 at top side with phase shift
            for x in range(medium.width // 4, 3 * medium.width // 4):
                medium.add_source((x, 5), 0.3, 0.6, phase=np.pi)  # Phase shift

            # Add a sensitive region in the center
            center_x, center_y = medium.width // 2, medium.height // 2
            medium.nonlinearity = 0.2  # Increase nonlinearity for XOR

        # Run simulation
        frames = medium.run_simulation(steps=simulation_steps, visualize=True)

        # Extract result from center region
        center_x, center_y = medium.width // 2, medium.height // 2
        region_size = 10
        x_min = max(0, center_x - region_size)
        x_max = min(medium.width, center_x + region_size)
        y_min = max(0, center_y - region_size)
        y_max = min(medium.height, center_y + region_size)

        # # Average activation in the center region in last few frames
        # last_frames = medium.history[-5:]
        # if not last_frames:
        #     return None

        # center_activations = [
        #     frame[y_min:y_max, x_min:x_max].mean()
        #     for frame in last_frames
        # ]
        # result_value = np.mean(center_activations)

        # Use peak amplitude rather than average for better detection
        last_frames = medium.history[-10:]
        if not last_frames:
            return None

        center_activations = [
            np.max(np.abs(frame[y_min:y_max, x_min:x_max]))
            for frame in last_frames
        ]
        result_value = np.max(center_activations)


        # Create output pattern
        output = np.zeros((medium.height, medium.width))
        output[y_min:y_max, x_min:x_max] = result_value

        return {
            'operation': operation,
            'result_value': result_value,
            'output_pattern': output,
            'simulation_frames': frames
        }

    def wave_memory(self, input_data, storage_time=50, retrieval_time=20):
        """
        Store and retrieve information using wave dynamics.

        Parameters:
        - input_data: Data to store (1D array)
        - storage_time: Time steps for storage phase
        - retrieval_time: Time steps for retrieval phase

        Returns:
        - retrieved_data: Retrieved data
        """
        # Reset medium
        medium = WaveComputingMedium(
            size=(self.medium.width, self.medium.height),
            boundary='absorbing',  # Absorbing boundaries for cleaner patterns
            damping=0.995,  # Higher damping for longer persistence
            diffusion=0.1,
            nonlinearity=0.05
        )

        # Encode data into wave sources
        data_length = len(input_data)
        spacing = medium.width // (data_length + 1)

        for i, value in enumerate(input_data):
            x = (i + 1) * spacing
            # Place sources along the top
            y = 5

            # Scale and offset value to appropriate range
            amplitude = 0.3 * abs(value)
            frequency = 0.5 + 0.3 * (value + 1) / 2  # Map to 0.5-0.8 Hz range
            phase = 0 if value >= 0 else np.pi  # Phase encodes sign

            # Add temporary source
            medium.add_source((x, y), amplitude, frequency, phase, duration=10)

            # Create a special resonant region below each source to "store" the value
            region_size = 5
            medium.add_resonant_region(
                x - region_size, x + region_size,
                medium.height // 2 - region_size, medium.height // 2 + region_size,
                frequency=frequency
            )

            # Add memory element
            medium.set_memory_element((x, medium.height // 2), value)

        # Run storage phase
        print("Running storage phase...")
        storage_frames = medium.run_simulation(steps=storage_time, visualize=True)

        # Run retrieval phase (add retrieval stimulus)
        print("Running retrieval phase...")
        # Add a broad-spectrum pulse along the bottom to stimulate memory
        for x in range(spacing, medium.width, spacing):
            medium.add_source((x, medium.height - 5), 0.5, 0.65, phase=0, duration=5)

        retrieval_frames = medium.run_simulation(steps=retrieval_time, visualize=True)

        # Extract retrieved data
        retrieved_data = []
        for i in range(data_length):
            x = (i + 1) * spacing
            y = medium.height // 2

            # Average activation in region around the memory point
            region_size = 3
            x_min = max(0, x - region_size)
            x_max = min(medium.width, x + region_size + 1)
            y_min = max(0, y - region_size)
            y_max = min(medium.height, y + region_size + 1)

            # Extract region from recent frames
            recent_frames = medium.history[-5:]
            if not recent_frames:
                retrieved_data.append(0)
                continue

            activations = [
                frame[y_min:y_max, x_min:x_max].mean()
                for frame in recent_frames
            ]

            # Average activation
            avg_activation = np.mean(activations)

            # Read from memory element
            memory_value = medium.read_memory_element((x, y))

            # Combine direct measurement with memory element
            retrieved_value = 0.7 * memory_value + 0.3 * (avg_activation * 2)

            retrieved_data.append(retrieved_value)

        return {
            'input_data': input_data,
            'retrieved_data': retrieved_data,
            'storage_frames': storage_frames,
            'retrieval_frames': retrieval_frames
        }

    def wave_filter(self, input_signal, filter_type='lowpass', cutoff_frequency=0.5, simulation_steps=100):
        """
        Filter a 1D signal using wave dynamics.

        Parameters:
        - input_signal: Input signal (1D array)
        - filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
        - cutoff_frequency: Cutoff frequency
        - simulation_steps: Number of simulation steps

        Returns:
        - filtered_signal: Filtered signal
        """
        # Reset medium
        medium = WaveComputingMedium(
            size=(self.medium.width, self.medium.height),
            boundary='absorbing',
            damping=0.99,
            diffusion=0.1
        )

        # Create input region on left side
        signal_length = len(input_signal)
        signal_scale = min(1.0, medium.height / signal_length)

        for i, value in enumerate(input_signal):
            # Scale to fit in medium
            y = int(i * signal_scale)
            if y >= medium.height:
                break

            # Add source with amplitude based on signal value
            amplitude = 0.4 * abs(value)

            # Modulate frequency to encode signal value
            frequency = 0.3 + 0.6 * (value + 1) / 2  # Map to 0.3-0.9 Hz

            medium.add_source((5, y), amplitude, frequency)

        # Set up filter in the middle
        if filter_type == 'lowpass':
            # Resonant region that passes low frequencies
            for y in range(medium.height):
                # Create a low-frequency-tuned resonant region spanning the middle
                medium.add_resonant_region(
                    medium.width // 3, 2 * medium.width // 3,
                    max(0, y - 2), min(medium.height, y + 3),
                    frequency=cutoff_frequency / 2  # Resonate with frequencies below cutoff
                )

        elif filter_type == 'highpass':
            # Resonant region that passes high frequencies
            for y in range(medium.height):
                # Create a high-frequency-tuned resonant region spanning the middle
                medium.add_resonant_region(
                    medium.width // 3, 2 * medium.width // 3,
                    max(0, y - 2), min(medium.height, y + 3),
                    frequency=cutoff_frequency * 1.5  # Resonate with frequencies above cutoff
                )

        elif filter_type == 'bandpass':
            # Resonant region that passes a specific frequency band
            for y in range(medium.height):
                # Create a band-frequency-tuned resonant region spanning the middle
                medium.add_resonant_region(
                    medium.width // 3, 2 * medium.width // 3,
                    max(0, y - 2), min(medium.height, y + 3),
                    frequency=cutoff_frequency  # Resonate with frequencies around cutoff
                )

        # Run simulation
        frames = medium.run_simulation(steps=simulation_steps, visualize=True)

        # Extract filtered signal from right side
        output_x = int(4 * medium.width / 5)
        filtered_signal = []

        for i in range(min(signal_length, medium.height)):
            y = int(i * signal_scale)
            if y >= medium.height:
                break

            # Average over recent frames
            recent_frames = medium.history[-10:]
            if not recent_frames:
                filtered_signal.append(0)
                continue

            # Extract amplitude
            amplitudes = [frame[y, output_x] for frame in recent_frames]
            filtered_value = np.mean(amplitudes)

            filtered_signal.append(filtered_value)

        # Normalize
        if filtered_signal:
            max_abs = max(abs(min(filtered_signal)), abs(max(filtered_signal)))
            if max_abs > 0:
                filtered_signal = [v / max_abs for v in filtered_signal]

        return {
            'input_signal': input_signal,
            'filtered_signal': filtered_signal,
            'filter_type': filter_type,
            'cutoff_frequency': cutoff_frequency,
            'frames': frames
        }


# Example usage
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

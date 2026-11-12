"""
Wave Computing Medium -- core simulation engine.

A 2D medium obeying the damped wave equation

    u_tt = c^2 * laplacian(u) - gamma * u_t - nonlinearity * u^3

with point sources, obstacles, resonant regions and memory elements. Information is
encoded in wave properties (amplitude, frequency, phase) and computation emerges
from wave interactions.

Numerics: leapfrog (second order in time) with a 9-point Laplacian on a grid of
spacing dx (cells). Each output step of length `time_step` is split into
`substeps` internal steps dt_sub chosen so that c * dt_sub / dx satisfies the CFL
condition of this stencil (with a safety factor), so the scheme is stable for any
wave speed. `damping` is the fraction of wave velocity kept per output step.
Sources and memory elements are "hard" (they set the field in their cells), the
absorbing boundary is a 5-cell sponge, resonant regions lose no energy while
their dominant frequency matches their tuning. (Before 2026-11-11 the update
overwrote the previous state before using it, so it was a first-order smoothing
rather than a wave equation; see SESSION_LOG.md.)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA


class WaveComputingMedium:
    # 9-point Laplacian: neighbour mean (kernel below) minus centre ~= LAPLACIAN_SCALE * dx^2 * laplacian(u)
    LAPLACIAN_SCALE = 0.3
    # Largest eigenvalue of -laplacian_h for dx = 1 (checkerboard mode: 1.6 / 0.3)
    LAMBDA_MAX = 1.6 / 0.3
    # Leapfrog is stable for c * dt * sqrt(LAMBDA_MAX) / dx <= 2; keep this fraction of the limit
    CFL_SAFETY = 0.9
    SPONGE_WIDTH = 5

    def __init__(self, size=(100, 100), boundary='absorbing', damping=0.99,
                wave_speed=10.0, nonlinearity=0.1, time_step=0.1, dx=1.0, diffusion=None):
        """
        Initialize a wave computing medium with specified properties.

        Parameters:
        - size: Size of the 2D grid as (width, height); arrays have shape (height, width)
          and are indexed [y, x]
        - boundary: 'absorbing' (5-cell sponge), 'reflecting' (mirror, zero normal
          gradient) or 'periodic' (wrap-around)
        - damping: fraction of the wave velocity kept per output step (1 = lossless)
        - wave_speed: wave speed c in grid cells per time unit
        - nonlinearity: coefficient of the cubic restoring term (-nonlinearity * u^3)
        - time_step: output time step (time units); split into CFL-stable substeps
        - dx: grid spacing (cells)
        - diffusion: deprecated alias for c^2 (if given, wave_speed = sqrt(diffusion))
        """
        self.width, self.height = size
        shape = (self.height, self.width)
        self.boundary = boundary
        self.damping = damping
        if diffusion is not None:
            wave_speed = float(np.sqrt(diffusion))
        self.wave_speed = float(wave_speed)
        self.diffusion = self.wave_speed ** 2       # c^2, kept for callers that read it
        self.nonlinearity = nonlinearity
        self.time_step = time_step
        self.dx = dx
        self.substeps, self.dt_sub = self._cfl_substeps()

        # Initialize wave state variables
        self.current = np.zeros(shape)
        self.previous = np.zeros(shape)
        self.frequency = np.zeros(shape)
        self.phase = np.zeros(shape)

        # Neighbour-weight kernel for the Laplacian, sponge mask for absorbing edges
        self.kernel = self._create_diffusion_kernel()
        self.sponge = self._sponge_mask(shape)

        # Initialize wave sources and obstacles
        self.sources = []
        self.obstacles = np.zeros(shape, dtype=bool)

        # Resonant regions with specific frequency responses
        self.resonant_regions = {}

        # Wave memory elements
        self.memory_elements = {}

        # History for analysis
        self.history = []
        self.energy_history = []
        self.peak_frequency_history = []

    def _cfl_substeps(self):
        """Number of internal steps per output step and their length, from the CFL condition."""
        if self.wave_speed <= 0:
            return 1, self.time_step
        dt_max = self.CFL_SAFETY * 2.0 * self.dx / (self.wave_speed * np.sqrt(self.LAMBDA_MAX))
        n = max(1, int(np.ceil(self.time_step / dt_max - 1e-12)))
        return n, self.time_step / n

    def _sponge_mask(self, shape):
        """Absorbing layer: factor exp(-(0.2 * (width - i))^2) in the i-th cell from each edge, 1 inside."""
        mask = np.ones(shape)
        bw = self.SPONGE_WIDTH
        for i in range(bw):
            f = np.exp(-(0.2 * (bw - i)) ** 2)
            for row in (i, shape[0] - 1 - i):
                if 0 <= row < shape[0]:
                    mask[row, :] = np.minimum(mask[row, :], f)
            for col in (i, shape[1] - 1 - i):
                if 0 <= col < shape[1]:
                    mask[:, col] = np.minimum(mask[:, col], f)
        return mask

    def laplacian(self, u):
        """9-point Laplacian (mirror edges, or wrap-around for periodic boundaries)."""
        bc = 'wrap' if self.boundary == 'periodic' else 'symm'
        mean = convolve2d(u, self.kernel, mode='same', boundary=bc)
        return (mean - u) / (self.LAPLACIAN_SCALE * self.dx ** 2)

    def wave_energy(self):
        """Discrete leapfrog energy 0.5*sum(((u - u_prev)/dt)^2) - 0.5*c^2*sum(u * L(u_prev)) (exactly conserved
        without damping, sources, nonlinearity or sponge on a periodic grid)."""
        v = (self.current - self.previous) / self.dt_sub
        return 0.5 * np.sum(v ** 2) - 0.5 * self.diffusion * np.sum(self.current * self.laplacian(self.previous))

    def _create_diffusion_kernel(self):
        """Create the 3x3 neighbour-weight kernel (sums to 1, zero centre) used for the Laplacian."""
        kernel = np.array([
            [0.05, 0.2, 0.05],
            [0.2, 0.0, 0.2],
            [0.05, 0.2, 0.05]
        ])
        return kernel / kernel.sum()

    def add_source(self, position, amplitude, frequency, phase=0, duration=None):
        """Add a wave source to the medium."""
        self.sources.append({
            'position': position,
            'amplitude': amplitude,
            'frequency': frequency,
            'phase': phase,
            'start_time': 0,
            'duration': duration
        })

    def add_obstacle(self, x_min, x_max, y_min, y_max):
        """Add an obstacle that blocks wave propagation."""
        self.obstacles[y_min:y_max, x_min:x_max] = True

    def add_resonant_region(self, x_min, x_max, y_min, y_max, frequency):
        """Add a region that resonates with a specific frequency."""
        self.resonant_regions[(x_min, x_max, y_min, y_max)] = frequency

    def set_memory_element(self, position, value):
        """Set a value in a specific position for wave memory."""
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            decay_rate = 0.999
            self.memory_elements[(x, y)] = (value, decay_rate)

    def read_memory_element(self, position):
        """Read a value from a memory element."""
        x, y = position
        if (x, y) in self.memory_elements:
            return self.memory_elements[(x, y)][0]
        return 0.0

    def _source_field(self, t):
        """Sum of the active sources' values at time t on their 3x3 Gaussian footprints, and the covered cells."""
        field = np.zeros_like(self.current)
        covered = np.zeros(self.current.shape, dtype=bool)
        for source in self.sources:
            if source['duration'] is not None and t >= source['start_time'] + source['duration']:
                continue
            x, y = source['position']
            value = source['amplitude'] * np.sin(2 * np.pi * source['frequency'] * t + source['phase'])
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx_pos, ny_pos = x + dx, y + dy
                    if 0 <= nx_pos < self.width and 0 <= ny_pos < self.height:
                        field[ny_pos, nx_pos] += value * np.exp(-(dx**2 + dy**2) / 2)
                        covered[ny_pos, nx_pos] = True
        return field, covered

    def _velocity_factor(self):
        """Per-cell fraction of velocity kept per substep: `damping` spread over the substeps, and no loss in a
        resonant region while its dominant frequency matches its tuning (match weight exp(-((f - f0)/0.1)^2))."""
        keep = np.full(self.current.shape, self.damping ** (1.0 / self.substeps))
        if len(self.history) > 10:
            for (x_min, x_max, y_min, y_max), freq in self.resonant_regions.items():
                local_freq = self._estimate_local_frequency(x_min, x_max, y_min, y_max)
                match = np.exp(-((local_freq - freq) / 0.1) ** 2)
                region = keep[y_min:y_max, x_min:x_max]
                keep[y_min:y_max, x_min:x_max] = np.maximum(region, region ** (1.0 - match))
        return keep

    def update(self, t):
        """Advance the medium by one output step (time t -> t + time_step) in CFL-stable substeps."""
        dt = self.dt_sub
        keep = self._velocity_factor()
        for k in range(self.substeps):
            u, u_prev = self.current, self.previous
            accel = self.diffusion * self.laplacian(u)
            if self.nonlinearity > 0:
                accel = accel - self.nonlinearity * u ** 3
            next_state = u + keep * (u - u_prev) + dt ** 2 * accel
            next_state[self.obstacles] = 0
            if self.boundary == 'absorbing':
                next_state *= self.sponge
                u = u * self.sponge
            field, covered = self._source_field(t + (k + 1) * dt)
            next_state[covered] = field[covered]
            for (x, y), (value, _) in self.memory_elements.items():
                next_state[y, x] = value
            self.previous, self.current = u, next_state

        for (x, y), (value, decay_rate) in list(self.memory_elements.items()):
            self.memory_elements[(x, y)] = (value * decay_rate, decay_rate)

        if len(self.history) > 5:
            self._update_frequency_phase_estimates()

        self.history.append(self.current.copy())
        if len(self.history) > 100:
            self.history.pop(0)

        energy = np.sum(self.current**2)
        self.energy_history.append(energy)
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)

    def _estimate_local_frequency(self, x_min, x_max, y_min, y_max):
        """Estimate local frequency in a region using FFT."""
        region_history = [frame[y_min:y_max, x_min:x_max].mean() for frame in self.history[-50:]]
        if len(region_history) < 10:
            return 0.0
        fft_result = np.abs(np.fft.rfft(region_history))
        freqs = np.fft.rfftfreq(len(region_history), d=self.time_step)
        if len(fft_result) > 1:
            peak_idx = np.argmax(fft_result[1:]) + 1
            return freqs[peak_idx]
        return 0.0

    def _update_frequency_phase_estimates(self):
        """Update local frequency (sign changes over the last 10 frames) and phase estimates."""
        if len(self.history) < 10:
            return
        recent = np.array(self.history[-10:])
        crossings = np.sum(np.diff(np.signbit(recent), axis=0), axis=0)
        self.frequency = crossings / (10 * self.time_step) / 2
        min_val = recent.min(axis=0)
        max_val = recent.max(axis=0)
        span = max_val - min_val
        safe = np.where(span > 0, span, 1.0)
        self.phase = np.where(span > 0, (self.current - min_val) / safe * 2 * np.pi, 0.0)

    def run_simulation(self, steps=100, visualize=False):
        """Run simulation for specified number of steps."""
        frames = []
        for t in range(steps):
            self.update(t * self.time_step)
            if visualize and t % 5 == 0:
                frames.append(self.current.copy())
        return frames if visualize else None

    def analyze_dynamics(self):
        """Analyze wave dynamics and patterns."""
        if len(self.history) < 10:
            return {}
        spectral_analysis = self._analyze_spectral_properties()
        pattern_analysis = self._identify_wave_patterns()
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
        """Analyze spectral properties of the wave dynamics."""
        if len(self.history) < 10:
            return {}
        recent = np.array(self.history[-10:])
        n_samples = recent.shape[0]
        flat_data = recent.reshape(n_samples, -1)
        if np.any(np.isnan(flat_data)) or np.std(flat_data) < 1e-10:
            flat_data = flat_data + np.random.normal(0, 1e-8, flat_data.shape)
        try:
            pca = PCA(n_components=min(5, n_samples))
            pca.fit(flat_data)
            dominant_modes = pca.explained_variance_ratio_
        except Exception as e:
            print(f"PCA analysis failed: {e}")
            dominant_modes = np.zeros(min(5, n_samples))

        fft_data = np.abs(np.fft.rfft(flat_data, axis=0))
        freqs = np.fft.rfftfreq(n_samples, d=self.time_step)
        peak_freq_idx = np.argmax(fft_data, axis=0)
        peak_frequencies = freqs[peak_freq_idx]
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
        """Identify wave patterns in the medium."""
        if len(self.history) < 5:
            return {}
        current = self.current
        dx = np.gradient(current, axis=1)
        dy = np.gradient(current, axis=0)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        wave_front_threshold = np.percentile(gradient_magnitude, 80)
        wave_fronts = gradient_magnitude > wave_front_threshold

        from scipy import ndimage
        labeled_fronts, num_fronts = ndimage.label(wave_fronts)
        front_sizes = []
        if num_fronts > 0:
            for i in range(1, num_fronts + 1):
                front_sizes.append(np.sum(labeled_fronts == i))

        # Detect spiral patterns
        curl = np.gradient(dy, axis=1) - np.gradient(dx, axis=0)
        curl_magnitude = np.abs(curl)
        spiral_threshold = np.percentile(curl_magnitude, 95)
        potential_spirals = curl_magnitude > spiral_threshold
        labeled_spirals, num_spirals = ndimage.label(potential_spirals)

        # Stability
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
        """Visualize the current state of the wave medium."""
        fig, ax = plt.subplots(figsize=(10, 8))
        if mode == 'amplitude':
            vmin, vmax = -0.5, 0.5
            im = ax.imshow(self.current, cmap='seismic',
                          vmin=vmin, vmax=vmax, interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='Wave Amplitude')
            ax.set_title('Wave Amplitude')
            for source in self.sources:
                x, y = source['position']
                ax.plot(x, y, 'yo', markersize=10)
            for y in range(self.height):
                for x in range(self.width):
                    if self.obstacles[y, x]:
                        ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1,
                                                 fill=True, color='black', alpha=0.7))
            for (x_min, x_max, y_min, y_max), freq in self.resonant_regions.items():
                rect = plt.Rectangle((x_min-0.5, y_min-0.5), x_max-x_min, y_max-y_min,
                                   fill=False, edgecolor='green', linewidth=2)
                ax.add_patch(rect)
                ax.text(x_min + (x_max-x_min)/2, y_min + (y_max-y_min)/2,
                       f"{freq:.1f}Hz", color='white', ha='center', va='center')
            for (x, y), (value, _) in self.memory_elements.items():
                color = 'blue' if value > 0 else 'red'
                size = min(20, max(5, abs(value) * 15))
                ax.plot(x, y, 'o', color=color, markersize=size, alpha=0.7)
        elif mode == 'frequency':
            im = ax.imshow(self.frequency, cmap='viridis',
                          vmin=0, vmax=2, interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='Frequency (Hz)')
            ax.set_title('Local Frequency Estimates')
        elif mode == 'phase':
            im = ax.imshow(self.phase / (2*np.pi), cmap='hsv',
                          vmin=0, vmax=1, interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='Phase (normalized)')
            ax.set_title('Local Phase Estimates')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.tight_layout()
        return fig

    def create_animation(self, frames=None):
        """Create an animation of wave dynamics."""
        if frames is None:
            if len(self.history) < 2:
                return None
            frames = self.history
        fig, ax = plt.subplots(figsize=(8, 8))
        vmin = min(np.min(frame) for frame in frames)
        vmax = max(np.max(frame) for frame in frames)
        cdict = {
            'red': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
            'green': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)],
            'blue': [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)]
        }
        wave_cmap = LinearSegmentedColormap('wave_cmap', cdict)
        im = ax.imshow(frames[0], cmap=wave_cmap,
                      vmin=vmin, vmax=vmax, interpolation='bilinear')
        for y in range(self.height):
            for x in range(self.width):
                if self.obstacles[y, x]:
                    ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1,
                                             fill=True, color='black', alpha=0.7))
        for source in self.sources:
            x, y = source['position']
            ax.plot(x, y, 'yo', markersize=8)
        plt.colorbar(im, ax=ax, label='Wave Amplitude')
        ax.set_title('Wave Dynamics')

        def update_frame(frame):
            im.set_array(frame)
            return [im]

        ani = FuncAnimation(fig, update_frame, frames=frames, interval=50, blit=True)
        return ani

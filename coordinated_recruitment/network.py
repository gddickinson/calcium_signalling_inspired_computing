"""
CoordinatedRecruitmentNetwork -- grid of recruitment elements.

Manages a 2D grid of CoordinatedRecruitmentElements with coupling,
cluster identification (DBSCAN), fault injection, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

from .element import CoordinatedRecruitmentElement


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

        self.elements = np.empty(grid_size, dtype=object)
        self.faulty_elements = set()

        for i in range(self.height):
            for j in range(self.width):
                sensitivity = max(0.2, min(0.8, 0.4 + 0.2 * np.random.randn()))
                threshold = max(0.1, min(0.5, 0.3 + 0.1 * np.random.randn()))
                self.elements[i, j] = CoordinatedRecruitmentElement(
                    sensitivity=sensitivity, recruitment_threshold=threshold
                )
                if np.random.random() < fault_probability:
                    self.faulty_elements.add((i, j))

        self.coupling_kernel = self._create_coupling_kernel(coupling_radius)
        self.activation_history = []
        self.cluster_history = []
        self.direct_input = np.zeros(grid_size)

    def _create_coupling_kernel(self, radius):
        """Create a kernel for element coupling based on distance."""
        size = 2 * radius + 1
        kernel = np.zeros((size, size))
        center = radius
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist <= radius:
                    kernel[i, j] = np.exp(-dist / (radius / 2))
        kernel[center, center] = 0
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()
        return kernel

    def set_input_region(self, center, radius, strength):
        """Set a circular input region."""
        center_i, center_j = center
        for i in range(self.height):
            for j in range(self.width):
                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                if dist <= radius:
                    self.direct_input[i, j] = strength * (1 - dist/radius)

    def set_input_pattern(self, pattern):
        """Set an arbitrary input pattern."""
        if pattern.shape != (self.height, self.width):
            raise ValueError(f"Pattern shape {pattern.shape} doesn't match grid size ({self.height}, {self.width})")
        self.direct_input = pattern.copy()

    def update(self):
        """Update all elements in the network for one time step."""
        activation_grid = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.faulty_elements:
                    activation_grid[i, j] = self.elements[i, j].activation

        neighborhood_grid = convolve(activation_grid, self.coupling_kernel, mode='constant', cval=0.0)

        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.faulty_elements:
                    self.elements[i, j].update(
                        self.direct_input[i, j], neighborhood_grid[i, j]
                    )

        current_activation = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.faulty_elements:
                    current_activation[i, j] = self.elements[i, j].activation
        self.activation_history.append(current_activation)

        clusters = self._identify_clusters(current_activation)
        self.cluster_history.append(clusters)

        if len(self.activation_history) > 100:
            self.activation_history.pop(0)
            self.cluster_history.pop(0)

    def _identify_clusters(self, activation_grid, activation_threshold=0.3):
        """Identify clusters of activated elements."""
        active_positions = []
        for i in range(self.height):
            for j in range(self.width):
                if activation_grid[i, j] > activation_threshold:
                    active_positions.append((i, j))
        if not active_positions:
            return []
        active_positions = np.array(active_positions)
        clustering = DBSCAN(eps=1.5, min_samples=3).fit(active_positions)
        clusters = {}
        for pos, label in zip(active_positions, clustering.labels_):
            if label != -1:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(tuple(pos))
        return list(clusters.values())

    def run_simulation(self, steps=100, input_pattern_func=None):
        """Run the simulation for a specified number of steps."""
        for step in range(steps):
            if input_pattern_func is not None:
                self.set_input_pattern(input_pattern_func(step))
            self.update()
        return self._analyze_results()

    def _analyze_results(self):
        """Analyze simulation results."""
        if not self.activation_history:
            return {}
        cluster_sizes = []
        for clusters in self.cluster_history:
            sizes = [len(cluster) for cluster in clusters]
            cluster_sizes.append(sizes if sizes else [0])
        avg_cluster_sizes = [np.mean(sizes) if sizes else 0 for sizes in cluster_sizes]
        num_clusters = [len(clusters) for clusters in self.cluster_history]
        total_activation = [np.sum(act) for act in self.activation_history]
        return {
            'avg_cluster_sizes': avg_cluster_sizes,
            'num_clusters': num_clusters,
            'total_activation': total_activation,
            'cluster_history': self.cluster_history,
            'activation_history': self.activation_history
        }

    def visualize_state(self, time_step=-1):
        """Visualize network state at a specified time step."""
        if not self.activation_history:
            return None
        if time_step >= len(self.activation_history) or time_step < -len(self.activation_history):
            time_step = -1
        activation = self.activation_history[time_step]
        clusters = self.cluster_history[time_step]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        im = axes[0].imshow(activation, cmap='hot', vmin=0, vmax=1)
        axes[0].set_title('Element Activation Levels')
        plt.colorbar(im, ax=axes[0])
        for i, j in self.faulty_elements:
            axes[0].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='blue', linewidth=1))
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
        """Create animation of network dynamics."""
        if len(self.activation_history) < frames:
            self.run_simulation(steps=frames)
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.activation_history[0], cmap='hot', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_title('Element Activation Dynamics')
        for i, j in self.faulty_elements:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='blue', linewidth=1))

        def update_frame(frame):
            im.set_array(self.activation_history[frame])
            ax.set_title(f'Element Activation Dynamics (t={frame})')
            return [im]

        ani = FuncAnimation(fig, update_frame, frames=min(frames, len(self.activation_history)),
                           interval=150, blit=True)
        return ani

    def visualize_clusters_3d(self):
        """Visualize cluster formation over time in 3D."""
        if len(self.cluster_history) < 2:
            return None
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for t, clusters in enumerate(self.cluster_history):
            for cluster_idx, cluster in enumerate(clusters):
                xs = [j for i, j in cluster]
                ys = [i for i, j in cluster]
                zs = [t] * len(cluster)
                color = plt.cm.tab20(cluster_idx % 20)
                ax.scatter(xs, ys, zs, c=[color], s=30, alpha=0.7)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Time')
        ax.set_title('Cluster Formation Over Time')
        return fig

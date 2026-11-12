"""
PatternRecognitionComputer -- recognise input patterns by the early dynamics they produce in a
SpatiotemporalComputing model.

Signature of a run: the first `signature_frames` frames of the model's history (frame 0 is the input itself),
each coarse-grained to `block` x `block` cell means per channel with the channel's mean subtracted (removing the
uniform offset that noise produces through the tanh). Training stores the signature of each clean pattern and an
acceptance radius: `radius_factor` times the largest distance of `n_noisy` noisy training copies (noise sd
`train_noise`) from it. recognize() returns the nearest stored pattern if the input lies within that pattern's
radius, else None (a pattern that was never stored).

Earlier frames are used because the model is dissipative: later frames lose the pattern and are dominated by
amplified noise. (Before 2026-11-11 each pattern was clustered with its own PCA + k-means and transition-graph
edges were compared across runs, where the cluster labels meant different things; the demo matched 1 of 3.)
"""

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class PatternRecognitionComputer:
    def __init__(self, model, block=5, signature_frames=5, train_noise=0.2, n_noisy=5, radius_factor=2.0):
        self.model = model
        self.block = block
        self.signature_frames = signature_frames
        self.train_noise = train_noise
        self.n_noisy = n_noisy
        self.radius_factor = radius_factor
        self.reference_patterns = {}
        self.signatures = {}
        self.radii = {}
        self.pattern_graphs = {}
        self.last_distances = {}

    def _run(self, pattern, steps):
        """Run the model from `pattern` with a fresh history; return the history as an array."""
        self.model.state = pattern.copy()
        self.model.history = []
        self.model.run_simulation(steps=steps)
        return np.array(self.model.history)

    def signature(self, history):
        """Coarse-grained, per-channel mean-centred first frames of a run, shape (frames, features)."""
        frames = np.asarray(history[:self.signature_frames])
        t, c, h, w = frames.shape
        b = self.block
        hb, wb = h // b, w // b
        coarse = frames[:, :, :hb * b, :wb * b].reshape(t, c, hb, b, wb, b).mean(axis=(3, 5))
        coarse = coarse - coarse.mean(axis=(2, 3), keepdims=True)
        return coarse.reshape(t, -1)

    @staticmethod
    def distance(sig1, sig2):
        """Mean over frames of the Euclidean distance between two signatures."""
        n = min(len(sig1), len(sig2))
        return float(np.linalg.norm(sig1[:n] - sig2[:n], axis=1).mean())

    def train(self, patterns, steps=20):
        """
        Store patterns.

        Parameters:
        - patterns: Dictionary of named patterns, each a numpy array matching state dimensions
        - steps: simulation steps per run (at least signature_frames)
        """
        self.reference_patterns = patterns
        steps = max(steps, self.signature_frames)
        for name, pattern in patterns.items():
            history = self._run(pattern, steps)
            self.signatures[name] = self.signature(history)
            self.pattern_graphs[name] = self._compute_pattern_graph(list(history))
            spread = [self.distance(self.signature(self._run(
                pattern + np.random.normal(0, self.train_noise, pattern.shape), steps)), self.signatures[name])
                for _ in range(self.n_noisy)] if self.train_noise > 0 and self.n_noisy > 0 else [0.0]
            self.radii[name] = self.radius_factor * max(spread) if max(spread) > 0 else np.inf

    def recognize(self, input_pattern, steps=20):
        """
        Recognize an input pattern by comparing the dynamics it produces with the stored ones.

        Returns:
        - (best matching pattern name, or None if the input is outside every stored pattern's radius,
           score = radius / (radius + distance) of the nearest pattern: above 0.5 inside the radius)
        """
        history = self._run(input_pattern, max(steps, self.signature_frames))
        sig = self.signature(history)
        self.last_distances = {name: self.distance(sig, ref) for name, ref in self.signatures.items()}
        if not self.last_distances:
            return None, 0.0
        best = min(self.last_distances, key=self.last_distances.get)
        d, r = self.last_distances[best], self.radii[best]
        score = 1.0 if np.isinf(r) else r / (r + d)
        return (best if d <= r else None), float(score)

    # --- legacy helpers: transition graph of a run's k-means states (labels are per run) ---------------------

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

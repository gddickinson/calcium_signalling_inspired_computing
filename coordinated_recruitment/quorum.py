"""
QuorumComputer -- applications of coordinated recruitment.

Implements fault tolerance testing, signal amplification analysis,
and self-organizing computation using the CoordinatedRecruitmentNetwork.
"""

import numpy as np
from .network import CoordinatedRecruitmentNetwork


class QuorumComputer:
    def __init__(self, network):
        self.network = network

    def fault_tolerant_computation(self, input_patterns, fault_prob_range, trials=5):
        """Test fault tolerance by increasing fault probability."""
        results = []
        for fault_prob in fault_prob_range:
            trial_results = []
            for _ in range(trials):
                test_network = CoordinatedRecruitmentNetwork(
                    grid_size=(self.network.height, self.network.width),
                    coupling_radius=self.network.coupling_radius,
                    fault_probability=fault_prob
                )
                for pattern in input_patterns:
                    test_network.set_input_pattern(pattern)
                    test_network.run_simulation(steps=20)
                    analysis = test_network._analyze_results()
                    final_activation = analysis['total_activation'][-1] if analysis['total_activation'] else 0
                    max_cluster_size = max(analysis['avg_cluster_sizes']) if analysis['avg_cluster_sizes'] else 0
                    trial_results.append({
                        'fault_prob': fault_prob,
                        'final_activation': final_activation,
                        'max_cluster_size': max_cluster_size,
                        'num_clusters': analysis['num_clusters'][-1] if analysis['num_clusters'] else 0
                    })
            results.append({
                'fault_prob': fault_prob,
                'avg_activation': np.mean([r['final_activation'] for r in trial_results]),
                'avg_max_cluster': np.mean([r['max_cluster_size'] for r in trial_results]),
                'avg_num_clusters': np.mean([r['num_clusters'] for r in trial_results])
            })
        return results

    def signal_amplification(self, input_strengths, duration=20):
        """Demonstrate signal amplification for weak inputs."""
        results = []
        for strength in input_strengths:
            pattern = np.zeros((self.network.height, self.network.width))
            center_i, center_j = self.network.height // 2, self.network.width // 2
            radius = min(self.network.height, self.network.width) // 6
            for i in range(self.network.height):
                for j in range(self.network.width):
                    dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                    if dist <= radius:
                        pattern[i, j] = strength

            self.network = CoordinatedRecruitmentNetwork(
                grid_size=(self.network.height, self.network.width),
                coupling_radius=self.network.coupling_radius
            )
            self.network.set_input_pattern(pattern)
            self.network.run_simulation(steps=duration)

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
        """Demonstrate how computation self-organizes when transitioning between inputs."""
        def blend_patterns(step):
            if step < 10:
                return input_pattern1
            elif step >= 10 + blend_steps:
                return input_pattern2
            else:
                alpha = (step - 10) / blend_steps
                return (1 - alpha) * input_pattern1 + alpha * input_pattern2

        self.network = CoordinatedRecruitmentNetwork(
            grid_size=(self.network.height, self.network.width),
            coupling_radius=self.network.coupling_radius
        )
        total_steps = 10 + blend_steps + 20
        self.network.run_simulation(steps=total_steps, input_pattern_func=blend_patterns)

        cluster_transitions = []
        for t in range(1, len(self.network.cluster_history)):
            prev_elements = set()
            for cluster in self.network.cluster_history[t-1]:
                prev_elements.update(cluster)
            curr_elements = set()
            for cluster in self.network.cluster_history[t]:
                curr_elements.update(cluster)
            cluster_transitions.append({
                'time': t,
                'persisting': len(prev_elements.intersection(curr_elements)),
                'new': len(curr_elements - prev_elements),
                'disbanded': len(prev_elements - curr_elements),
                'num_clusters': len(self.network.cluster_history[t])
            })
        return {
            'cluster_transitions': cluster_transitions,
            'activation_history': self.network.activation_history
        }

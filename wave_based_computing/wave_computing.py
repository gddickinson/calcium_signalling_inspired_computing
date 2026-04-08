"""
Wave Computing -- higher-level computing operations on wave media.

Implements pattern recognition, logic gates, wave memory, and
signal filtering using the WaveComputingMedium as a substrate.
"""

import numpy as np
from sklearn.decomposition import PCA

from .wave_medium import WaveComputingMedium


class WaveComputing:
    """Wave-based computing operations built on a WaveComputingMedium."""

    def __init__(self, medium: WaveComputingMedium):
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
            medium = WaveComputingMedium(
                size=(self.medium.width, self.medium.height),
                boundary=self.medium.boundary,
                damping=self.medium.damping,
                diffusion=self.medium.diffusion,
                nonlinearity=self.medium.nonlinearity
            )
            self._encode_pattern_as_sources(medium, test_pattern)
            medium.run_simulation(steps=simulation_steps)
            test_signature = self._extract_wave_signature(medium)

            pattern_matches = []
            for ref_idx, ref_pattern in enumerate(reference_patterns):
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
                similarity = self._calculate_signature_similarity(test_signature, ref_signature)
                pattern_matches.append({
                    'reference_index': ref_idx,
                    'similarity': similarity
                })
            pattern_matches.sort(key=lambda x: x['similarity'], reverse=True)
            results.append({
                'test_index': test_idx,
                'matches': pattern_matches,
                'wave_signature': test_signature
            })
        return results

    def _encode_pattern_as_sources(self, medium, pattern):
        """Encode a pattern as wave sources."""
        height, width = pattern.shape
        scale_y = medium.height / height
        scale_x = medium.width / width
        threshold = np.percentile(pattern, 80)
        y_indices, x_indices = np.where(pattern > threshold)
        for y, x in zip(y_indices, x_indices):
            medium_x = int(x * scale_x)
            medium_y = int(y * scale_y)
            intensity = pattern[y, x]
            amplitude = 0.5 * intensity
            frequency = 0.5 + 0.5 * (x / width)
            medium.add_source((medium_x, medium_y), amplitude, frequency)

    def _extract_wave_signature(self, medium):
        """Extract a signature from wave dynamics for pattern matching."""
        if len(medium.history) < 10:
            return np.zeros(5)
        recent = np.array(medium.history[-10:])
        n_samples = recent.shape[0]
        flat_data = recent.reshape(n_samples, -1)
        if np.std(flat_data) < 1e-6:
            flat_data += np.random.normal(0, 1e-5, flat_data.shape)
        try:
            pca = PCA(n_components=min(8, n_samples))
            pca.fit(flat_data)
            signature = []
            for component, variance in zip(pca.components_, pca.explained_variance_):
                signature.extend([
                    variance,
                    np.mean(component),
                    np.std(component),
                    np.percentile(component, 10),
                    np.percentile(component, 90),
                    np.max(component) - np.min(component)
                ])
            last_frame = medium.current
            signature.extend([
                np.mean(last_frame),
                np.std(last_frame),
                np.max(last_frame),
                np.min(last_frame)
            ])
        except Exception as e:
            print(f"PCA signature extraction failed: {e}")
            signature = [0.0] * 52
        if np.std(signature) > 0:
            signature = (signature - np.mean(signature)) / np.std(signature)
        return np.array(signature)

    def _calculate_signature_similarity(self, sig1, sig2):
        """Calculate similarity between two wave signatures."""
        min_len = min(len(sig1), len(sig2))
        sig1 = sig1[:min_len]
        sig2 = sig2[:min_len]
        sig1_norm = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-6)
        sig2_norm = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-6)
        dot_product = np.dot(sig1_norm, sig2_norm)
        magnitude1 = np.sqrt(np.dot(sig1_norm, sig1_norm))
        magnitude2 = np.sqrt(np.dot(sig2_norm, sig2_norm))
        similarity = dot_product / (magnitude1 * magnitude2 + 1e-6)
        return (similarity + 1) / 2

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
        medium = WaveComputingMedium(
            size=(self.medium.width, self.medium.height),
            boundary=self.medium.boundary,
            damping=self.medium.damping,
            diffusion=self.medium.diffusion,
            nonlinearity=self.medium.nonlinearity
        )
        if operation == 'and':
            for y in range(medium.height // 4, 3 * medium.height // 4):
                medium.add_source((5, y), 0.3, 0.5, phase=0)
            for x in range(medium.width // 4, 3 * medium.width // 4):
                medium.add_source((x, 5), 0.3, 0.5, phase=0)
            center_x, center_y = medium.width // 2, medium.height // 2
            medium.add_resonant_region(
                center_x - 10, center_x + 10,
                center_y - 10, center_y + 10,
                frequency=0.5
            )
        elif operation == 'or':
            for y in range(medium.height // 4, 3 * medium.height // 4):
                medium.add_source((5, y), 0.2, 0.7, phase=0)
            for x in range(medium.width // 4, 3 * medium.width // 4):
                medium.add_source((x, 5), 0.2, 0.7, phase=0)
            center_x, center_y = medium.width // 2, medium.height // 2
            medium.add_resonant_region(
                center_x - 15, center_x + 15,
                center_y - 15, center_y + 15,
                frequency=0.7
            )
        elif operation == 'xor':
            for y in range(medium.height // 4, 3 * medium.height // 4):
                medium.add_source((5, y), 0.3, 0.6, phase=0)
            for x in range(medium.width // 4, 3 * medium.width // 4):
                medium.add_source((x, 5), 0.3, 0.6, phase=np.pi)
            medium.nonlinearity = 0.2

        frames = medium.run_simulation(steps=simulation_steps, visualize=True)
        center_x, center_y = medium.width // 2, medium.height // 2
        region_size = 10
        x_min = max(0, center_x - region_size)
        x_max = min(medium.width, center_x + region_size)
        y_min = max(0, center_y - region_size)
        y_max = min(medium.height, center_y + region_size)

        last_frames = medium.history[-10:]
        if not last_frames:
            return None
        center_activations = [
            np.max(np.abs(frame[y_min:y_max, x_min:x_max]))
            for frame in last_frames
        ]
        result_value = np.max(center_activations)

        output = np.zeros((medium.height, medium.width))
        output[y_min:y_max, x_min:x_max] = result_value
        return {
            'operation': operation,
            'result_value': result_value,
            'output_pattern': output,
            'simulation_frames': frames
        }

    def wave_memory(self, input_data, storage_time=50, retrieval_time=20):
        """Store and retrieve information using wave dynamics."""
        medium = WaveComputingMedium(
            size=(self.medium.width, self.medium.height),
            boundary='absorbing',
            damping=0.995,
            diffusion=0.1,
            nonlinearity=0.05
        )
        data_length = len(input_data)
        spacing = medium.width // (data_length + 1)
        for i, value in enumerate(input_data):
            x = (i + 1) * spacing
            y = 5
            amplitude = 0.3 * abs(value)
            frequency = 0.5 + 0.3 * (value + 1) / 2
            phase = 0 if value >= 0 else np.pi
            medium.add_source((x, y), amplitude, frequency, phase, duration=10)
            region_size = 5
            medium.add_resonant_region(
                x - region_size, x + region_size,
                medium.height // 2 - region_size, medium.height // 2 + region_size,
                frequency=frequency
            )
            medium.set_memory_element((x, medium.height // 2), value)

        storage_frames = medium.run_simulation(steps=storage_time, visualize=True)
        for x in range(spacing, medium.width, spacing):
            medium.add_source((x, medium.height - 5), 0.5, 0.65, phase=0, duration=5)
        retrieval_frames = medium.run_simulation(steps=retrieval_time, visualize=True)

        retrieved_data = []
        for i in range(data_length):
            x = (i + 1) * spacing
            y = medium.height // 2
            region_size = 3
            x_min = max(0, x - region_size)
            x_max = min(medium.width, x + region_size + 1)
            y_min = max(0, y - region_size)
            y_max = min(medium.height, y + region_size + 1)
            recent_frames = medium.history[-5:]
            if not recent_frames:
                retrieved_data.append(0)
                continue
            activations = [
                frame[y_min:y_max, x_min:x_max].mean()
                for frame in recent_frames
            ]
            avg_activation = np.mean(activations)
            memory_value = medium.read_memory_element((x, y))
            retrieved_value = 0.7 * memory_value + 0.3 * (avg_activation * 2)
            retrieved_data.append(retrieved_value)
        return {
            'input_data': input_data,
            'retrieved_data': retrieved_data,
            'storage_frames': storage_frames,
            'retrieval_frames': retrieval_frames
        }

    def wave_filter(self, input_signal, filter_type='lowpass', cutoff_frequency=0.5, simulation_steps=100):
        """Filter a 1D signal using wave dynamics."""
        medium = WaveComputingMedium(
            size=(self.medium.width, self.medium.height),
            boundary='absorbing',
            damping=0.99,
            diffusion=0.1
        )
        signal_length = len(input_signal)
        signal_scale = min(1.0, medium.height / signal_length)
        for i, value in enumerate(input_signal):
            y = int(i * signal_scale)
            if y >= medium.height:
                break
            amplitude = 0.4 * abs(value)
            frequency = 0.3 + 0.6 * (value + 1) / 2
            medium.add_source((5, y), amplitude, frequency)

        if filter_type == 'lowpass':
            for y in range(medium.height):
                medium.add_resonant_region(
                    medium.width // 3, 2 * medium.width // 3,
                    max(0, y - 2), min(medium.height, y + 3),
                    frequency=cutoff_frequency / 2
                )
        elif filter_type == 'highpass':
            for y in range(medium.height):
                medium.add_resonant_region(
                    medium.width // 3, 2 * medium.width // 3,
                    max(0, y - 2), min(medium.height, y + 3),
                    frequency=cutoff_frequency * 1.5
                )
        elif filter_type == 'bandpass':
            for y in range(medium.height):
                medium.add_resonant_region(
                    medium.width // 3, 2 * medium.width // 3,
                    max(0, y - 2), min(medium.height, y + 3),
                    frequency=cutoff_frequency
                )

        frames = medium.run_simulation(steps=simulation_steps, visualize=True)
        output_x = int(4 * medium.width / 5)
        filtered_signal = []
        for i in range(min(signal_length, medium.height)):
            y = int(i * signal_scale)
            if y >= medium.height:
                break
            recent_frames = medium.history[-10:]
            if not recent_frames:
                filtered_signal.append(0)
                continue
            amplitudes = [frame[y, output_x] for frame in recent_frames]
            filtered_signal.append(np.mean(amplitudes))

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

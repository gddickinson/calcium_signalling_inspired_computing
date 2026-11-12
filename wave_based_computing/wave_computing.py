"""
Wave Computing -- higher-level computing operations on wave media.

Implements pattern recognition, logic gates, wave memory, and
signal filtering using the WaveComputingMedium as a substrate.
"""

import numpy as np

from .wave_medium import WaveComputingMedium


class WaveComputing:
    """Wave-based computing operations built on a WaveComputingMedium."""

    def __init__(self, medium: WaveComputingMedium):
        self.medium = medium

    MATCH_THRESHOLD = 0.9      # minimum signature correlation for a match
    SIGNATURE_BLOCK = 5        # cells per side of the blocks the amplitude map is averaged over
    SIGNATURE_FRAMES = 20      # frames the RMS amplitude is taken over

    def pattern_recognition(self, test_patterns, reference_patterns, simulation_steps=100,
                            match_threshold=None):
        """
        Use wave dynamics for pattern recognition.

        Each pattern drives its own medium (see _encode_pattern_as_sources); its signature is the map of RMS wave
        amplitude over the last frames (_extract_wave_signature). A test pattern matches the reference whose
        signature correlates best with its own, if that correlation reaches match_threshold (default 0.9);
        otherwise it is reported as unknown.

        Parameters:
        - test_patterns: List of patterns to recognize
        - reference_patterns: Known reference patterns
        - simulation_steps: Number of steps to run for each pattern
        - match_threshold: minimum correlation for a match

        Returns:
        - per test pattern: test_index, matches (reference_index, similarity = correlation in [-1, 1], best
          first), best_match (reference index or None) and wave_signature
        """
        threshold = self.MATCH_THRESHOLD if match_threshold is None else match_threshold

        def signature(pattern):
            medium = self._new_medium()
            self._encode_pattern_as_sources(medium, pattern)
            medium.run_simulation(steps=simulation_steps)
            return self._extract_wave_signature(medium)

        ref_signatures = [signature(p) for p in reference_patterns]
        results = []
        for test_idx, test_pattern in enumerate(test_patterns):
            test_signature = signature(test_pattern)
            pattern_matches = sorted(
                ({'reference_index': i, 'similarity': self._calculate_signature_similarity(test_signature, ref)}
                 for i, ref in enumerate(ref_signatures)),
                key=lambda x: x['similarity'], reverse=True)
            best = pattern_matches[0] if pattern_matches else None
            results.append({
                'test_index': test_idx,
                'matches': pattern_matches,
                'best_match': best['reference_index'] if best and best['similarity'] >= threshold else None,
                'wave_signature': test_signature
            })
        return results

    def _encode_pattern_as_sources(self, medium, pattern):
        """Encode a pattern as wave sources: one source per pixel above the middle of the pattern's value range
        (amplitude 0.5 x pixel value, frequency 0.5 to 1.0 from left to right). (The old rule, pixels strictly
        above the 80th percentile, selected no pixels at all for binary patterns covering more than 20% of the
        grid, and a different subset for noisy copies.)"""
        height, width = pattern.shape
        scale_y = medium.height / height
        scale_x = medium.width / width
        lo, hi = float(np.min(pattern)), float(np.max(pattern))
        y_indices, x_indices = np.where(pattern > lo + 0.5 * (hi - lo))
        for y, x in zip(y_indices, x_indices):
            medium.add_source((int(x * scale_x), int(y * scale_y)), 0.5 * pattern[y, x], 0.5 + 0.5 * (x / width))

    def _extract_wave_signature(self, medium):
        """Signature of a run: RMS wave amplitude over the last SIGNATURE_FRAMES frames, averaged over
        SIGNATURE_BLOCK x SIGNATURE_BLOCK cell blocks, mean removed, scaled to unit length. (It replaces summary
        statistics of PCA components, which came out nearly the same for every pattern.)"""
        frames = np.array(medium.history[-self.SIGNATURE_FRAMES:])
        if len(frames) == 0:
            return np.zeros(1)
        rms = np.sqrt((frames ** 2).mean(axis=0))
        b = self.SIGNATURE_BLOCK
        hb, wb = rms.shape[0] // b, rms.shape[1] // b
        blocks = rms[:hb * b, :wb * b].reshape(hb, b, wb, b).mean(axis=(1, 3)).ravel()
        blocks = blocks - blocks.mean()
        norm = np.linalg.norm(blocks)
        return blocks / norm if norm > 0 else blocks

    def _calculate_signature_similarity(self, sig1, sig2):
        """Correlation of two unit-length, mean-removed signatures (1 = same amplitude map)."""
        n = min(len(sig1), len(sig2))
        return float(np.dot(sig1[:n], sig2[:n]))

    # Gate set-up: one line of sources per input, equidistant from the readout point at the grid centre
    GATE_AMPLITUDE = 0.3
    GATE_FREQUENCY = 0.5
    GATE_THRESHOLDS = {'and': 1.5, 'or': 0.5, 'xor': 0.5}   # x the single-input readout

    def _new_medium(self, **overrides):
        """A fresh medium with this computer's grid and physical parameters (overrides win)."""
        kw = dict(size=(self.medium.width, self.medium.height), boundary=self.medium.boundary,
                  damping=self.medium.damping, wave_speed=self.medium.wave_speed,
                  nonlinearity=self.medium.nonlinearity, time_step=self.medium.time_step)
        kw.update(overrides)
        return WaveComputingMedium(**kw)

    def gate_medium(self, bits, operation):
        """Medium for a gate: input A is a line of sources at x = 5 (rows H/4..3H/4), input B a line at y = 5
        (columns W/4..3W/4); a line is switched on when its bit is 1. B is in phase with A for AND/OR and in
        antiphase for XOR, so at the centre (equidistant from both lines on a square grid) the two waves add
        or cancel."""
        medium = self._new_medium(boundary='absorbing')
        phase_b = np.pi if operation == 'xor' else 0.0
        if bits[0]:
            for y in range(medium.height // 4, 3 * medium.height // 4):
                medium.add_source((5, y), self.GATE_AMPLITUDE, self.GATE_FREQUENCY, phase=0.0)
        if bits[1]:
            for x in range(medium.width // 4, 3 * medium.width // 4):
                medium.add_source((x, 5), self.GATE_AMPLITUDE, self.GATE_FREQUENCY, phase=phase_b)
        return medium

    @staticmethod
    def gate_readout(medium, window=20):
        """Wave amplitude at the centre: largest |mean of the 3x3 centre cells| over the last `window` frames
        (20 frames = one period at the gate frequency with time_step 0.1)."""
        cx, cy = medium.width // 2, medium.height // 2
        frames = medium.history[-window:]
        if not frames:
            return 0.0
        return float(max(abs(f[cy - 1:cy + 2, cx - 1:cx + 2].mean()) for f in frames))

    def _gate_reference(self, simulation_steps):
        """Readout with only input A on (cached per grid and parameters): the unit for the thresholds."""
        key = (self.medium.width, self.medium.height, self.medium.damping, self.medium.wave_speed,
               self.medium.nonlinearity, self.medium.time_step, simulation_steps)
        cache = self.__dict__.setdefault('_gate_refs', {})
        if key not in cache:
            ref = self.gate_medium((True, False), 'or')
            ref.run_simulation(steps=simulation_steps)
            cache[key] = self.gate_readout(ref)
        return cache[key]

    def wave_logic_gates(self, input_patterns, operation='and', simulation_steps=100):
        """
        Implement logic gates using wave interference.

        Each of the first two input patterns is read as a bit (pattern mean > 0.5) that switches a line of
        sources on or off (see gate_medium). The output is the wave amplitude at the grid centre compared with
        the amplitude one input alone produces there (the reference): AND needs more than 1.5x the reference
        (both waves arriving in phase), OR more than 0.5x, and XOR more than 0.5x with input B in antiphase, so
        that two active inputs cancel at the centre.

        Parameters:
        - input_patterns: List of at least two input patterns (arrays; bit = mean > 0.5)
        - operation: Logic operation ('and', 'or', 'xor')
        - simulation_steps: Number of simulation steps (the waves need ~(W/2 - 5) / wave_speed time units
          to reach the centre)

        Returns:
        - dict with operation, inputs (bits), output (bool), result_value (centre amplitude), reference,
          threshold, output_pattern and simulation_frames; None if fewer than two inputs
        """
        if len(input_patterns) < 2:
            return None
        if operation not in self.GATE_THRESHOLDS:
            raise ValueError(f"Unknown operation {operation!r}; use 'and', 'or' or 'xor'")
        bits = [bool(float(np.mean(p)) > 0.5) for p in input_patterns[:2]]
        medium = self.gate_medium(bits, operation)
        frames = medium.run_simulation(steps=simulation_steps, visualize=True)
        result_value = self.gate_readout(medium)
        reference = self._gate_reference(simulation_steps)
        threshold = self.GATE_THRESHOLDS[operation] * reference
        output = bool(result_value > threshold)

        cx, cy = medium.width // 2, medium.height // 2
        output_pattern = np.zeros((medium.height, medium.width))
        output_pattern[max(0, cy - 10):cy + 10, max(0, cx - 10):cx + 10] = result_value
        return {
            'operation': operation,
            'inputs': bits,
            'output': output,
            'result_value': result_value,
            'reference': reference,
            'threshold': threshold,
            'output_pattern': output_pattern,
            'simulation_frames': frames
        }

    def wave_memory(self, input_data, storage_time=50, retrieval_time=20):
        """Store and retrieve information using wave dynamics.

        Each retrieved value is 0.7 * the (slowly decaying) value held in the
        memory element at that position plus 0.3 * twice the mean wave activity
        around it, so most of the readout comes from the stored number itself.
        """
        medium = self._new_medium(boundary='absorbing', damping=0.995, nonlinearity=0.05)
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
        medium = self._new_medium(boundary='absorbing', damping=0.99, nonlinearity=0.1)
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

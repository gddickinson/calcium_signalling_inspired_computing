"""
CoordinatedRecruitmentElement -- single computing element.

Models one unit with quorum-sensing activation, fatigue,
and recovery dynamics inspired by calcium channel clusters.
"""

import numpy as np


class CoordinatedRecruitmentElement:
    def __init__(self, sensitivity=0.5, recruitment_threshold=0.3, coupling_strength=0.2,
                 recovery_time=5, max_activation=1.0):
        """
        Initialize a coordinated recruitment computing element.

        Parameters:
        - sensitivity: Sensitivity to input signals
        - recruitment_threshold: Threshold for recruiting neighboring elements
        - coupling_strength: Strength of coupling between elements
        - recovery_time: Time it takes to recover after activation
        - max_activation: Maximum activation level
        """
        self.sensitivity = sensitivity
        self.recruitment_threshold = recruitment_threshold
        self.coupling_strength = coupling_strength
        self.recovery_time = recovery_time
        self.max_activation = max_activation

        self.activation = 0.0
        self.recovery_countdown = 0
        self.neighborhood_activation = 0.0
        self.fatigue = 0.0

    def update(self, direct_input, neighborhood_activation):
        """
        Update element state based on direct input and neighborhood activation.

        Returns:
        - recruitment_signal: Signal to recruit neighbors
        """
        self.neighborhood_activation = neighborhood_activation
        effective_sensitivity = self.sensitivity * (1.0 - 0.5 * self.fatigue)

        if self.recovery_countdown > 0:
            self.recovery_countdown -= 1
            effective_sensitivity *= 0.5

        total_input = (direct_input * effective_sensitivity +
                      neighborhood_activation * self.coupling_strength)

        recruitment_signal = 0.0
        if total_input > self.recruitment_threshold and self.recovery_countdown == 0:
            self.activation = min(self.max_activation, total_input)
            recruitment_signal = self.activation * 0.8
            self.fatigue = min(1.0, self.fatigue + 0.1)
            self.recovery_countdown = self.recovery_time
        else:
            self.activation *= 0.8
            self.fatigue = max(0.0, self.fatigue - 0.01)

        return recruitment_signal

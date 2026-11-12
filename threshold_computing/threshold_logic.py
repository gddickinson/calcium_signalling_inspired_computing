"""
ThresholdLogicComputer -- applications of a ThresholdComputingNetwork.

Temporal integration of weak signals, threshold detection over a range of
signal strengths, and analog memory traces.
"""

import numpy as np


class ThresholdLogicComputer:
    def __init__(self, network):
        self.network = network
    
    def temporal_integration(self, input_sequence, duration=50):
        """
        Demonstrate temporal integration of weak signals.
        
        Parameters:
        - input_sequence: Sequence of input patterns
        - duration: Duration to run the simulation
        
        Returns:
        - output_sequence: Sequence of outputs
        """
        # Pad input sequence if needed
        if len(input_sequence) < duration:
            padded_sequence = input_sequence + [np.zeros(len(self.network.input_region))] * (duration - len(input_sequence))
        else:
            padded_sequence = input_sequence[:duration]
        
        # Run simulation
        outputs = self.network.run_simulation(steps=duration, input_patterns=padded_sequence)
        
        return outputs
    
    def threshold_detection(self, signal_strengths, duration_per_signal=20):
        """
        Demonstrate threshold detection with varying signal strengths.
        
        Parameters:
        - signal_strengths: List of signal strengths to test
        - duration_per_signal: Duration to apply each signal
        
        Returns:
        - detection_results: Whether each signal was detected

        Each strength is tested on the network reset to rest, so results do not
        depend on the order of signal_strengths.
        """
        detection_results = []
        
        for strength in signal_strengths:
            # Each strength is tested on the network at rest
            self.network.reset()

            # Create input pattern with given strength
            input_pattern = [strength] * len(self.network.input_region)
            input_sequence = [input_pattern] * duration_per_signal
            
            # Run simulation
            outputs = self.network.run_simulation(steps=duration_per_signal, input_patterns=input_sequence)
            
            # Check if output exceeded detection threshold (0.5)
            max_output = max([max(output) for output in outputs])
            detected = max_output > 0.5
            
            detection_results.append((strength, max_output, detected))
        
        return detection_results
    
    def analog_memory(self, input_values, retention_time=30):
        """
        Demonstrate analog memory capabilities.
        
        Parameters:
        - input_values: Input values to store
        - retention_time: How long to check retention
        
        Returns:
        - memory_trace: How values are retained over time

        Each value is stored in the network reset to rest.
        """
        memory_traces = []
        
        for value in input_values:
            # Each value is stored in the network at rest
            self.network.reset()

            # Set input
            input_pattern = [value] * len(self.network.input_region)
            
            # Apply input for 5 steps then remove
            input_sequence = [input_pattern] * 5 + [np.zeros(len(self.network.input_region))] * (retention_time - 5)
            
            # Run simulation
            outputs = self.network.run_simulation(steps=retention_time, input_patterns=input_sequence)
            
            # Track output over time
            output_trace = [max(output) for output in outputs]
            memory_traces.append((value, output_trace))
        
        return memory_traces

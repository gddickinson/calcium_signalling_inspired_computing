"""Smoke tests for threshold computing module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_unit_init():
    from threshold_computing.threshold_computing import ThresholdComputingUnit
    unit = ThresholdComputingUnit(threshold=0.7)
    assert unit.threshold == 0.7
    assert unit.activation == 0.0
    assert unit.membrane_potential == 0.0


def test_unit_update():
    from threshold_computing.threshold_computing import ThresholdComputingUnit
    unit = ThresholdComputingUnit(threshold=0.7)
    activation = unit.update(1.0)
    assert isinstance(activation, float)
    assert 0.0 <= activation <= 1.0


def test_network_init():
    from threshold_computing.threshold_computing import ThresholdComputingNetwork
    np.random.seed(42)
    net = ThresholdComputingNetwork(size=(5, 5), connectivity_radius=2, connectivity_prob=0.3)
    assert net.height == 5
    assert net.width == 5


def test_network_simulation():
    from threshold_computing.threshold_computing import ThresholdComputingNetwork
    np.random.seed(42)
    net = ThresholdComputingNetwork(size=(5, 5), connectivity_radius=2, connectivity_prob=0.3)
    input_pattern = np.random.random(len(net.input_region)) * 0.8
    net.set_input(input_pattern)
    outputs = net.run_simulation(steps=5)
    assert len(outputs) == 5


if __name__ == "__main__":
    test_unit_init()
    test_unit_update()
    test_network_init()
    test_network_simulation()
    print("All threshold computing tests passed.")

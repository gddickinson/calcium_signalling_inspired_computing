"""Smoke tests for coordinated recruitment module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_element_init():
    from coordinated_recruitment.element import CoordinatedRecruitmentElement
    elem = CoordinatedRecruitmentElement(sensitivity=0.5, recruitment_threshold=0.3)
    assert elem.activation == 0.0
    assert elem.fatigue == 0.0


def test_element_update():
    from coordinated_recruitment.element import CoordinatedRecruitmentElement
    elem = CoordinatedRecruitmentElement(sensitivity=0.5, recruitment_threshold=0.3)
    signal = elem.update(direct_input=1.0, neighborhood_activation=0.5)
    assert isinstance(signal, float)


def test_network_init():
    from coordinated_recruitment.network import CoordinatedRecruitmentNetwork
    np.random.seed(42)
    net = CoordinatedRecruitmentNetwork(grid_size=(10, 10), coupling_radius=2)
    assert net.height == 10
    assert net.width == 10
    assert net.elements.shape == (10, 10)


def test_network_simulation():
    from coordinated_recruitment.network import CoordinatedRecruitmentNetwork
    np.random.seed(42)
    net = CoordinatedRecruitmentNetwork(grid_size=(10, 10), coupling_radius=2)
    net.set_input_region(center=(5, 5), radius=3, strength=0.8)
    results = net.run_simulation(steps=5)
    assert 'total_activation' in results
    assert len(results['total_activation']) == 5


if __name__ == "__main__":
    test_element_init()
    test_element_update()
    test_network_init()
    test_network_simulation()
    print("All coordinated recruitment tests passed.")

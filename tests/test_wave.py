"""Smoke tests for wave-based computing module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_wave_medium_init():
    from wave_based_computing.wave_medium import WaveComputingMedium
    medium = WaveComputingMedium(size=(20, 20))
    assert medium.width == 20
    assert medium.height == 20
    assert medium.current.shape == (20, 20)


def test_wave_medium_add_source():
    from wave_based_computing.wave_medium import WaveComputingMedium
    medium = WaveComputingMedium(size=(20, 20))
    medium.add_source((10, 10), amplitude=0.5, frequency=0.8)
    assert len(medium.sources) == 1


def test_wave_medium_update():
    from wave_based_computing.wave_medium import WaveComputingMedium
    np.random.seed(42)
    medium = WaveComputingMedium(size=(20, 20))
    medium.add_source((10, 10), amplitude=0.5, frequency=0.8)
    medium.update(0.0)
    assert len(medium.history) == 1
    assert len(medium.energy_history) == 1


def test_wave_medium_simulation():
    from wave_based_computing.wave_medium import WaveComputingMedium
    np.random.seed(42)
    medium = WaveComputingMedium(size=(20, 20))
    medium.add_source((10, 10), amplitude=0.5, frequency=0.8)
    frames = medium.run_simulation(steps=10, visualize=True)
    assert frames is not None
    assert len(frames) == 2  # Every 5th frame: frames 0 and 5


def test_wave_memory_element():
    from wave_based_computing.wave_medium import WaveComputingMedium
    medium = WaveComputingMedium(size=(20, 20))
    medium.set_memory_element((5, 5), 0.75)
    assert medium.read_memory_element((5, 5)) == 0.75
    assert medium.read_memory_element((0, 0)) == 0.0


if __name__ == "__main__":
    test_wave_medium_init()
    test_wave_medium_add_source()
    test_wave_medium_update()
    test_wave_medium_simulation()
    test_wave_memory_element()
    print("All wave computing tests passed.")

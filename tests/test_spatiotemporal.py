"""Smoke tests for spatiotemporal computing module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_model_init():
    from spatiotemporal_computing.spatiotemporal_computing import SpatiotemporalComputing
    np.random.seed(42)
    model = SpatiotemporalComputing(grid_size=20, n_channels=3)
    assert model.grid_size == 20
    assert model.n_channels == 3
    assert model.state.shape == (3, 20, 20)


def test_model_update():
    from spatiotemporal_computing.spatiotemporal_computing import SpatiotemporalComputing
    np.random.seed(42)
    model = SpatiotemporalComputing(grid_size=20, n_channels=3)
    model.update()
    assert len(model.history) == 1


def test_model_simulation():
    from spatiotemporal_computing.spatiotemporal_computing import SpatiotemporalComputing
    np.random.seed(42)
    model = SpatiotemporalComputing(grid_size=20, n_channels=3)
    results = model.run_simulation(steps=10)
    assert 'spatial_analysis' in results
    assert 'temporal_analysis' in results


if __name__ == "__main__":
    test_model_init()
    test_model_update()
    test_model_simulation()
    print("All spatiotemporal computing tests passed.")

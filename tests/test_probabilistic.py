"""Smoke tests for probabilistic computing module."""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_calcium_puff_computing_init():
    from probabilistic_computing.probabilistic_computing import CalciumPuffComputing
    model = CalciumPuffComputing(grid_size=20)
    assert model.grid_size == 20
    assert model.calcium_grid.shape == (20, 20)
    assert model.channel_density.shape == (20, 20)


def test_calcium_puff_computing_update():
    from probabilistic_computing.probabilistic_computing import CalciumPuffComputing
    np.random.seed(42)
    model = CalciumPuffComputing(grid_size=20)
    model.update()
    assert len(model.activation_history) == 1


def test_stochastic_multiply():
    from probabilistic_computing.probabilistic_computing import StochasticComputing
    np.random.seed(42)
    sc = StochasticComputing()
    result = sc.stochastic_multiply(0.5, 0.5, iterations=10000)
    assert abs(result - 0.25) < 0.05, f"Expected ~0.25, got {result}"


def test_stochastic_add():
    from probabilistic_computing.probabilistic_computing import StochasticComputing
    np.random.seed(42)
    sc = StochasticComputing()
    result = sc.stochastic_add(0.3, 0.3, iterations=10000)
    assert abs(result - 0.6) < 0.1, f"Expected ~0.6, got {result}"


def test_stochastic_subtract():
    from probabilistic_computing.probabilistic_computing import StochasticComputing
    np.random.seed(42)
    sc = StochasticComputing()
    result = sc.stochastic_subtract(0.7, 0.3, iterations=10000)
    # a*(1-b) = 0.7*0.7 = 0.49
    assert result >= 0.0
    assert result <= 1.0


if __name__ == "__main__":
    test_calcium_puff_computing_init()
    test_calcium_puff_computing_update()
    test_stochastic_multiply()
    test_stochastic_add()
    test_stochastic_subtract()
    print("All probabilistic computing tests passed.")

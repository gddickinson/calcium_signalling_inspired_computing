"""Regression tests for bugs fixed in the model code (one test per fix; see SESSION_LOG.md)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- probabilistic_computing: subtraction and division ----------------------------------------------------------------

def test_stochastic_subtract_computes_a_minus_b():
    """a AND NOT b on independent streams gave a*(1-b) (0.49 for 0.7 - 0.3), not max(0, a - b)."""
    from probabilistic_computing.probabilistic_computing import StochasticComputing
    np.random.seed(0)
    sc = StochasticComputing()
    assert abs(sc.stochastic_subtract(0.7, 0.3, iterations=20000) - 0.4) < 0.02
    assert sc.stochastic_subtract(0.2, 0.6, iterations=20000) == 0.0


def test_stochastic_divide_computes_a_over_b():
    """P(a | b) on independent streams is just a (0.3 for 0.3 / 0.6), not min(1, a / b)."""
    from probabilistic_computing.probabilistic_computing import StochasticComputing
    np.random.seed(0)
    sc = StochasticComputing()
    assert abs(sc.stochastic_divide(0.3, 0.6, iterations=20000) - 0.5) < 0.02
    assert sc.stochastic_divide(0.8, 0.4, iterations=20000) == 1.0


# --- wave_based_computing: grid shape and the wave equation -------------------------------------------------------------

def test_wave_medium_non_square_grid():
    """size=(width, height) made arrays of shape (width, height) but indexed them [y, x]: IndexError if width != height."""
    from wave_based_computing.wave_medium import WaveComputingMedium
    medium = WaveComputingMedium(size=(30, 20))
    assert medium.current.shape == (20, 30)
    medium.add_source((25, 10), amplitude=0.5, frequency=0.8)
    medium.run_simulation(steps=12)
    assert medium.current[10, 25] != 0


# --- threshold_computing --------------------------------------------------------------------------------------------

def test_threshold_unit_zero_refractory_period():
    """refractory_period=0 divided by zero on the first update."""
    from threshold_computing.threshold_computing import ThresholdComputingUnit
    unit = ThresholdComputingUnit(refractory_period=0)
    for _ in range(3):
        a = unit.update(2.0)
    assert 0.0 <= a <= 1.0


def test_threshold_trials_start_from_rest():
    """threshold_detection / analog_memory ran every trial on the state the previous trial left behind, so a result
    depended on which strengths were tested before it."""
    from threshold_computing.threshold_computing import ThresholdComputingNetwork, ThresholdLogicComputer

    def make():
        np.random.seed(3)
        return ThresholdLogicComputer(ThresholdComputingNetwork(size=(6, 6), connectivity_radius=2))

    alone = make().threshold_detection([0.5])[0]
    after = make().threshold_detection([1.0, 0.5])[1]
    assert alone == pytest.approx(after)
    alone = make().analog_memory([0.5], retention_time=12)[0][1]
    after = make().analog_memory([0.8, 0.5], retention_time=12)[1][1]
    np.testing.assert_allclose(after, alone)


# --- spatiotemporal_computing ----------------------------------------------------------------------------------------

def test_pattern_recognition_uses_only_its_own_history():
    """train() and recognize() kept appending to the model history, so the dynamics graph of the second pattern was
    built from the first pattern's run as well."""
    from spatiotemporal_computing.spatiotemporal_computing import (PatternRecognitionComputer,
                                                                   SpatiotemporalComputing)
    np.random.seed(1)
    model = SpatiotemporalComputing(grid_size=20)
    rec = PatternRecognitionComputer(model)
    a = np.zeros((3, 20, 20)); a[0, 5:10, 5:10] = 1
    b = np.zeros((3, 20, 20)); b[1, 8:12, :] = 0.8
    rec.train({'a': a, 'b': b})
    assert len(model.history) == 20
    rec.recognize(a, steps=15)
    assert len(model.history) == 15


# --- multistore_computing ---------------------------------------------------------------------------------------------

def test_multistore_accepts_numpy_scalars():
    """numpy integers (and float32) are not int/float: write raised TypeError and query silently used [0.5, 0.4, 0.3]."""
    from multistore_computing.computer import MultiStoreComputer
    np.random.seed(0)
    c = MultiStoreComputer()
    for s in c.stores.values():
        s.reliability = 1.0
    ok, name = c.write_data(np.int64(1), priority='high')
    assert ok and name == 'primary'
    assert c.stores['primary'].contents.max() == 1.0
    ok, _ = c.write_data(np.float32(0.25), priority='medium')
    assert ok
    best = c.query(np.int64(1))[0]
    np.testing.assert_allclose(best['result'], [1.0, 0.8, 0.5])

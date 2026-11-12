"""Wave medium physics: a real second-order wave equation with a CFL-stable internal time step.

The old update overwrote `previous` before using it and used the neighbour mean as the Laplacian, so it was a
first-order smoothing that kept activity at the sources (see SESSION_LOG.md, 2026-11-11).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wave_based_computing.wave_medium import WaveComputingMedium  # noqa: E402


def bump(medium, x, y, width=2.0):
    yy, xx = np.mgrid[0:medium.height, 0:medium.width]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * width ** 2))
    medium.current[:] = g
    medium.previous[:] = g          # released at rest


def test_time_step_respects_cfl():
    """Each output step is split into substeps with c * dt_sub / dx below the stability limit of the stencil."""
    for c in (0.5, 10.0, 40.0):
        m = WaveComputingMedium(size=(20, 20), wave_speed=c, time_step=0.1)
        assert m.substeps >= 1
        assert m.dt_sub * m.substeps == np.float64(0.1) or abs(m.dt_sub * m.substeps - 0.1) < 1e-12
        assert c * m.dt_sub * np.sqrt(m.LAMBDA_MAX) <= 2.0 * m.CFL_SAFETY + 1e-12


def test_energy_conserved_without_damping():
    """Periodic grid, no damping, no nonlinearity, no sources: the leapfrog energy is conserved."""
    m = WaveComputingMedium(size=(40, 40), boundary='periodic', damping=1.0, nonlinearity=0.0)
    bump(m, 20, 20)
    m.update(0.0)
    e0 = m.wave_energy()
    for k in range(1, 200):
        m.update(k * m.time_step)
    assert abs(m.wave_energy() - e0) / e0 < 1e-9


def test_energy_decays_with_damping():
    m = WaveComputingMedium(size=(40, 40), boundary='periodic', damping=0.98, nonlinearity=0.0)
    bump(m, 20, 20)
    m.update(0.0)
    energies = [m.wave_energy()]
    for k in range(1, 100):
        m.update(k * m.time_step)
        energies.append(m.wave_energy())
    assert all(b <= a * (1 + 1e-12) for a, b in zip(energies, energies[1:]))
    assert energies[-1] < 0.5 * energies[0]


def test_pulse_travels_at_wave_speed_and_oscillates():
    m = WaveComputingMedium(size=(81, 81), boundary='absorbing', damping=1.0, nonlinearity=0.0, wave_speed=10.0)
    bump(m, 40, 40, width=1.5)
    arrival, centre = None, []
    for k in range(40):
        m.update(k * m.time_step)
        centre.append(m.current[40, 40])
        if arrival is None and abs(m.current[40, 70]) > 0.01:
            arrival = (k + 1) * m.time_step
    assert min(centre) < 0                                   # second order in time: the centre swings through zero
    assert arrival is not None and 2.0 < arrival < 3.6       # 30 cells at 10 cells per time unit -> about 3


def test_stable_at_defaults_with_sources():
    """The demo set-up (two sources, obstacle, resonant box) stays bounded over a long run."""
    m = WaveComputingMedium(size=(100, 100), boundary='absorbing')
    m.add_source((20, 50), amplitude=0.5, frequency=0.8)
    m.add_source((80, 50), amplitude=0.5, frequency=0.8, phase=np.pi)
    m.add_obstacle(45, 55, 40, 60)
    m.add_resonant_region(70, 90, 70, 90, frequency=0.8)
    m.run_simulation(steps=400)
    assert np.isfinite(m.current).all()
    assert np.abs(m.current).max() < 2.0
    assert np.abs(m.current[50, 30:40]).max() > 0.05        # waves leave the sources
    assert m.current[40:60, 45:55].max() == 0                # obstacle held at zero


def test_reflecting_and_periodic_boundaries_stable():
    for b in ('reflecting', 'periodic'):
        m = WaveComputingMedium(size=(30, 40), boundary=b)
        m.add_source((10, 10), amplitude=0.5, frequency=0.6)
        m.run_simulation(steps=300)
        assert np.isfinite(m.current).all() and np.abs(m.current).max() < 3.0

"""WaveComputing.pattern_recognition tells its reference patterns apart and rejects a pattern it was not given.

Before 2026-11-11 the encoding kept only pixels strictly above the pattern's 80th percentile, which for the demo's
clean circle, square and diagonal is none at all, so every reference medium was empty; and the PCA-statistics
signature scored about 0.46 for every pair.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wave_based_computing import WaveComputing, WaveComputingMedium  # noqa: E402


def references(size=(20, 20)):
    circle = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            if np.hypot(i - 10, j - 10) < 6:
                circle[i, j] = 1.0
    square = np.zeros(size); square[5:-5, 5:-5] = 1.0
    diagonal = np.zeros(size)
    for i in range(size[0]):
        for d in (-1, 0, 1):
            if 0 <= i + d < size[1]:
                diagonal[i, i + d] = 0.8
    return [circle, square, diagonal]


def novel(size=(20, 20)):
    cross = np.zeros(size); cross[9:11, :] = 1.0; cross[:, 9:11] = 1.0
    corner = np.zeros(size); corner[1:6, 13:19] = 1.0
    return [cross, corner]


@pytest.fixture(scope='module')
def computer():
    return WaveComputing(WaveComputingMedium(size=(100, 100)))


@pytest.mark.parametrize('steps', [50, 100])
def test_references_recognised_and_novel_rejected(computer, steps):
    np.random.seed(5)
    refs = references()
    tests = [p + np.random.normal(0, 0.1, p.shape) for p in refs + novel()]
    results = computer.pattern_recognition(tests, refs, simulation_steps=steps)
    assert [r['best_match'] for r in results] == [0, 1, 2, None, None], \
        [[round(m['similarity'], 3) for m in r['matches']] for r in results]
    for r in results[:3]:
        assert r['matches'][0]['similarity'] > 0.95
    for r in results[3:]:
        assert r['matches'][0]['similarity'] < 0.8


def test_clean_reference_has_sources(computer):
    m = WaveComputingMedium(size=(100, 100))
    computer._encode_pattern_as_sources(m, references()[0])
    assert len(m.sources) > 50

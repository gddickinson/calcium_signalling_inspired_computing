"""PatternRecognitionComputer recognises its stored patterns from noisy copies and rejects patterns it never stored.

Before 2026-11-11 each pattern got its own PCA + k-means, so cluster labels (and the transition-graph edges compared
between patterns) meant different things in every run; the demo matched 1 of 3 patterns.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatiotemporal_computing.spatiotemporal_computing import (PatternRecognitionComputer,  # noqa: E402
                                                               SpatiotemporalComputing)

N = 50


def stored():
    p1 = np.zeros((3, N, N)); p1[0, 20:30, 20:30] = 1.0
    p2 = np.zeros((3, N, N)); p2[1, 20:30, :] = 0.8
    p3 = np.zeros((3, N, N)); p3[2, :, 20:30] = 0.8
    return {'central_pulse': p1, 'horizontal_stripe': p2, 'vertical_stripe': p3}


def novel():
    corner = np.zeros((3, N, N)); corner[0, 5:15, 35:45] = 1.0
    diagonal = np.zeros((3, N, N))
    for i in range(N):
        diagonal[1, i, max(0, i - 3):i + 3] = 0.8
    return {'corner_pulse': corner, 'diagonal_stripe': diagonal}


@pytest.fixture(scope='module', params=[42, 7, 2026])
def trained(request):
    np.random.seed(request.param)
    rec = PatternRecognitionComputer(SpatiotemporalComputing(grid_size=N))
    rec.train(stored())
    return rec


def test_stored_patterns_recognised(trained):
    for name, p in stored().items():
        match, score = trained.recognize(p + np.random.normal(0, 0.2, p.shape))
        assert match == name, (name, match, trained.last_distances)
        assert score > 0.5


def test_unstored_patterns_rejected(trained):
    for name, p in novel().items():
        match, score = trained.recognize(p + np.random.normal(0, 0.2, p.shape))
        assert match is None, (name, match, trained.last_distances)
        assert score < 0.5

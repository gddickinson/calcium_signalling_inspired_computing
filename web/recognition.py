"""Spatiotemporal pattern recognition (PatternRecognitionComputer) run with the demo's patterns from
spatiotemporal_computing.py's __main__, seeded, for the web page: noisy copies of the three stored patterns and of a
pattern that was never stored. Precomputed at build time (it is cheap, but keeps the page independent of Python)."""
from functools import lru_cache

import numpy as np

from spatiotemporal_computing import PatternRecognitionComputer, SpatiotemporalComputing

SEED, GRID, NOISE = 42, 50, 0.2


def demo_patterns(n=GRID):
    """The three reference patterns of the demo: central pulse (channel 0), horizontal and vertical stripes."""
    p1 = np.zeros((3, n, n)); p1[0, 20:30, 20:30] = 1.0
    p2 = np.zeros((3, n, n)); p2[1, 20:30, :] = 0.8
    p3 = np.zeros((3, n, n)); p3[2, :, 20:30] = 0.8
    return {"central_pulse": p1, "horizontal_stripe": p2, "vertical_stripe": p3}


def unstored_patterns(n=GRID):
    corner = np.zeros((3, n, n)); corner[0, 5:15, 35:45] = 1.0
    diagonal = np.zeros((3, n, n))
    for i in range(n):
        diagonal[1, i, max(0, i - 3):i + 3] = 0.8
    return {"corner_pulse": corner, "diagonal_stripe": diagonal}


@lru_cache(maxsize=1)
def recognition():
    np.random.seed(SEED)
    rec = PatternRecognitionComputer(SpatiotemporalComputing(grid_size=GRID, n_channels=3))
    stored = demo_patterns()
    rec.train(stored)
    tests = []
    for kind, pats in (("stored", stored), ("not stored", unstored_patterns())):
        for name, p in pats.items():
            match, score = rec.recognize(p + np.random.normal(0, NOISE, p.shape))
            expected = name if kind == "stored" else None
            tests.append({"input": name, "kind": kind, "match": match, "score": score, "correct": match == expected,
                          "distances": {k: float(v) for k, v in rec.last_distances.items()}})
    return {"seed": SEED, "grid": GRID, "noise_sd": NOISE, "block": rec.block, "frames": rec.signature_frames,
            "n_noisy": rec.n_noisy, "radius_factor": rec.radius_factor,
            "radii": {k: float(v) for k, v in rec.radii.items()}, "tests": tests,
            "correct": sum(t["correct"] for t in tests)}

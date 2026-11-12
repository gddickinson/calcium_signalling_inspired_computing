"""Spatiotemporal pattern recognition (PatternRecognitionComputer, scikit-learn PCA + KMeans) run with the demo's
patterns from spatiotemporal_computing.py's __main__, seeded, for the web page. Python only: precomputed at build time."""
from functools import lru_cache

import numpy as np

from spatiotemporal_computing.spatiotemporal_computing import PatternRecognitionComputer, SpatiotemporalComputing

SEED, GRID, NOISE = 42, 50, 0.2


def demo_patterns(n=GRID):
    """The three reference patterns of the demo: central pulse (channel 0), horizontal and vertical stripes."""
    p1 = np.zeros((3, n, n)); p1[0, 20:30, 20:30] = 1.0
    p2 = np.zeros((3, n, n)); p2[1, 20:30, :] = 0.8
    p3 = np.zeros((3, n, n)); p3[2, :, 20:30] = 0.8
    return {"central_pulse": p1, "horizontal_stripe": p2, "vertical_stripe": p3}


@lru_cache(maxsize=1)
def recognition():
    np.random.seed(SEED)
    model = SpatiotemporalComputing(grid_size=GRID, n_channels=3)
    rec = PatternRecognitionComputer(model)
    patterns = demo_patterns()
    rec.train(patterns)
    graphs = {k: sorted([int(a), int(b)] for a, b in g.edges()) for k, g in rec.pattern_graphs.items()}
    tests = []
    for name, p in patterns.items():
        noisy = p + np.random.normal(0, NOISE, p.shape)
        match, score = rec.recognize(noisy)
        test_graph = rec._compute_pattern_graph(model.history)
        scores = {ref: float(rec._graph_similarity(test_graph, g)) for ref, g in rec.pattern_graphs.items()}
        tests.append({"input": name, "match": match, "score": float(score), "scores": scores})
    return {"seed": SEED, "grid": GRID, "noise_sd": NOISE, "steps": 20, "graphs": graphs, "tests": tests,
            "correct": sum(t["input"] == t["match"] for t in tests)}

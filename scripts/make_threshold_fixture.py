"""Reference numbers for web/static/threshold_engine.js from threshold_computing (np.random.seed).

Run: python scripts/make_threshold_fixture.py [--write]
"""
import numpy as np

from fixture_common import main

from threshold_computing import ThresholdComputingNetwork, ThresholdLogicComputer

NAME = "threshold"


def run(seed, size, radius, prob):
    np.random.seed(seed)
    net = ThresholdComputingNetwork(size=size, connectivity_radius=radius, connectivity_prob=prob)
    n_conn = sum(len(v) for v in net.connections.values())
    thresholds = [net.units[i, j].threshold for i in range(size[0]) for j in range(size[1])]
    pattern = np.random.random(len(net.input_region)) * 0.8
    net.set_input(pattern)
    free = net.run_simulation(steps=15)
    logic = ThresholdLogicComputer(net)
    weak = [np.random.random(len(net.input_region)) * 0.3 for _ in range(8)]
    integ = logic.temporal_integration(weak, duration=20)
    detect = logic.threshold_detection([0.2, 0.6, 1.0], duration_per_signal=10)
    memory = logic.analog_memory([0.5, 0.9], retention_time=15)
    return {"seed": seed, "size": size, "radius": radius, "prob": prob, "n_conn": n_conn, "thresholds": thresholds,
            "free": free, "integ": integ, "detect": [[s, m, bool(d)] for s, m, d in detect],
            "memory": [[v, t] for v, t in memory], "potentials": net.membrane_potential_history[-1]}


def fixture():
    return {"runs": [run(42, (10, 10), 3, 0.3), run(9, (6, 8), 2, 0.5)]}


if __name__ == "__main__":
    main(NAME, fixture)

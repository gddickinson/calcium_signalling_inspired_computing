"""Reference numbers for web/static/recruit_engine.js from coordinated_recruitment (np.random.seed).

Run: python scripts/make_recruit_fixture.py [--write]
"""
import numpy as np

from fixture_common import main

from coordinated_recruitment import CoordinatedRecruitmentNetwork, QuorumComputer

NAME = "recruit"


def run(seed, size, radius, fault, steps):
    np.random.seed(seed)
    net = CoordinatedRecruitmentNetwork(grid_size=size, coupling_radius=radius, fault_probability=fault)
    net.set_input_region((size[0] // 2, size[1] // 2), radius=5, strength=0.7)
    res = net.run_simulation(steps=steps)
    w = size[1]
    clusters = [[[int(i) * w + int(j) for i, j in c] for c in cs] for cs in net.cluster_history[-3:]]
    q = QuorumComputer(net)
    pat = np.zeros(size)
    pat[3:9, 4:10] = 0.8
    ft = q.fault_tolerant_computation([pat], [0.0, 0.3], trials=2)
    amp = q.signal_amplification([0.3, 0.9], duration=8)
    return {"seed": seed, "size": size, "radius": radius, "fault": fault, "steps": steps,
            "faulty": sorted(i * w + j for i, j in net.faulty_elements),
            "avg_cluster_sizes": res["avg_cluster_sizes"], "num_clusters": res["num_clusters"],
            "total_activation": res["total_activation"], "clusters": clusters, "last": net.activation_history[-1],
            "ft": ft, "amp": [{k: r[k] for k in ("input_strength", "initial_input", "final_activation",
                                                 "amplification_ratio", "time_series")} for r in amp]}


def dbscan_cases():
    """Random binary grids clustered with the network's own _identify_clusters (sklearn DBSCAN eps 1.5, 3 samples)."""
    np.random.seed(123)
    net = CoordinatedRecruitmentNetwork(grid_size=(16, 16), coupling_radius=2)
    cases = []
    for density in (0.25, 0.4, 0.55):
        grid = (np.random.random((16, 16)) < density).astype(float)
        clusters = net._identify_clusters(grid, activation_threshold=0.5)
        cases.append({"grid": grid, "clusters": [[int(i) * 16 + int(j) for i, j in c] for c in clusters]})
    return cases


def fixture():
    return {"runs": [run(42, (20, 20), 3, 0.05, 15), run(3, (14, 18), 2, 0.1, 10)], "dbscan": dbscan_cases()}


if __name__ == "__main__":
    main(NAME, fixture)

"""Reference numbers for web/static/st_engine.js from spatiotemporal_computing.py (np.random.seed).

Run: python scripts/make_st_fixture.py [--write]
"""
import numpy as np

from fixture_common import main

from spatiotemporal_computing.spatiotemporal_computing import SpatiotemporalComputing

NAME = "st"


def run(seed, n, steps):
    np.random.seed(seed)
    m = SpatiotemporalComputing(grid_size=n)
    regions = m.spatial_regions.copy()
    s0 = m.state.copy()
    m.run_simulation(steps=steps)
    corr = np.corrcoef([m.history[-1][i].flatten() for i in range(3)])
    return {"seed": seed, "n": n, "steps": steps, "regions_sum": regions.sum(), "regions_row": regions[n // 2],
            "s0_row": s0[2, n // 3], "rows": [m.state[c, (c + 1) * n // 4] for c in range(3)],
            "sums": [m.state[c].sum() for c in range(3)], "corr": corr}


def fixture():
    return {"runs": [run(42, 50, 12), run(5, 32, 6)]}


if __name__ == "__main__":
    main(NAME, fixture)

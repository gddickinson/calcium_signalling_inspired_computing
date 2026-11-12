"""Reference numbers for web/static/puff_engine.js from probabilistic_computing.py (np.random.seed).

Run: python scripts/make_puff_fixture.py [--write]
"""
import numpy as np

from fixture_common import main, quiet

from probabilistic_computing.probabilistic_computing import CalciumPuffComputing, StochasticComputing

NAME = "puff"


def run(seed, n, steps, decay, threshold):
    np.random.seed(seed)
    m = CalciumPuffComputing(grid_size=n, decay_rate=decay, threshold=threshold)
    dens = m.channel_density.copy()
    with quiet():
        res = m.run_simulation(steps=steps)
    res.pop("fit_line", None)
    sc = StochasticComputing(m)
    ops = {"multiply": sc.stochastic_multiply(0.3, 0.7, 500), "add": sc.stochastic_add(0.3, 0.4, 500),
           "subtract": sc.stochastic_subtract(0.7, 0.3, 500), "divide": sc.stochastic_divide(0.3, 0.6, 500)}
    return {"seed": seed, "n": n, "steps": steps, "decay": decay, "threshold": threshold,
            "density_row": dens[n // 3], "calcium_row": m.calcium_grid[n // 2], "calcium_sum": m.calcium_grid.sum(),
            "last_activations": int(m.activation_history[-1].sum()), "results": res, "ops": ops}


def fixture():
    return {"runs": [run(42, 20, 30, 0.95, 0.7), run(7, 16, 25, 0.9, 0.5)]}


if __name__ == "__main__":
    main(NAME, fixture)

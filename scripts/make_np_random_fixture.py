"""Reference numbers for web/static/np_random.js: numpy's legacy global RNG after np.random.seed.

Run: python scripts/make_np_random_fixture.py [--write]
"""
import numpy as np

from fixture_common import main

NAME = "np_random"


def fixture():
    np.random.seed(42)
    out = {"seed": 42, "random": np.random.random(5).tolist(), "randn": [np.random.randn() for _ in range(5)],
           "normal": np.random.normal(0.5, 0.05, 7).tolist(), "randint": [int(np.random.randint(5, 20)) for _ in range(6)]}
    out["random_after"] = float(np.random.random())          # after an odd number of gaussians (cached value kept)
    out["choice_arr"] = [int(np.random.choice(np.array([3, 8, 13, 21, 34]))) for _ in range(5)]
    out["choice_noreplace"] = np.random.choice([4, 9, 16, 25, 36, 49], size=3, replace=False).tolist()
    return out


if __name__ == "__main__":
    main(NAME, fixture)

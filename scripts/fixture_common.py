"""Helpers shared by the scripts/make_*_fixture.py scripts.

Each fixture script exposes ``NAME`` and ``fixture()`` (a JSON-able dict computed by the repo's own model classes
after np.random.seed, so the JS engines, which use the same numpy MT19937 stream from np_random.js, can replay it).
``main`` prints it or writes tests/fixtures/<NAME>.json; tests/test_web.py recomputes every fixture and checks the
committed file still matches (numbers to rtol 1e-9, so Linux CI passes).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"


def quiet():
    """Context manager silencing the scripts' progress prints."""
    import contextlib
    import io
    return contextlib.redirect_stdout(io.StringIO())


def to_json(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_json(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return str(x)            # JSON has no inf/nan: "inf", "-inf", "nan"
    return x


def close(a, b, rtol=1e-9, atol=1e-12, path="$"):
    """Recursive comparison with a numeric tolerance; returns a list of mismatch descriptions."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(set(a) ^ set(b))}"]
        return [m for k in a for m in close(a[k], b[k], rtol, atol, f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in close(x, y, rtol, atol, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if abs(a - b) <= atol + rtol * max(abs(a), abs(b)) else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def main(name, fixture):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=f"write tests/fixtures/{name}.json")
    data = to_json(fixture())
    text = json.dumps(data, separators=(",", ":"))
    if ap.parse_args().write:
        FIXTURES.mkdir(parents=True, exist_ok=True)
        (FIXTURES / f"{name}.json").write_text(text + "\n")
        print(f"wrote {FIXTURES / (name + '.json')} ({len(text)} bytes)")
    else:
        print(text)


os.environ.setdefault("MPLBACKEND", "Agg")

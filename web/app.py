"""FastAPI web app for Calcium-Inspired Computing: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8144, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.
The six models run in the browser (static/*_engine.js, ports of the Python classes checked against
tests/fixtures/*.json); the API serves the repository's own text, class defaults and the Python-only
pattern-recognition run.
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.readme import ROOT, bullets, models, section
from web.recognition import recognition

STATIC = Path(__file__).resolve().parent / "static"
FIXTURES = ROOT / "tests" / "fixtures"

app = FastAPI(title="Calcium-Inspired Computing")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# JS engine -> (fixture, Python source it reproduces)
ENGINES = {
    "puff_engine.js": ("puff", "probabilistic_computing/probabilistic_computing.py"),
    "st_engine.js": ("st", "spatiotemporal_computing/spatiotemporal_computing.py"),
    "threshold_engine.js": ("threshold", "threshold_computing/threshold_computing.py, threshold_logic.py"),
    "recruit_engine.js": ("recruit", "coordinated_recruitment/element.py, network.py, quorum.py"),
    "store_engine.js": ("store", "multistore_computing/store.py, computer.py, hybrid.py"),
    "wave_engine.js": ("wave", "wave_based_computing/wave_medium.py, wave_computing.py"),
    "np_random.js": ("np_random", "numpy legacy RandomState (np.random.seed)"),
}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Title, README overview and counts for the home page."""
    ms = models()
    return {"title": "Calcium-Inspired Computing",
            "subtitle": "Novel Computational Architectures Inspired by Calcium Signaling",
            "overview": section("Overview"), "models": len(ms),
            "classes": sum(len(m["classes"]) for m in ms), "engines": len(ENGINES) - 1,
            "fixtures": len(list(FIXTURES.glob("*.json")))}


@app.get("/api/models")
def model_list():
    """One entry per model: README title, intro, key features, demo command, module docstring, class defaults."""
    return models()


@app.get("/api/overview")
def overview():
    """README sections used on the home and about pages."""
    return {"overview": section("Overview"), "applications": bullets(section("Applications")),
            "background": bullets(section("Research Background")), "license": section("License"),
            "acknowledgments": bullets(section("Acknowledgments").split("---")[0])}


@app.get("/api/checks")
def checks():
    """Which JS engine reproduces which Python code, and the size of each reference fixture."""
    out = []
    for js, (fx, py) in ENGINES.items():
        f = FIXTURES / f"{fx}.json"
        out.append({"engine": js, "python": py, "fixture": f"tests/fixtures/{fx}.json",
                    "bytes": f.stat().st_size if f.exists() else 0,
                    "runs": len(json.loads(f.read_text()).get("runs", [])) if f.exists() else 0})
    return out


@app.get("/api/recognition")
def pattern_recognition():
    """PatternRecognitionComputer on the demo's three patterns (seeded; scikit-learn, so Python only)."""
    return recognition()


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/models", "/overview", "/checks", "/recognition"]

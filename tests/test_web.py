"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific ---------------------------------------------------------------------------------------------

FIXTURE_SCRIPTS = sorted((ROOT / "scripts").glob("make_*_fixture.py"))


@pytest.mark.parametrize("script", FIXTURE_SCRIPTS, ids=lambda p: p.stem)
def test_js_fixture_is_current(script):
    """tests/fixtures/<name>.json must match what the current Python model code produces (regenerate with
    python scripts/make_<name>_fixture.py --write). Numbers compared with rtol 1e-9: libm differs across platforms."""
    import importlib.util
    import json
    sys.path.insert(0, str(ROOT / "scripts"))
    import fixture_common
    spec = importlib.util.spec_from_file_location(script.stem, script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    now = fixture_common.to_json(mod.fixture())
    saved = json.loads((ROOT / "tests" / "fixtures" / f"{mod.NAME}.json").read_text())
    assert fixture_common.close(now, saved) == []


def test_every_engine_has_a_fixture_and_test():
    from web.app import ENGINES
    for js, (fx, _) in ENGINES.items():
        assert (STATIC / js).exists(), js
        assert (STATIC / js.replace(".js", ".test.mjs")).exists(), js
        assert (ROOT / "tests" / "fixtures" / f"{fx}.json").exists(), fx


def test_models_come_from_readme_and_code():
    rows = client.get("/api/models").json()
    assert [r["number"] for r in rows] == [1, 2, 3, 4, 5, 6]
    for r in rows:
        assert r["title"] and r["intro"] and r["features"], r["route"]
        assert (ROOT / r["script"]).exists()
    waves = next(r for r in rows if r["route"] == "waves")
    assert waves["classes"][0]["defaults"]["diffusion"] == 0.2


def test_recognition_is_reported_honestly():
    r = client.get("/api/recognition").json()
    assert len(r["tests"]) == 3
    assert r["correct"] == sum(t["input"] == t["match"] for t in r["tests"])


def test_pages_registered():
    main = (STATIC / "main.js").read_text()
    for m in client.get("/api/models").json():
        assert f"'{m['route']}'" in main and (STATIC / f"{m['route']}.js").exists(), m["route"]

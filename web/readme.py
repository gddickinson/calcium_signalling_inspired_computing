"""Text for the web pages taken from the repository itself: README sections, module docstrings and constructor
defaults of the model classes (so the site's descriptions and slider defaults come from the code, not from the web)."""
import ast
import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Web route -> (README model number, package folder, demo script, model classes as "module:Class")
MODELS = {
    "puffs": (1, "probabilistic_computing", "probabilistic_computing/probabilistic_computing.py",
              ["probabilistic_computing.probabilistic_computing:CalciumPuffComputing",
               "probabilistic_computing.probabilistic_computing:StochasticComputing"]),
    "patterns": (2, "spatiotemporal_computing", "spatiotemporal_computing/spatiotemporal_computing.py",
                 ["spatiotemporal_computing.spatiotemporal_computing:SpatiotemporalComputing",
                  "spatiotemporal_computing.pattern_recognition:PatternRecognitionComputer"]),
    "threshold": (3, "threshold_computing", "threshold_computing/threshold_computing.py",
                  ["threshold_computing.threshold_computing:ThresholdComputingUnit",
                   "threshold_computing.threshold_computing:ThresholdComputingNetwork",
                   "threshold_computing.threshold_logic:ThresholdLogicComputer"]),
    "recruitment": (4, "coordinated_recruitment", "coordinated_recruitment/coordinated_recruitment.py",
                    ["coordinated_recruitment.element:CoordinatedRecruitmentElement",
                     "coordinated_recruitment.network:CoordinatedRecruitmentNetwork",
                     "coordinated_recruitment.quorum:QuorumComputer"]),
    "stores": (5, "multistore_computing", "multistore_computing/multistore_computing.py",
               ["multistore_computing.store:CalciumStore", "multistore_computing.computer:MultiStoreComputer",
                "multistore_computing.hybrid:HybridComputing"]),
    "waves": (6, "wave_based_computing", "wave_based_computing/wave_based_computing.py",
              ["wave_based_computing.wave_medium:WaveComputingMedium",
               "wave_based_computing.wave_computing:WaveComputing"]),
}


def readme_text() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def section(name: str) -> str:
    """Body of a level-2 README section (without the heading)."""
    m = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", readme_text(), re.S | re.M)
    return m.group(1).strip() if m else ""


def bullets(text: str) -> list:
    return [re.sub(r"\*\*(.+?)\*\*", r"\1", line[2:].strip()) for line in text.splitlines() if line.startswith("- ")]


def model_sections() -> dict:
    """README '### N. Title' blocks: title, intro paragraph, key features, run command."""
    out = {}
    for m in re.finditer(r"^### (\d+)\. (.+?)\n(.*?)(?=^### |^## |\Z)", section("Models") + "\n", re.S | re.M):
        body = m.group(3)
        intro = body.split("**Key Features:**")[0].strip()
        feats = bullets(body.split("**Key Features:**")[1]) if "**Key Features:**" in body else []
        run = re.search(r"```bash\n(.+?)\n```", body)
        out[int(m.group(1))] = {"title": m.group(2).strip(), "intro": intro, "features": feats,
                                "run": run.group(1).strip() if run else ""}
    return out


def module_doc(path: str) -> str:
    """A demo script's module docstring (without the note on where the classes live)."""
    doc = ast.get_docstring(ast.parse((ROOT / path).read_text(encoding="utf-8"))) or ""
    return doc.split("The classes live in the package modules")[0].strip()


def class_defaults(spec: str) -> dict:
    """Constructor keyword defaults of a model class, read with inspect (JSON-safe values only)."""
    mod, cls = spec.split(":")
    obj = getattr(__import__(mod, fromlist=[cls]), cls)
    out = {}
    for k, p in inspect.signature(obj.__init__).parameters.items():
        if k == "self" or p.default is inspect.Parameter.empty:
            continue
        v = p.default
        out[k] = list(v) if isinstance(v, tuple) else v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)
    return out


def models() -> list:
    secs = model_sections()
    rows = []
    for route, (num, pkg, script, classes) in MODELS.items():
        s = secs.get(num, {})
        rows.append({"route": route, "number": num, "package": pkg, "script": script, **s,
                     "run": f"python {script}", "docstring": module_doc(script),
                     "classes": [{"name": c.split(":")[1], "module": c.split(":")[0], "defaults": class_defaults(c)}
                                 for c in classes]})
    return rows

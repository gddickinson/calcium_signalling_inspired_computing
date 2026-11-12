"""Web interface for the calcium-signalling-inspired computing models.

Contents:
- app.py: FastAPI app (front end in static/, JSON API under /api, STATIC_ENDPOINTS for the GitHub Pages export)
- readme.py: README sections, module docstrings and class defaults used as the site's text
- recognition.py: Python-only extra (spatiotemporal pattern recognition uses scikit-learn), precomputed for the site
- static/: ES-module front end; *_engine.js are JS ports of the models, node-tested against tests/fixtures/*.json
"""

"""Streamlit Cloud entry point for AirSense AI.

The repository also has an ``app/`` package for the FastAPI and Streamlit
modules. Defining ``__path__`` keeps ``import app.api`` working when this file
is imported as the top-level ``app`` module during tests.
"""

from __future__ import annotations

import runpy
from pathlib import Path


APP_DIR = Path(__file__).parent / "app"
__path__ = [str(APP_DIR)]


if __name__ == "__main__":
    runpy.run_path(str(APP_DIR / "streamlit_app.py"), run_name="__main__")

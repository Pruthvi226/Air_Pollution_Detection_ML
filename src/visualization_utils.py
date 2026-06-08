"""Visualization helpers with optional Plotly support."""

from __future__ import annotations

import pandas as pd


def make_bar_chart(frame: pd.DataFrame, x: str, y: str, title: str = ""):
    """Return a Plotly bar chart when Plotly is available; otherwise return None."""
    try:
        import plotly.express as px
    except ImportError:
        return None
    return px.bar(frame, x=x, y=y, title=title)


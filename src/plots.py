"""One place for chart styling, so every figure in the repository looks like the same study.

Deliberately small. The charts themselves are built in the notebooks next to the sentence that
interprets them, because a chart without its sentence is decoration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from .data import ROOT

FIGURES = ROOT / "reports" / "figures"

# One colour per role, used consistently: the baseline is the reference line everything else is
# read against, so it keeps the neutral colour and the models get the saturated ones.
COLORS = {
    "naive": "#000000",
    "drift": "#7f7f7f",
    "seasonal_naive": "#9467bd",
    "arima": "#1f77b4",
    "sarima": "#17becf",
    "xgboost_returns": "#d62728",
    "xgboost_returns_tuned": "#e377c2",
    "xgboost_levels": "#ff9896",
    "SGD_per_USD": "#1f77b4",
    "CNY_per_USD": "#d62728",
}


def use_style() -> None:
    """Apply the project's matplotlib defaults. Call once per notebook."""
    mpl.rcParams.update(
        {
            "figure.figsize": (11, 4.2),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )


def save(fig, name: str) -> Path:
    """Save a figure into reports/figures/ and return the path."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    return path


def color(key: str, default: str = "#333333") -> str:
    return COLORS.get(key, default)

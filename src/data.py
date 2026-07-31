"""Loading and cleaning for the FX dataset.

One place owns the data, so the notebooks and the backtest harness can never disagree about
what "the series" means. Every cleaning decision below is a decision, not a default, and the
reason is stated next to it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repository root, resolved from this file so the code runs from any working directory.
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "Foreign_Exchange_Rates.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# The source column names, kept verbatim so the mapping below is auditable.
SOURCE_COLUMNS = {
    "SINGAPORE - SINGAPORE DOLLAR/US$": "SGD_per_USD",
    "CHINA - YUAN/US$": "CNY_per_USD",
}

# The string the source uses for a non-trading day. It is not a number, so it silently turns
# the whole column into dtype object if it is not declared here.
MISSING_TOKEN = "ND"

SERIES = list(SOURCE_COLUMNS.values())


def load_raw() -> pd.DataFrame:
    """Read the source file exactly as it is, with no cleaning at all.

    Everything stays a string so the audit notebook can count the defects itself instead of
    being handed a frame where pandas has already hidden them.
    """
    return pd.read_csv(RAW_FILE, dtype=str)


def load_clean(fill_holidays: bool = True) -> pd.DataFrame:
    """Return the analysis frame: a business-day DatetimeIndex and two float columns.

    Three decisions are baked in here.

    1. ``ND`` becomes NaN rather than being dropped. Dropping the row would remove a valid
       trading day from the *other* currency, because the two series do not share holidays.
    2. The index is the date, sorted ascending, with no reindexing to a calendar frequency.
       The series is a business-day series; inserting weekend rows would invent observations.
    3. Holiday gaps are forward-filled when ``fill_holidays`` is True. For a price series that
       is the honest fill: on a day the market did not trade, the last traded price is the last
       known price. Interpolation would invent a trade that never happened, which for a
       forecasting benchmark is leakage from the future into a gap.

    Pass ``fill_holidays=False`` to get the NaNs back, which is what the audit notebook uses to
    show what the fill actually changed.
    """
    df = pd.read_csv(
        RAW_FILE,
        na_values=[MISSING_TOKEN],
        parse_dates=["DATE"],
    )
    df = df.rename(columns={"DATE": "date", **SOURCE_COLUMNS})
    df = df.set_index("date").sort_index()
    df = df[SERIES].astype("float64")

    if fill_holidays:
        df = df.ffill()
        # A leading NaN cannot be forward-filled. Both series start on a trading day here, so
        # this is a guard rather than an expected case.
        df = df.dropna()

    # The file turns out to carry every single weekday in the range with no omissions, so the
    # index is exactly a business-day range and the frequency can be declared. This is not
    # cosmetic: statsmodels uses the declared frequency to date its forecasts, and without it
    # every fit emits a warning and falls back to integer positions.
    expected = pd.bdate_range(df.index.min(), df.index.max())
    if df.index.equals(expected):
        df.index.freq = "B"

    return df


def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns. The first row is dropped because it has no predecessor."""
    import numpy as np

    return np.log(df).diff().dropna()


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Write a frame to data/processed/ and return the path."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_csv(path)
    return path

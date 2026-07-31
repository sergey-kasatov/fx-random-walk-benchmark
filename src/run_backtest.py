"""Run the full walk-forward comparison and write the results to disk.

    python src/run_backtest.py

This is the expensive step, so it is a script rather than a notebook cell: it runs once, writes
its output, and the notebooks read that output. Everything it does is deterministic, so a rerun
reproduces the same numbers.

Written outputs
---------------
``data/processed/predictions.csv``   every forecast made, one row per origin and horizon
``reports/metrics.csv``              the scored comparison table
``reports/persistence_similarity.csv``  how far each model moved away from the naive forecast
``reports/arima_orders.csv``         the AIC grid used to choose the ARIMA order, per series
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import evaluate, persistence_similarity, walk_forward  # noqa: E402
from src.data import PROCESSED_DIR, ROOT, load_clean  # noqa: E402
from src.models import (  # noqa: E402
    ArimaForecaster,
    DriftForecaster,
    NaiveForecaster,
    SarimaForecaster,
    SeasonalNaiveForecaster,
    XGBForecaster,
    select_arima_order,
)

TEST_START = "2016-01-01"
HORIZONS = (1, 5, 21)
REFIT_EVERY = 63  # about one quarter of trading days
REPORTS = ROOT / "reports"


def main() -> int:
    started = time.perf_counter()
    df = load_clean()
    train = df.loc[:pd.Timestamp(TEST_START) - pd.Timedelta(days=1)]

    print(f"data      {df.index.min().date()} to {df.index.max().date()}, {len(df)} business days")
    print(f"train     up to {train.index.max().date()} ({len(train)} rows)")
    print(f"test      from {TEST_START}, horizons {HORIZONS}, refit every {REFIT_EVERY}\n")

    # ---- ARIMA order selection, on the training window only -------------------------------
    orders: dict[str, tuple[int, int, int]] = {}
    grids = []
    for column in df.columns:
        order, grid = select_arima_order(train[column])
        orders[column] = order
        grid.insert(0, "series", column)
        grids.append(grid)
        print(f"ARIMA order for {column}: {order} (chosen by AIC on the training window)", flush=True)
    pd.concat(grids, ignore_index=True).to_csv(REPORTS / "arima_orders.csv", index=False)
    print()

    # ---- The comparison -------------------------------------------------------------------
    frames = []
    for column in df.columns:
        print(f"{column}:", flush=True)
        order = orders[column]
        factories = {
            "naive": NaiveForecaster,
            "drift": DriftForecaster,
            "seasonal_naive": SeasonalNaiveForecaster,
            "arima": lambda o=order: ArimaForecaster(order=o),
            "sarima": lambda o=order: SarimaForecaster(order=o, seasonal_order=(1, 0, 1, 5)),
            "xgboost_returns": lambda: XGBForecaster(mode="returns"),
            "xgboost_levels": lambda: XGBForecaster(mode="levels"),
        }
        for factory in factories.values():
            frames.append(
                walk_forward(
                    df,
                    column,
                    factory(),
                    horizons=HORIZONS,
                    test_start=TEST_START,
                    refit_every=REFIT_EVERY,
                    verbose=True,
                )
            )

    predictions = pd.concat(frames, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PROCESSED_DIR / "predictions.csv", index=False)

    metrics = evaluate(predictions)
    metrics.to_csv(REPORTS / "metrics.csv", index=False)

    similarity = pd.concat(
        [
            persistence_similarity(
                predictions[predictions["series"] == column], df[column]
            )
            for column in df.columns
        ],
        ignore_index=True,
    )
    similarity.to_csv(REPORTS / "persistence_similarity.csv", index=False)

    print(f"\nwrote {len(predictions):,} forecasts")
    print(f"total {time.perf_counter() - started:.0f}s\n")
    print(metrics.round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

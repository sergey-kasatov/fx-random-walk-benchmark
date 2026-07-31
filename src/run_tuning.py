"""Tune XGBoost inside the training window, then backtest the tuned model on the same origins.

    python src/run_tuning.py

Answers the objection the main result invites: "your gradient boosting lost because you did not
tune it." The search runs on data ending in 2015, so the chosen configuration is one that could
have been picked before the backtest window began, and the tuned model is then scored by exactly
the same harness as everything else.

Written outputs
---------------
``reports/xgb_tuning_grid.csv``       every configuration tried, with its time-series CV score
``reports/xgb_tuning_best.csv``       the winner per series, and the naive forecast's CV score
``data/processed/predictions_tuned.csv``  the tuned model's walk-forward forecasts
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import walk_forward  # noqa: E402
from src.data import PROCESSED_DIR, ROOT, load_clean  # noqa: E402
from src.models import XGBForecaster  # noqa: E402
from src.tuning import naive_cv_benchmark, search_xgb  # noqa: E402

TEST_START = "2016-01-01"
HORIZONS = (1, 5, 21)
REFIT_EVERY = 63
REPORTS = ROOT / "reports"


class TunedXGBForecaster(XGBForecaster):
    """XGBoost with the searched parameters, named so it is distinguishable in the results."""

    @property
    def name(self) -> str:
        return f"xgboost_{self.mode}_tuned"


def main() -> int:
    started = time.perf_counter()
    df = load_clean()
    train = df.loc[: pd.Timestamp(TEST_START) - pd.Timedelta(days=1)]

    grids, winners, frames = [], [], []

    for column in df.columns:
        print(f"searching {column} on {len(train):,} training rows...", flush=True)
        best, grid = search_xgb(train, column, horizon=1, mode="returns")
        baseline_cv = naive_cv_benchmark(train, column, horizon=1)

        grid.insert(0, "series", column)
        grids.append(grid)
        winners.append({
            "series": column,
            **best,
            "best_cv_rmse": grid["cv_rmse"].min(),
            "naive_cv_rmse": baseline_cv,
            "beats_naive_in_cv": bool(grid["cv_rmse"].min() < baseline_cv),
        })
        print(f"  best: {best}", flush=True)
        print(f"  CV RMSE {grid['cv_rmse'].min():.6f} against the naive forecast's "
              f"{baseline_cv:.6f}", flush=True)

        frames.append(
            walk_forward(
                df,
                column,
                TunedXGBForecaster(mode="returns", **best),
                horizons=HORIZONS,
                test_start=TEST_START,
                refit_every=REFIT_EVERY,
                verbose=True,
            )
        )

    pd.concat(grids, ignore_index=True).to_csv(REPORTS / "xgb_tuning_grid.csv", index=False)
    pd.DataFrame(winners).to_csv(REPORTS / "xgb_tuning_best.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(
        PROCESSED_DIR / "predictions_tuned.csv", index=False
    )

    print(f"\ntotal {time.perf_counter() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

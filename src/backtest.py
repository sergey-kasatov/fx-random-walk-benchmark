"""Walk-forward backtest harness.

The comparison is the product of this repository, so the procedure that produces it lives in
one importable place rather than being retyped per model in a notebook. Every model is scored
on identical forecast origins, identical targets and identical horizons, which is the only way
a difference in RMSE can be attributed to the model rather than to the setup.

The design decisions, each of which is a way this could have been done wrong:

*Expanding window, never a random split.* A shuffled train/test split on a time series trains
on the future. Here the training set is always a prefix of the data and the forecast origin
always moves forward.

*Forecast at every trading day, not once.* A single train/test split gives one draw from the
distribution of possible results. A thousand origins gives an error distribution, which is what
the significance test then needs.

*Parameters re-estimated on a schedule, state updated daily.* Refitting ARIMA at every one of a
thousand origins is hours of compute for a result that does not move. Refitting quarterly while
feeding new observations in daily is the standard compromise, and ``refit_every`` makes it a
visible number rather than a hidden habit.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd

from . import metrics as M


def walk_forward(
    df: pd.DataFrame,
    target: str,
    forecaster,
    horizons: Sequence[int] = (1, 5, 21),
    test_start: str | pd.Timestamp = "2016-01-01",
    refit_every: int = 63,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run one forecaster over one series and return every forecast it made.

    Returns a long frame with one row per (origin, horizon):

    ``origin``      the last day the model was allowed to see
    ``horizon``     steps ahead
    ``target_date`` the day being forecast
    ``y_true``      what actually happened
    ``y_pred``      what the model said
    ``model``       forecaster name
    ``series``      target column name
    """
    horizons = list(horizons)
    max_h = max(horizons)
    n = len(df)

    test_start = pd.Timestamp(test_start)
    origins = np.where(df.index >= test_start)[0]
    # An origin is only usable if the longest horizon still lands inside the data.
    origins = [i for i in origins if i + max_h < n]
    if not origins:
        raise ValueError("no usable forecast origins: check test_start against the data range")

    records = []
    started = time.perf_counter()
    fits = 0

    for step, i in enumerate(origins):
        history = df.iloc[: i + 1]

        needs_refit = (step % refit_every == 0) if forecaster.expensive else True
        if needs_refit or not hasattr(forecaster, "history"):
            forecaster.fit(history, target, horizons)
            fits += 1
        else:
            forecaster.update(history)

        for h in horizons:
            records.append(
                {
                    "origin": df.index[i],
                    "horizon": h,
                    "target_date": df.index[i + h],
                    "y_true": float(df[target].iloc[i + h]),
                    "y_pred": forecaster.forecast(h),
                    "model": forecaster.name,
                    "series": target,
                }
            )

    if verbose:
        elapsed = time.perf_counter() - started
        # flush, because this loop is the long pole of the whole project and a reader watching a
        # redirected log should not have to wait an hour to learn that it started.
        print(
            f"  {forecaster.name:<18} {target:<12} "
            f"{len(origins)} origins, {fits} fits, {elapsed:5.1f}s",
            flush=True,
        )

    return pd.DataFrame.from_records(records)


def run_all(
    df: pd.DataFrame,
    forecaster_factories: dict,
    series: Sequence[str] | None = None,
    horizons: Sequence[int] = (1, 5, 21),
    test_start: str = "2016-01-01",
    refit_every: int = 63,
    verbose: bool = True,
) -> pd.DataFrame:
    """Every model against every series, stacked into one frame.

    ``forecaster_factories`` maps a label to a zero-argument callable returning a fresh
    forecaster. A factory rather than an instance, because a fitted model must never be reused
    across series.
    """
    series = list(series or df.columns)
    frames = []
    for col in series:
        if verbose:
            print(f"{col}:", flush=True)
        for factory in forecaster_factories.values():
            frames.append(
                walk_forward(
                    df,
                    col,
                    factory(),
                    horizons=horizons,
                    test_start=test_start,
                    refit_every=refit_every,
                    verbose=verbose,
                )
            )
    return pd.concat(frames, ignore_index=True)


def evaluate(
    predictions: pd.DataFrame,
    baseline: str = "naive",
) -> pd.DataFrame:
    """Score the predictions frame against the baseline, per series and horizon.

    Adds three things a plain RMSE table does not have:

    ``rmse_ratio``  model RMSE divided by baseline RMSE. Above 1.0 means worse than doing
                    nothing, which is the result this harness exists to be able to report
    ``skill_pct``   the same number restated as a percentage of baseline error removed
    ``dm_p``        Diebold-Mariano p-value against the baseline. Without it, a ratio of 0.98
                    is indistinguishable from luck
    """
    rows = []
    for (series, horizon), block in predictions.groupby(["series", "horizon"], sort=True):
        base = block[block["model"] == baseline]
        if base.empty:
            raise ValueError(f"baseline '{baseline}' missing for {series} h={horizon}")
        base = base.set_index("origin").sort_index()
        base_errors = (base["y_true"] - base["y_pred"]).to_numpy()
        base_rmse = M.rmse(base["y_true"], base["y_pred"])

        for model, part in block.groupby("model", sort=True):
            part = part.set_index("origin").sort_index()
            if not part.index.equals(base.index):
                raise ValueError(f"origins differ between {model} and {baseline}")

            errors = (part["y_true"] - part["y_pred"]).to_numpy()
            row = {
                "series": series,
                "horizon": horizon,
                "model": model,
                "n": len(part),
                **M.summarize(part["y_true"], part["y_pred"]),
            }
            row["rmse_ratio"] = row["rmse"] / base_rmse
            row["skill_pct"] = 100 * M.skill_score(row["rmse"], base_rmse)

            if model == baseline:
                row["dm_stat"], row["dm_p"] = np.nan, np.nan
            else:
                # errors_a is the model, errors_b the baseline: a negative statistic means the
                # model had the smaller loss.
                row["dm_stat"], row["dm_p"] = M.diebold_mariano(
                    errors, base_errors, horizon=horizon
                )
            rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["series", "horizon", "rmse"]).reset_index(drop=True)


def persistence_similarity(predictions: pd.DataFrame, levels: pd.Series) -> pd.DataFrame:
    """How much of each forecast is just the last observed value.

    The headline claim of this repository is that a model can post a good RMSE by reproducing
    the naive forecast. That claim needs its own measurement rather than an assertion, so this
    reports, per model and horizon, the mean absolute distance between the model's forecast and
    the naive forecast, expressed as a fraction of the mean absolute distance between the naive
    forecast and the truth.

    A value near 0 means the model said almost exactly what persistence said. A value near or
    above 1 means it took a genuinely different position, which may be better or worse.
    """
    rows = []
    for (series, horizon, model), part in predictions.groupby(
        ["series", "horizon", "model"], sort=True
    ):
        part = part.sort_values("origin")
        anchor = levels.loc[part["origin"]].to_numpy() if isinstance(levels, pd.Series) else None
        if anchor is None:
            continue
        gap_to_naive = np.mean(np.abs(part["y_pred"].to_numpy() - anchor))
        naive_gap_to_truth = np.mean(np.abs(part["y_true"].to_numpy() - anchor))
        rows.append(
            {
                "series": series,
                "horizon": horizon,
                "model": model,
                "mean_move_from_last": gap_to_naive,
                "naive_mean_error": naive_gap_to_truth,
                "move_ratio": gap_to_naive / naive_gap_to_truth,
            }
        )
    return pd.DataFrame(rows).sort_values(["series", "horizon", "move_ratio"]).reset_index(drop=True)

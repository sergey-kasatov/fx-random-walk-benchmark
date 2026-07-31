"""Leak-free supervised framing of a time series for the tree models.

The single rule this module exists to enforce: every feature attached to row ``t`` uses only
information that existed at the close of day ``t``, and the target is what happened ``horizon``
days later. Everything is built with ``shift`` and backward-looking ``rolling`` windows, and
the frame is assembled once so a notebook cannot quietly add a centred rolling mean.

Two framings are offered on purpose, because the difference between them is a finding rather
than a preference:

``levels``
    Features and target are the rate itself. This is the framing most tutorials use and it is
    the one that manufactures a flattering RMSE: with the last observed rate in the feature
    set, a tree learns to echo it, which is the naive forecast in an expensive wrapper. It is
    also structurally broken for a trending series, because a tree cannot predict a value
    outside the range it was trained on.

``returns``
    Features and target are log returns; the level forecast is reconstructed afterwards as
    ``y_t * exp(predicted cumulative return)``. This is the defensible framing: the model is
    asked the question that is actually hard, which is where the series goes next, rather than
    the question that is trivial, which is where it is now.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAGS = (0, 1, 2, 3, 5, 10, 21)
DEFAULT_WINDOWS = (5, 21, 63)


def _add_lag_block(out: pd.DataFrame, source: pd.Series, prefix: str, lags, windows) -> None:
    """Add lag and backward-rolling features for one series, in place.

    The rule these follow, stated once because the whole comparison depends on it: a row dated
    ``t`` may use any function of observations up to and including ``t``, because the forecast
    is made at the close of ``t``. So ``lag0`` is today's own closing value and a rolling window
    ending today is legitimate, while anything that reaches to ``t+1`` is not. No feature here
    is centred, and none uses a negative shift.
    """
    for lag in lags:
        # shift(0) is a copy; it is written this way so the lag list needs no special case and
        # the intent of "lag 0 means today, which is known" stays visible.
        out[f"{prefix}_lag{lag}"] = source.shift(lag)

    for window in windows:
        # min_periods equal to the window so a partial window never produces a value that
        # looks like a full one. No shift: the window ends today and today is observed.
        roll = source.rolling(window=window, min_periods=window)
        out[f"{prefix}_mean{window}"] = roll.mean()
        out[f"{prefix}_std{window}"] = roll.std()


def make_supervised(
    df: pd.DataFrame,
    target: str,
    horizon: int,
    mode: str = "returns",
    lags=DEFAULT_LAGS,
    windows=DEFAULT_WINDOWS,
    use_cross_series: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build ``(X, y, anchor)`` for one target series at one horizon.

    ``df``
        Clean level series, business-day index, one column per currency.
    ``target``
        Column name to forecast.
    ``horizon``
        Steps ahead. 1 is next trading day, 5 is about a week, 21 about a month.
    ``mode``
        ``"returns"`` or ``"levels"``. See the module docstring.
    ``use_cross_series``
        Add the other currency's lagged returns as features. The brief asks whether the two
        series move together, so the honest way to answer it is to let the model try to use
        one to forecast the other.

    Returns
    -------
    X : feature frame, indexed by the date the forecast is *made*
    y : target, aligned to X. Cumulative log return over the horizon in ``returns`` mode,
        the level in ``levels`` mode
    anchor : the last observed level at forecast time, needed to turn a predicted return back
        into a predicted rate. Also the naive forecast, which is why it is returned rather
        than recomputed elsewhere
    """
    if mode not in {"returns", "levels"}:
        raise ValueError("mode must be 'returns' or 'levels'")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    levels = df[target]
    out = pd.DataFrame(index=df.index)

    if mode == "returns":
        rets = np.log(df).diff()
        _add_lag_block(out, rets[target], "ret", lags, windows)
        if use_cross_series:
            for other in df.columns:
                if other != target:
                    _add_lag_block(out, rets[other], f"x_{other}", lags, windows)
        # Cumulative log return from t to t+horizon, known only at t+horizon.
        y = np.log(levels.shift(-horizon)) - np.log(levels)
    else:
        _add_lag_block(out, levels, "lvl", lags, windows)
        if use_cross_series:
            for other in df.columns:
                if other != target:
                    _add_lag_block(out, df[other], f"x_{other}", lags, windows)
        y = levels.shift(-horizon)

    # Calendar features. Cheap, leak-free, and they are the honest test of the brief's
    # "seasonality detection" question: if day of week carries signal, the model can use it.
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month

    anchor = levels.rename("anchor")

    frame = out.join(y.rename("__target__")).join(anchor)
    frame = frame.dropna()

    X = frame.drop(columns=["__target__", "anchor"])
    return X, frame["__target__"], frame["anchor"]

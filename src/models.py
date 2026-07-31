"""The forecasters, behind one interface so the backtest can treat them identically.

Every forecaster answers the same three calls:

``fit(history, target, horizons)``
    Estimate parameters using only the rows in ``history``, which the harness guarantees end at
    the day the forecast is made.
``update(history)``
    Take on new observations without re-estimating parameters. This is what makes a daily
    walk-forward affordable: the state advances every day, the parameters are re-estimated on a
    schedule.
``forecast(horizon)``
    One number, the predicted level ``horizon`` business days after the end of the history.

Everything returns a *level*, never a return, so the comparison table is in the units the brief
asks for and no model gets to be scored on an easier quantity than another.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from .features import make_supervised


class Forecaster:
    """Interface. Subclasses override fit and forecast."""

    name = "base"
    #: Whether re-estimation is expensive enough that the harness should honour refit_every.
    expensive = False

    def fit(self, history: pd.DataFrame, target: str, horizons: Sequence[int]) -> "Forecaster":
        self.history = history
        self.target = target
        self.horizons = list(horizons)
        return self

    def update(self, history: pd.DataFrame) -> None:
        """Advance the state by the rows appended since the last call."""
        self.history = history

    def forecast(self, horizon: int) -> float:
        raise NotImplementedError


class NaiveForecaster(Forecaster):
    """The random walk: tomorrow's rate is today's rate, at every horizon.

    This is the thing to beat. It is also exactly ARIMA(0,1,0), which is worth knowing before
    reading any ARIMA result: a fitted ARIMA that lands near this one has not found anything.
    """

    name = "naive"

    def forecast(self, horizon: int) -> float:
        return float(self.history[self.target].iloc[-1])


class DriftForecaster(Forecaster):
    """Random walk with drift: extend the average slope of the whole history.

    Included because it is the cheapest possible way to be wrong in an interesting direction.
    If a currency has trended for years, a model that ignores the trend should at least be
    compared against one that assumes it continues.
    """

    name = "drift"

    def forecast(self, horizon: int) -> float:
        series = self.history[self.target]
        last = float(series.iloc[-1])
        slope = (last - float(series.iloc[0])) / (len(series) - 1)
        return last + slope * horizon


class SeasonalNaiveForecaster(Forecaster):
    """The value from the same weekday one week ago.

    The brief asks about weekly seasonality. This is the direct test of it: if there is a
    weekly pattern in a daily FX rate, this should beat the plain random walk. It will not,
    and that negative is the answer to the seasonality question.
    """

    name = "seasonal_naive"

    def forecast(self, horizon: int) -> float:
        series = self.history[self.target]
        return float(series.iloc[-5]) if len(series) >= 5 else float(series.iloc[-1])


class ArimaForecaster(Forecaster):
    """statsmodels ARIMA on the level series.

    Fitted on levels with ``d=1`` rather than on returns, so the forecast comes back in the
    units of the comparison without a reconstruction step that could hide an error.
    """

    name = "arima"
    expensive = True

    def __init__(self, order=(1, 1, 1), trend=None):
        self.order = order
        self.trend = trend

    def fit(self, history, target, horizons):
        from statsmodels.tsa.arima.model import ARIMA

        super().fit(history, target, horizons)
        with warnings.catch_warnings():
            # Convergence chatter on a near random walk is expected and says nothing useful
            # here; the parameter estimates are inspected in the notebook instead.
            warnings.simplefilter("ignore")
            self._res = ARIMA(
                history[target],
                order=self.order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
        self._n_fitted = len(history)
        self._path = None
        return self

    def update(self, history):
        new_rows = history[self.target].iloc[self._n_fitted :]
        if len(new_rows) == 0:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # refit=False keeps the parameters and only advances the state, which is the whole
            # point: parameters come from the last scheduled refit, information comes from
            # every day.
            self._res = self._res.append(new_rows, refit=False)
        self._n_fitted = len(history)
        self.history = history
        self._path = None

    def forecast(self, horizon: int) -> float:
        # The forecast path to the longest horizon is produced once per state and then indexed,
        # rather than calling the (not cheap) forecast routine once per horizon. Same numbers,
        # about a third of the runtime.
        if self._path is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._path = self._res.forecast(steps=max(self.horizons)).to_numpy()
        return float(self._path[horizon - 1])


class SarimaForecaster(ArimaForecaster):
    """SARIMAX with a weekly seasonal term (period 5, one trading week).

    The brief names SARIMA explicitly. Daily FX has no calendar season to speak of, so this is
    run as a hypothesis to be rejected on evidence rather than skipped on opinion.
    """

    name = "sarima"
    expensive = True

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 5)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, history, target, horizons):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        Forecaster.fit(self, history, target, horizons)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._res = SARIMAX(
                history[target],
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        self._n_fitted = len(history)
        self._path = None
        return self


class XGBForecaster(Forecaster):
    """Gradient boosting on lag features, one booster per horizon.

    ``mode="returns"`` predicts the cumulative log return and reconstructs the level from the
    last observed rate. ``mode="levels"`` predicts the rate directly. Both are run, because the
    gap between them is the demonstration the repository is named after: the levels model
    posts the better RMSE while having learned nothing except how to repeat its own input.
    """

    expensive = True

    def __init__(self, mode: str = "returns", n_estimators: int = 300, max_depth: int = 4,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42):
        self.mode = mode
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=4,
            tree_method="hist",
            objective="reg:squarederror",
        )

    @property
    def name(self) -> str:
        return f"xgboost_{self.mode}"

    def fit(self, history, target, horizons):
        from xgboost import XGBRegressor

        super().fit(history, target, horizons)
        self._models = {}
        self._feature_names = {}
        for h in self.horizons:
            X, y, _ = make_supervised(history, target, horizon=h, mode=self.mode)
            model = XGBRegressor(**self.params)
            model.fit(X, y)
            self._models[h] = model
            self._feature_names[h] = list(X.columns)
        self._row_cache = None
        return self

    def update(self, history):
        super().update(history)
        self._row_cache = None

    def forecast(self, horizon: int) -> float:
        # Rebuild the feature frame on the current history and take its last complete row: that
        # row is dated today and contains only lagged information, so it is the legitimate
        # input for a forecast of today + horizon. The row does not depend on the horizon, only
        # the booster does, so it is built once per origin.
        if getattr(self, "_row_cache", None) is None:
            self._row_cache = _feature_row_for_today(self.history, self.target, self.mode)
        X_all = self._row_cache[self._feature_names[horizon]]
        pred = float(self._models[horizon].predict(X_all.to_numpy())[0])

        if self.mode == "returns":
            last_level = float(self.history[self.target].iloc[-1])
            return last_level * float(np.exp(pred))
        return pred


def _feature_row_for_today(history: pd.DataFrame, target: str, mode: str) -> pd.DataFrame:
    """The one feature row dated at the end of ``history``.

    ``make_supervised`` drops rows whose target is unknown, and the target for today is exactly
    that. So the row is rebuilt here from the same helpers rather than taken from that frame,
    which is why the feature construction lives in one function that both paths call.
    """
    from .features import DEFAULT_LAGS, DEFAULT_WINDOWS, _add_lag_block

    out = pd.DataFrame(index=history.index)
    if mode == "returns":
        rets = np.log(history).diff()
        _add_lag_block(out, rets[target], "ret", DEFAULT_LAGS, DEFAULT_WINDOWS)
        for other in history.columns:
            if other != target:
                _add_lag_block(out, rets[other], f"x_{other}", DEFAULT_LAGS, DEFAULT_WINDOWS)
    else:
        _add_lag_block(out, history[target], "lvl", DEFAULT_LAGS, DEFAULT_WINDOWS)
        for other in history.columns:
            if other != target:
                _add_lag_block(out, history[other], f"x_{other}", DEFAULT_LAGS, DEFAULT_WINDOWS)

    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    return out.iloc[[-1]]


def select_arima_order(series: pd.Series, p_range=(0, 1, 2), q_range=(0, 1, 2), d: int = 1):
    """Pick (p, d, q) by AIC on the training series only.

    Order selection is a modelling decision made from data, so it has to happen inside the
    training window. Running it on the full series would be a quiet form of leakage: the test
    period would have voted on the model that is then scored against it.
    """
    from statsmodels.tsa.arima.model import ARIMA

    best = None
    rows = []
    for p in p_range:
        for q in q_range:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(
                        series,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
                rows.append({"p": p, "d": d, "q": q, "aic": res.aic, "bic": res.bic})
                if best is None or res.aic < best[1]:
                    best = ((p, d, q), res.aic)
            except Exception as exc:  # a non-converging order is information, not a crash
                rows.append({"p": p, "d": d, "q": q, "aic": np.nan, "bic": np.nan,
                             "error": type(exc).__name__})
    return best[0], pd.DataFrame(rows)

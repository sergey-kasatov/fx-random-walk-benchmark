"""Forecast error metrics, and the test that says whether a difference in them is real.

The point of the repository is a comparison, so every metric here is either reported against
the baseline or is a test of the gap between two forecasts. An absolute RMSE on a near random
walk carries almost no information on its own.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(e**2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(e)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent. Safe here: FX rates are never zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def skill_score(model_error: float, baseline_error: float) -> float:
    """Fraction of the baseline error removed by the model.

    Positive means better than the baseline, negative means worse, zero means identical.
    Reported alongside the raw ratio because a ratio of 1.02 reads as "about the same" until
    it is restated as "2 percent worse".
    """
    return float(1.0 - model_error / baseline_error)


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    horizon: int = 1,
    power: int = 2,
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy, with the Harvey correction.

    Answers the question the RMSE table cannot: is model A's error genuinely different from
    model B's, or is the gap inside the noise of this particular test window?

    The loss differential ``d_t = |e_a|^power - |e_b|^power`` is autocorrelated up to lag
    ``horizon - 1`` for an h-step forecast, so the variance of its mean uses that many
    autocovariance terms rather than assuming independence. The Harvey, Leybourne and Newbold
    (1997) small-sample correction is applied, because without it the test rejects too often on
    a few hundred observations.

    Returns ``(statistic, p_value)`` for the two-sided test. A negative statistic means A had
    the smaller loss. The null is that both forecasts are equally accurate.
    """
    e_a = np.asarray(errors_a, dtype=float)
    e_b = np.asarray(errors_b, dtype=float)
    if e_a.shape != e_b.shape:
        raise ValueError("error series must have the same length")

    d = np.abs(e_a) ** power - np.abs(e_b) ** power
    n = d.size
    d_bar = d.mean()

    # Long-run variance of the mean: gamma_0 plus twice the autocovariances that an h-step
    # forecast is expected to carry.
    gamma_0 = np.sum((d - d_bar) ** 2) / n
    gamma = [gamma_0]
    for lag in range(1, horizon):
        cov = np.sum((d[lag:] - d_bar) * (d[:-lag] - d_bar)) / n
        gamma.append(cov)
    var_d_bar = (gamma[0] + 2 * sum(gamma[1:])) / n

    if var_d_bar <= 0:
        # Happens when the two forecasts are identical, or on a degenerate window.
        return float("nan"), float("nan")

    dm = d_bar / np.sqrt(var_d_bar)

    # Harvey-Leybourne-Newbold correction, then compare against t rather than the normal.
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_corrected = dm * correction
    p_value = 2 * (1 - stats.t.cdf(abs(dm_corrected), df=n - 1))

    return float(dm_corrected), float(p_value)


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """The three error metrics as one row."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
    }

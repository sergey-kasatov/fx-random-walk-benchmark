"""Tests that describe the series, as opposed to metrics that score a forecast.

These sit apart from ``metrics.py`` on purpose. A metric answers "how wrong was this model". The
functions here answer "what kind of process is this", which is the question that decides whether
any model was ever going to help.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def adf_table(df: pd.DataFrame, transform: str = "levels") -> pd.DataFrame:
    """Augmented Dickey-Fuller test on each column.

    The null hypothesis is a unit root, which in this context is exactly "this series is a
    random walk". Failing to reject it is not proof of a random walk, but on a price series it
    is the expected result and it sets the bar every model in this project has to clear.
    """
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for column in df.columns:
        series = df[column].dropna()
        if transform == "log_returns":
            series = np.log(series).diff().dropna()
        stat, p_value, lags, nobs, crit, _ = adfuller(series, autolag="AIC")
        rows.append(
            {
                "series": column,
                "transform": transform,
                "adf_stat": stat,
                "p_value": p_value,
                "lags_used": lags,
                "n": nobs,
                "crit_1pct": crit["1%"],
                "crit_5pct": crit["5%"],
                "unit_root_rejected_5pct": p_value < 0.05,
            }
        )
    return pd.DataFrame(rows)


def variance_ratio(series: pd.Series, q: int, use_log: bool = True) -> dict:
    """Lo and MacKinlay variance ratio test, heteroskedasticity-robust.

    The most direct test of the random walk hypothesis available. Under a random walk the
    variance of a q-period return is q times the variance of a one-period return, so the ratio
    VR(q) = Var(q-period) / (q * Var(1-period)) equals 1.

    VR above 1 means returns are positively autocorrelated, so moves persist and a trend-
    following model has something to find. VR below 1 means mean reversion. A VR statistically
    indistinguishable from 1 means there is no linear structure at that horizon for any model to
    exploit.

    The robust statistic ``z2`` is the one to read: FX returns are strongly heteroskedastic, and
    the homoskedastic version rejects the null far too often on volatility clustering alone.
    """
    x = np.log(series.dropna()) if use_log else series.dropna()
    x = np.asarray(x, dtype=float)
    n = len(x) - 1
    mu = (x[-1] - x[0]) / n

    # Variance of one-period differences.
    diffs = np.diff(x)
    var_1 = np.sum((diffs - mu) ** 2) / (n - 1)

    # Variance of q-period differences, overlapping, with the Lo-MacKinlay bias correction.
    m = q * (n - q + 1) * (1 - q / n)
    q_diffs = x[q:] - x[:-q]
    var_q = np.sum((q_diffs - q * mu) ** 2) / m

    vr = var_q / var_1

    # Heteroskedasticity-robust standard error. The n in delta and the sqrt(n) in the statistic
    # are both part of Lo and MacKinlay's definition and neither is optional: delta carries the
    # factor n so that theta is O(1), and the statistic then scales by sqrt(n) to converge to a
    # standard normal. Dropping the second one leaves every z near zero and turns the test into
    # an unconditional "cannot reject", which is a comfortable answer arrived at by accident.
    theta = 0.0
    for j in range(1, q):
        num = np.sum((diffs[j:] - mu) ** 2 * (diffs[: n - j] - mu) ** 2)
        den = np.sum((diffs - mu) ** 2) ** 2
        delta = n * num / den
        theta += (2 * (q - j) / q) ** 2 * delta

    z2 = np.sqrt(n) * (vr - 1) / np.sqrt(theta) if theta > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(z2))) if np.isfinite(z2) else np.nan

    return {
        "q": q,
        "variance_ratio": float(vr),
        "z_robust": float(z2),
        "p_value": float(p_value),
        "random_walk_rejected_5pct": bool(np.isfinite(p_value) and p_value < 0.05),
    }


def variance_ratio_table(df: pd.DataFrame, horizons=(2, 5, 10, 21)) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        for q in horizons:
            row = variance_ratio(df[column], q)
            row["series"] = column
            rows.append(row)
    return pd.DataFrame(rows)[
        ["series", "q", "variance_ratio", "z_robust", "p_value", "random_walk_rejected_5pct"]
    ]


def ljung_box(series: pd.Series, lags: int = 10, squared: bool = False) -> pd.DataFrame:
    """Ljung-Box test for autocorrelation up to ``lags``.

    Run twice in this project and the pair is the point: on returns it asks whether the
    direction of tomorrow's move is predictable from past moves, and on squared returns it asks
    whether its *size* is. On FX the first is usually no and the second is emphatically yes,
    which is the difference between forecasting a rate and forecasting risk.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    values = series.dropna()
    if squared:
        values = values**2
    out = acorr_ljungbox(values, lags=[lags], return_df=True)
    out.index.name = "lag"
    return out.reset_index()


def dayofweek_effect(returns: pd.Series) -> pd.DataFrame:
    """Mean return by weekday, with a t-test of each against zero.

    The brief asks for seasonality detection. This is the direct version of that question for a
    daily series: is any weekday systematically different from zero? Reported with a confidence
    interval, because a mean of 0.0001 with a standard error of 0.0004 is not a Tuesday effect.
    """
    rows = []
    labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for day, label in enumerate(labels):
        block = returns[returns.index.dayofweek == day].dropna()
        stat, p_value = stats.ttest_1samp(block, 0.0)
        rows.append(
            {
                "weekday": label,
                "n": len(block),
                "mean_return_bp": block.mean() * 10_000,
                "std_bp": block.std() * 10_000,
                "t_stat": stat,
                "p_value": p_value,
                "significant_5pct": p_value < 0.05,
            }
        )
    return pd.DataFrame(rows)

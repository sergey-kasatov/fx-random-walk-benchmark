"""Correctness checks for the parts of this repository that a wrong result would look fine in.

Run from the repository root:

    python tests/checks.py

Deliberately written with plain asserts and no test-runner dependency, so the checks run in the
same environment as the analysis with nothing extra installed.

What is checked and why each one earns its place:

1. **Feature leakage.** The claim that the backtest is honest rests entirely on features at
   time t using only data up to t. That is asserted mechanically here by rewriting the future
   and confirming the past does not move. It is the one bug that would invalidate every number
   in the repository while making the results look better.
2. **Target alignment.** A target off by one row is invisible in a plot and fatal to a metric.
3. **The naive forecaster really is the last observation.** The whole comparison is against
   this, so it is worth one assert.
4. **Walk-forward origins never see their own target.** The harness could be correct in the
   features and still hand a model tomorrow's row.
5. **Diebold-Mariano identity.** Comparing a forecast against itself must produce a zero loss
   differential, not a p-value.
6. **Variance ratio calibration.** This one is hand-rolled from the paper rather than taken from
   a library, so it is checked against series whose answer is known by construction: on a
   simulated random walk it must not reject, and on simulated momentum it must. An early version
   was missing the sqrt(n) scaling and therefore never rejected anything, which reads exactly
   like a clean result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_clean  # noqa: E402
from src.features import make_supervised  # noqa: E402
from src.diagnostics import variance_ratio  # noqa: E402
from src.metrics import diebold_mariano  # noqa: E402
from src.models import NaiveForecaster, XGBForecaster, _feature_row_for_today  # noqa: E402
from src.backtest import walk_forward  # noqa: E402

PASSED: list[str] = []


def check(name: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            fn(*args, **kwargs)
            PASSED.append(name)
            print(f"  ok   {name}")
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


@check("features at time t are unchanged when the future is rewritten")
def test_no_leakage(df: pd.DataFrame) -> None:
    cut = df.index[3000]
    for mode in ("returns", "levels"):
        original = _feature_row_for_today(df.loc[:cut], "SGD_per_USD", mode)

        # Corrupt everything after the cut beyond recognition. A feature that reaches forward
        # in time cannot survive this unchanged.
        tampered = df.copy()
        tampered.loc[tampered.index > cut] = tampered.loc[tampered.index > cut] * 5.0 + 100.0
        after = _feature_row_for_today(tampered.loc[:cut], "SGD_per_USD", mode)

        assert original.equals(after), f"{mode}: feature row moved when the future changed"

        # Same test through the supervised builder, which is the path the models train on.
        X_a, _, _ = make_supervised(df, "SGD_per_USD", horizon=5, mode=mode)
        X_b, _, _ = make_supervised(tampered, "SGD_per_USD", horizon=5, mode=mode)
        common = X_a.index[X_a.index <= cut]
        pd.testing.assert_frame_equal(X_a.loc[common], X_b.loc[common])


@check("target at row t is the level h business days later")
def test_target_alignment(df: pd.DataFrame) -> None:
    for horizon in (1, 5, 21):
        X, y, anchor = make_supervised(df, "CNY_per_USD", horizon=horizon, mode="levels")
        sample = y.index[len(y) // 2]
        position = df.index.get_loc(sample)
        assert y.loc[sample] == df["CNY_per_USD"].iloc[position + horizon]
        assert anchor.loc[sample] == df["CNY_per_USD"].iloc[position]

        # And in returns mode the target must reconstruct the same level.
        _, y_ret, anchor_ret = make_supervised(df, "CNY_per_USD", horizon=horizon, mode="returns")
        rebuilt = anchor_ret.loc[sample] * np.exp(y_ret.loc[sample])
        assert abs(rebuilt - df["CNY_per_USD"].iloc[position + horizon]) < 1e-9


@check("the naive forecast is the last observed value at every horizon")
def test_naive_is_persistence(df: pd.DataFrame) -> None:
    history = df.iloc[:2000]
    forecaster = NaiveForecaster().fit(history, "SGD_per_USD", [1, 5, 21])
    for horizon in (1, 5, 21):
        assert forecaster.forecast(horizon) == history["SGD_per_USD"].iloc[-1]


@check("walk-forward never lets a model see its own target row")
def test_walk_forward_alignment(df: pd.DataFrame) -> None:
    small = df.loc["2018-01-01":]
    preds = walk_forward(small, "SGD_per_USD", NaiveForecaster(), horizons=(1, 5, 21),
                         test_start="2019-01-01", refit_every=1)

    for _, row in preds.sample(40, random_state=0).iterrows():
        # y_true must be the real observation at the target date
        assert row["y_true"] == df.loc[row["target_date"], "SGD_per_USD"]
        # the target date must be exactly horizon business days after the origin
        gap = len(small.loc[row["origin"]:row["target_date"]]) - 1
        assert gap == row["horizon"], f"origin/target gap {gap} != horizon {row['horizon']}"
        # and for the naive model the prediction is the origin's own value, never later
        assert row["y_pred"] == df.loc[row["origin"], "SGD_per_USD"]


@check("XGBoost in returns mode reconstructs a level from the last observed rate")
def test_xgb_reconstruction(df: pd.DataFrame) -> None:
    history = df.iloc[:1500]
    forecaster = XGBForecaster(mode="returns", n_estimators=20).fit(
        history, "SGD_per_USD", [1]
    )
    prediction = forecaster.forecast(1)
    last = history["SGD_per_USD"].iloc[-1]
    # A daily FX move of more than 10 percent would mean the reconstruction is broken, not that
    # the model is bold.
    assert 0.9 * last < prediction < 1.1 * last, f"{prediction} implausible against {last}"


@check("Diebold-Mariano on identical forecasts is undefined rather than significant")
def test_dm_identity() -> None:
    rng = np.random.default_rng(0)
    errors = rng.normal(size=500)
    stat, p = diebold_mariano(errors, errors, horizon=1)
    assert np.isnan(stat) and np.isnan(p), "identical forecasts must not produce a verdict"

    # And a genuinely worse forecast must be detected as worse.
    worse = errors * 2
    stat, p = diebold_mariano(worse, errors, horizon=1)
    assert stat > 0 and p < 0.01, "a doubled error should be significantly worse"


@check("variance ratio accepts a simulated random walk and rejects simulated momentum")
def test_variance_ratio_calibration() -> None:
    rng = np.random.default_rng(7)

    # A true random walk. The statistic should be centred on zero with roughly unit variance,
    # and should reject at close to the nominal rate rather than never or always.
    z_scores = []
    for _ in range(80):
        walk = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.005, 800))))
        z_scores.append(variance_ratio(walk, q=5)["z_robust"])
    z_scores = np.array(z_scores)
    assert abs(z_scores.mean()) < 0.4, f"z not centred on zero: {z_scores.mean():.3f}"
    assert 0.6 < z_scores.std() < 1.6, f"z spread implausible: {z_scores.std():.3f}"
    rejection_rate = (np.abs(z_scores) > 1.96).mean()
    assert rejection_rate < 0.25, f"rejects a random walk {rejection_rate:.0%} of the time"

    # Positively autocorrelated returns. VR must exceed 1 and the test must notice.
    detected = []
    ratios = []
    for _ in range(40):
        noise = rng.normal(0, 0.005, 800)
        r = np.zeros(800)
        for t in range(1, 800):
            r[t] = 0.25 * r[t - 1] + noise[t]
        series = pd.Series(np.exp(np.cumsum(r)))
        result = variance_ratio(series, q=5)
        ratios.append(result["variance_ratio"])
        detected.append(result["random_walk_rejected_5pct"])
    assert np.mean(ratios) > 1.2, f"momentum not reflected in VR: {np.mean(ratios):.3f}"
    assert np.mean(detected) > 0.8, f"test has no power: detected {np.mean(detected):.0%}"


def main() -> int:
    print("Running checks...")
    df = load_clean()
    test_no_leakage(df)
    test_target_alignment(df)
    test_naive_is_persistence(df)
    test_walk_forward_alignment(df)
    test_xgb_reconstruction(df)
    test_dm_identity()
    test_variance_ratio_calibration()
    print(f"\n{len(PASSED)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

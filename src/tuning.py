"""Hyperparameter search for the gradient-boosted model, inside the training window only.

The brief asks for hyperparameter optimisation. The reason it gets its own module rather than a
few lines in a notebook is that on a time series it is one of the easiest places to leak: the
default ``GridSearchCV`` shuffles, which trains on the future, and any search scored on the
backtest window has let the test set pick the model.

Two rules are enforced here:

1. **The search never sees data after the backtest start.** It runs on the training window and
   nothing else, so the tuned configuration is one that could have been chosen in December 2015.
2. **The folds respect time.** ``TimeSeriesSplit`` produces expanding-window folds where the
   validation block always follows the training block.

The expected outcome is worth stating in advance, because it is the point of running this at
all: if the series is a random walk, tuning cannot help, and a tuned model that still fails to
beat persistence is much stronger evidence than an untuned one that fails.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from .features import make_supervised

# Deliberately small and coarse. A finer grid on a series with no signal buys nothing except a
# more overfitted choice and a longer runtime.
GRID = {
    "max_depth": [2, 4, 6],
    "learning_rate": [0.02, 0.05, 0.1],
    "n_estimators": [200, 400],
    "subsample": [0.8, 1.0],
}


def search_xgb(
    train: pd.DataFrame,
    target: str,
    horizon: int = 1,
    mode: str = "returns",
    n_splits: int = 5,
    grid: dict | None = None,
    verbose: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """Grid search XGBoost on the training window with time-respecting folds.

    Returns the best parameter dictionary and the full scored grid, so the notebook can show how
    flat the surface is rather than only the winner.
    """
    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBRegressor

    grid = grid or GRID
    X, y, _ = make_supervised(train, target, horizon=horizon, mode=mode)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    keys = list(grid)
    rows = []
    for combination in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combination))
        fold_scores = []
        for train_idx, valid_idx in splitter.split(X):
            model = XGBRegressor(
                **params,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=4,
                tree_method="hist",
                objective="reg:squarederror",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
            predicted = model.predict(X.iloc[valid_idx])
            fold_scores.append(
                float(np.sqrt(np.mean((y.iloc[valid_idx].to_numpy() - predicted) ** 2)))
            )
        rows.append({**params, "cv_rmse": float(np.mean(fold_scores)),
                     "cv_rmse_std": float(np.std(fold_scores))})
        if verbose:
            print(f"  {params} -> {rows[-1]['cv_rmse']:.6f}")

    results = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    best = {k: results.loc[0, k] for k in keys}
    # numpy scalars out of a DataFrame do not always survive being passed as keyword arguments.
    best = {k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in best.items()}
    return best, results


def naive_cv_benchmark(train: pd.DataFrame, target: str, horizon: int = 1,
                       n_splits: int = 5) -> float:
    """The random walk's score on the same folds, so the search has a reference line.

    In ``returns`` mode the naive forecast is "the return over the next h days will be zero", so
    its error on the validation fold is simply the realised return.
    """
    from sklearn.model_selection import TimeSeriesSplit

    X, y, _ = make_supervised(train, target, horizon=horizon, mode="returns")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    scores = [
        float(np.sqrt(np.mean(y.iloc[valid_idx].to_numpy() ** 2)))
        for _, valid_idx in splitter.split(X)
    ]
    return float(np.mean(scores))

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
    r2_score,
    make_scorer,
)
from sklearn.inspection import permutation_importance

# Performance metrics
def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tweedie_power: float = 1.5,
) -> dict[str, float]:
    """
    Compute standard regression performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    tweedie_power : float, optional
        Power parameter for the Tweedie deviance, by default 1.5.

    Returns
    -------
    dict[str, float]
        Dictionary containing regression performance metrics.
    """
    return {
        "tweedie_deviance": float(
            mean_tweedie_deviance(y_true, y_pred, power=tweedie_power)
        ),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

# Feature relevance (Permutation Importance)
def permutation_importance_table(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    tweedie_power: float = 1.5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation feature importance using Tweedie deviance
    as the evaluation metric.

    This corresponds to the "most relevant features" requirement
    in Project Part 4 and is conceptually aligned with the
    evaluation logic used in PS3, without insurance-specific
    constructs such as exposure or Gini/Lorenz curves.

    Parameters
    ----------
    model :
        Fitted model or pipeline with a predict method.
    X : pd.DataFrame
        Feature matrix used for evaluation (typically validation set).
    y : pd.Series
        True target values.
    tweedie_power : float, optional
        Power parameter for Tweedie deviance, by default 1.5.
    n_repeats : int, optional
        Number of permutation repeats, by default 10.
    random_state : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with features ranked by decreasing importance.
    """
    scorer = make_scorer(
        mean_tweedie_deviance,
        greater_is_better=False,
        power=tweedie_power,
    )

    r = permutation_importance(
        model,
        X,
        y,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": r.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
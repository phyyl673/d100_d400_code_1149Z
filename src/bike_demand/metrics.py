from __future__ import annotations

import numpy as np
import pandas as pd


def mse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return float(np.mean((y_true_arr - y_pred_arr) ** 2))


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def mean_preds(y_pred: pd.Series | np.ndarray) -> float:
    return float(np.mean(np.asarray(y_pred)))


def mean_outcome(y_true: pd.Series | np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true)))


def relative_bias(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """
    Relative bias = (mean(pred) - mean(true)) / mean(true)
    """
    mu_y = mean_outcome(y_true)
    if mu_y == 0:
        return float("nan")
    return float((mean_preds(y_pred) - mu_y) / mu_y)

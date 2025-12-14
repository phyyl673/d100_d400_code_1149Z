from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dalex as dx


def plot_predicted_vs_actual(y_true, y_pred, *, title: str) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.35)
    max_val = float(max(y_true.max(), y_pred.max()))
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def make_dalex_explainer(model, X: pd.DataFrame, y: pd.Series, label: str):
    return dx.Explainer(model=model, data=X, y=y, label=label)


def plot_dalex_feature_importance(explainer, top_n: int = 10) -> list[str]:
    parts = explainer.model_parts()
    parts.plot()

    res = parts.result
    res = res[~res["variable"].isin(["_baseline_", "_full_model_"])].copy()
    top = res.sort_values("dropout_loss", ascending=False).head(top_n)
    return top["variable"].tolist()


def plot_dalex_pdp(explainer, features: list[str]) -> None:
    prof = explainer.model_profile(variables=features)
    prof.plot()

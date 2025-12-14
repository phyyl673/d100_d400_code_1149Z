from __future__ import annotations

import numpy as np
import pandas as pd

from bike_demand.data.load_data import load_cleaned_data
from bike_demand.feature_engineering import build_glm_pipeline, build_lgbm_pipeline
from bike_demand.metrics import rmse, mae, mse, mean_preds, mean_outcome, relative_bias
from bike_demand.plotting import (
    plot_predicted_vs_actual,
    make_dalex_explainer,
    plot_dalex_feature_importance,
    plot_dalex_pdp,
)

TARGET_COL = "rented_bike_count"
DATE_COL = "date"


GLM_FINAL_PARAMS = {"model__alpha": 0.1, "model__l1_ratio": 1.0}
LGBM_FINAL_PARAMS = {
    "model__learning_rate": 0.05,
    "model__min_child_weight": 0.001,
    "model__n_estimators": 1500,
    "model__num_leaves": 31,
}


def train_valid_time_split(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    valid_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: last valid_size fraction is validation."""
    if date_col not in df.columns:
        raise KeyError(f"Missing date column: {date_col}")

    out = df.sort_values(date_col).reset_index(drop=True)
    cut = int(np.floor((1 - valid_size) * len(out)))
    return out.iloc[:cut].copy(), out.iloc[cut:].copy()


def fit_final_models(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit tuned final GLM and LGBM with fixed best params (fast)."""
    glm = build_glm_pipeline(date_col=DATE_COL)
    glm.set_params(**GLM_FINAL_PARAMS)
    glm.fit(X_train, y_train)

    lgbm = build_lgbm_pipeline(date_col=DATE_COL)
    lgbm.set_params(**LGBM_FINAL_PARAMS)
    lgbm.fit(X_train, y_train)

    return glm, lgbm


def run_step4(valid_size: float = 0.2) -> pd.DataFrame:
    """
    STEP 4 (required):
    - evaluate tuned GLM and LGBM on validation set
    - predicted vs actual plots (both models)
    - DALEX: feature importance + PDP for top 5 features (from LGBM)
    """
    df = load_cleaned_data()
    train_df, valid_df = train_valid_time_split(df, valid_size=valid_size)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_valid = valid_df.drop(columns=[TARGET_COL])
    y_valid = valid_df[TARGET_COL]

    glm_final, lgbm_final = fit_final_models(X_train, y_train)

    glm_pred = glm_final.predict(X_valid)
    lgbm_pred = lgbm_final.predict(X_valid)

    # ----- Metrics table (good for report) -----
    rows = []
    for name, pred in [("GLM", glm_pred), ("LGBM", lgbm_pred)]:
        rows.append(
            {
                "model": name,
                "mean_preds": mean_preds(pred),
                "mean_outcome": mean_outcome(y_valid),
                "rel_bias": relative_bias(y_valid, pred),
                "mse": mse(y_valid, pred),
                "rmse": rmse(y_valid, pred),
                "mae": mae(y_valid, pred),
            }
        )

    table = pd.DataFrame(rows).sort_values("rmse")
    print("\n" + "=" * 80)
    print("STEP 4 — Validation metrics (tuned final models)")
    print(table.to_string(index=False))
    print("=" * 80 + "\n")

    # ----- Predicted vs actual (required) -----
    plot_predicted_vs_actual(y_valid, glm_pred, title="GLM (tuned): Predicted vs Actual — validation")
    plot_predicted_vs_actual(y_valid, lgbm_pred, title="LGBM (tuned): Predicted vs Actual — validation")

    # ----- DALEX interpretation (required: importance + PDP top 5) -----
    explainer = make_dalex_explainer(lgbm_final, X_valid, y_valid, label="LGBM (tuned)")
    top5 = plot_dalex_feature_importance(explainer, top_n=5)
    print("Top 5 features (DALEX permutation importance, LGBM):", top5)
    plot_dalex_pdp(explainer, features=top5)

    return table

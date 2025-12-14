from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
)

from bike_demand.data.load_data import load_cleaned_data
from bike_demand.feature_engineering import build_glm_pipeline, build_lgbm_pipeline

# =============================================================================
# Config
# =============================================================================
TARGET_COL = "rented_bike_count"
DATE_COL = "date"
RANDOM_STATE = 42

# Choose split strategy: "time" (default, recommended) or "random"
SPLIT_STRATEGY = "time"
VALID_SIZE = 0.2

# CV settings
N_SPLITS = 5

# Tuning methods to run: choose any subset of {"grid", "random"}
TUNING_METHODS = ("grid", "random")

# RandomizedSearch budget (number of sampled configs)
N_ITER_RANDOM = 20


# =============================================================================
# Splitting
# =============================================================================
def train_valid_time_split(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    valid_size: float = VALID_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: last valid_size fraction (by date) is validation."""
    if date_col not in df.columns:
        raise KeyError(f"Missing date column: {date_col}")

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    cut = int(np.floor((1.0 - valid_size) * len(df_sorted)))
    return df_sorted.iloc[:cut].copy(), df_sorted.iloc[cut:].copy()


def train_valid_random_split(
    df: pd.DataFrame,
    valid_size: float = VALID_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split (allowed by brief)."""
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_df.copy(), valid_df.copy()


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if SPLIT_STRATEGY == "time":
        return train_valid_time_split(df)
    if SPLIT_STRATEGY == "random":
        return train_valid_random_split(df)
    raise ValueError("SPLIT_STRATEGY must be 'time' or 'random'.")


# =============================================================================
# Evaluation helpers
# =============================================================================
def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def print_metrics_table(rows: list[dict[str, object]]) -> None:
    """Pretty print results (no external dependencies)."""
    df = pd.DataFrame(rows)
    # keep consistent column order
    cols = ["model", "stage", "tuning", "rmse", "mae"]
    cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
    df = df[cols]
    print("\n" + "=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")


# =============================================================================
# Baseline training (no tuning)
# =============================================================================
def fit_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """
    Fit baseline GLM & LGBM pipelines with default parameters, evaluate on validation.
    Returns:
      - rows for metrics table
      - fitted estimators dict
    """
    results: list[dict[str, object]] = []
    fitted: dict[str, object] = {}

    glm = build_glm_pipeline(date_col=DATE_COL)
    glm.fit(X_train, y_train)
    pred = glm.predict(X_valid)
    m = compute_metrics(y_valid, pred)
    results.append({"model": "GLM", "stage": "baseline", "tuning": "none", **m})
    fitted["GLM_baseline"] = glm

    lgbm = build_lgbm_pipeline(date_col=DATE_COL)
    lgbm.fit(X_train, y_train)
    pred = lgbm.predict(X_valid)
    m = compute_metrics(y_valid, pred)
    results.append({"model": "LGBM", "stage": "baseline", "tuning": "none", **m})
    fitted["LGBM_baseline"] = lgbm

    return results, fitted


# =============================================================================
# Tuning
# =============================================================================
def get_cv() -> KFold:
    # K-fold CV as required in brief
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def tune_glm_grid(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    pipe = build_glm_pipeline(date_col=DATE_COL)

    # brief: tune alpha and l1_ratio
    param_grid = {
        "model__alpha": [0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.0, 0.5, 1.0],
    }

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=get_cv(),
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def tune_glm_random(X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    pipe = build_glm_pipeline(date_col=DATE_COL)

    # simple distributions / candidates
    param_dist = {
        "model__alpha": np.logspace(-3, 2, 30),
        "model__l1_ratio": np.linspace(0.0, 1.0, 21),
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=N_ITER_RANDOM,
        scoring="neg_root_mean_squared_error",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def tune_lgbm_grid(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    pipe = build_lgbm_pipeline(date_col=DATE_COL)

    # brief: tune learning_rate, n_estimators, n_leaves, min_child_weight
    # in LightGBM sklearn API, "n_leaves" is "num_leaves"
    param_grid = {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__n_estimators": [300, 800, 1500],
        "model__num_leaves": [15, 31, 63],
        "model__min_child_weight": [1e-3, 1e-2, 1e-1],
    }

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=get_cv(),
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def tune_lgbm_random(X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    pipe = build_lgbm_pipeline(date_col=DATE_COL)

    param_dist = {
        "model__learning_rate": np.linspace(0.02, 0.15, 50),
        "model__n_estimators": [200, 400, 800, 1200, 2000],
        "model__num_leaves": [15, 31, 63, 127],
        "model__min_child_weight": np.logspace(-3, 0, 50),
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=N_ITER_RANDOM,
        scoring="neg_root_mean_squared_error",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_search(
    model_name: str,
    tuning_name: str,
    search_obj,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[dict[str, object], object]:
    best_est = search_obj.best_estimator_
    pred = best_est.predict(X_valid)
    m = compute_metrics(y_valid, pred)

    row = {
        "model": model_name,
        "stage": "tuned",
        "tuning": tuning_name,
        **m,
        "best_params": search_obj.best_params_,
    }
    return row, best_est


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    df = load_cleaned_data()

    train_df, valid_df = split_data(df)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_valid = valid_df.drop(columns=[TARGET_COL])
    y_valid = valid_df[TARGET_COL]

    print(f"Split strategy: {SPLIT_STRATEGY} | train={len(train_df)} valid={len(valid_df)}")

    # 1) Baselines
    baseline_rows, fitted = fit_baselines(X_train, y_train, X_valid, y_valid)

    # 2) Tuning (one or both methods)
    tuned_rows: list[dict[str, object]] = []

    if "grid" in TUNING_METHODS:
        glm_grid = tune_glm_grid(X_train, y_train)
        row, _ = evaluate_search("GLM", "grid", glm_grid, X_valid, y_valid)
        tuned_rows.append(row)

        lgbm_grid = tune_lgbm_grid(X_train, y_train)
        row, _ = evaluate_search("LGBM", "grid", lgbm_grid, X_valid, y_valid)
        tuned_rows.append(row)

    if "random" in TUNING_METHODS:
        glm_rand = tune_glm_random(X_train, y_train)
        row, _ = evaluate_search("GLM", "random", glm_rand, X_valid, y_valid)
        tuned_rows.append(row)

        lgbm_rand = tune_lgbm_random(X_train, y_train)
        row, _ = evaluate_search("LGBM", "random", lgbm_rand, X_valid, y_valid)
        tuned_rows.append(row)

    # 3) Print comparison table (baseline vs tuned)
    rows_for_table = baseline_rows + [
        {k: v for k, v in r.items() if k in {"model", "stage", "tuning", "rmse", "mae"}}
        for r in tuned_rows
    ]
    print_metrics_table(rows_for_table)

    # 4) Print best params for tuned runs (for your report)
    if tuned_rows:
        print("Best parameters (tuned runs):")
        for r in tuned_rows:
            print(f"- {r['model']} [{r['tuning']}]: {r['best_params']}")

    # Optional: decide “final model” as the best RMSE among tuned runs
    if tuned_rows:
        best = min(tuned_rows, key=lambda d: d["rmse"])
        print("\nSelected final model (lowest validation RMSE among tuned runs):")
        print(f"  Model={best['model']} | tuning={best['tuning']} | RMSE={best['rmse']:.3f} | MAE={best['mae']:.3f}")


if __name__ == "__main__":
    main()

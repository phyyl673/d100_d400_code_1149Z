from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import mean_tweedie_deviance, make_scorer
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV

from bike_demand.data.load_data import load_cleaned_data
from bike_demand.modelling.sample_split import create_sample_split
from bike_demand.modelling.glm_pipeline import build_glm_pipeline, split_xy as split_xy_glm
from bike_demand.modelling.lgbm_pipeline import build_lgbm_pipeline, split_xy as split_xy_lgbm


TWEEDIE_POWER = 1.5
RANDOM_STATE = 42


def _tweedie_scorer(power: float):
    # safer than relying on the string scorer, because we fix `power`
    return make_scorer(mean_tweedie_deviance, greater_is_better=False, power=power)


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main() -> None:
    # ***Output directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Load
    df_load = load_cleaned_data()
    
    # Restrict to functioning days ONLY (structural zeros)
    df_nf = df_load[df_load["functioning_day"].eq("Yes")].copy()



    # Split (ID-hash using composite key: date + hour)
    df = create_sample_split(
        df_nf,
        method="id_hash",
        key_cols=["date", "hour"],
        train_frac=0.8,
    )

    df.to_parquet("data/processed/seoul_bike_with_split.parquet")

    train_df = df[df["sample"] == "train"].copy()
    val_df = df[df["sample"] == "validation"].copy()

    # Prepare X/y
    X_train_glm, y_train = split_xy_glm(train_df, spec=None)
    X_val_glm, y_val = split_xy_glm(val_df, spec=None)

    X_train_lgbm, _ = split_xy_lgbm(train_df, spec=None)
    X_val_lgbm, _ = split_xy_lgbm(val_df, spec=None)

    # CV setup (tuning happens ONLY inside training set)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scorer = _tweedie_scorer(TWEEDIE_POWER)

    # Build pipelines
    glm_pipe = build_glm_pipeline(tweedie_power=TWEEDIE_POWER)
    lgbm_pipe = build_lgbm_pipeline(tweedie_power=TWEEDIE_POWER, random_state=RANDOM_STATE)

    # Hyperparameter tuning
    glm_param_grid = {
        "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        "model__l1_ratio": [0.25, 0.5, 0.75, 1.0],
    }

    glm_search = GridSearchCV(
        estimator=glm_pipe,
        param_grid=glm_param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )
    glm_search.fit(X_train_glm, y_train)

    print("\n[GLM] best params:", glm_search.best_params_)
    print("[GLM] best CV score (neg tweedie deviance):", glm_search.best_score_)

    lgbm_param_dist = {
        "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
        "model__n_estimators": [300, 500, 800, 1200],
        "model__num_leaves": [15, 31, 63, 127],
        "model__min_child_weight": [1e-3, 1e-2, 1e-1, 1.0],
    }

    lgbm_search = RandomizedSearchCV(
        estimator=lgbm_pipe,
        param_distributions=lgbm_param_dist,
        n_iter=20,
        scoring=scorer,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )
    lgbm_search.fit(X_train_lgbm, y_train)

    print("\n[LGBM] best params:", lgbm_search.best_params_)
    print("[LGBM] best CV score (neg tweedie deviance):", lgbm_search.best_score_)

    # Hold-out validation sanity check (NOT used for tuning)
    glm_best = glm_search.best_estimator_
    lgbm_best = lgbm_search.best_estimator_

    glm_pred = glm_best.predict(X_val_glm)
    lgbm_pred = lgbm_best.predict(X_val_lgbm)

    glm_val_dev = mean_tweedie_deviance(y_val, glm_pred, power=TWEEDIE_POWER)
    lgbm_val_dev = mean_tweedie_deviance(y_val, lgbm_pred, power=TWEEDIE_POWER)

    print("\n[Validation] Tweedie deviance (lower is better)")
    print("  GLM :", glm_val_dev)
    print("  LGBM:", lgbm_val_dev)

    # ***Save best params + key results for evaluation.py
    glm_best_params = glm_search.best_params_
    lgbm_best_params = lgbm_search.best_params_

    # ***Save separate files (easy to load per-model)
    _save_json(glm_best_params, results_dir / "glm_best_params.json")
    _save_json(lgbm_best_params, results_dir / "lgbm_best_params.json")

    # ***Save one combined file (single source of truth)
    payload = {
        "meta": {
            "tweedie_power": TWEEDIE_POWER,
            "random_state": RANDOM_STATE,
            "cv": {"type": "KFold", "n_splits": 5, "shuffle": True},
            "note": "Best params selected via CV on training split; hold-out validation used only for sanity check.",
        },
        "glm": {
            "best_params": glm_best_params,
            "best_cv_score_neg_tweedie_deviance": float(glm_search.best_score_),
            "validation_tweedie_deviance": float(glm_val_dev),
        },
        "lgbm": {
            "best_params": lgbm_best_params,
            "best_cv_score_neg_tweedie_deviance": float(lgbm_search.best_score_),
            "validation_tweedie_deviance": float(lgbm_val_dev),
        },
    }
    _save_json(payload, results_dir / "best_params.json")

    print(f"\n[Saved] {results_dir / 'glm_best_params.json'}")
    print(f"[Saved] {results_dir / 'lgbm_best_params.json'}")
    print(f"[Saved] {results_dir / 'best_params.json'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bike_demand.data.load_data import load_cleaned_data
from bike_demand.modelling.glm_pipeline import (
    build_glm_pipeline,
    split_xy as split_xy_glm,
)
from bike_demand.modelling.lgbm_pipeline import (
    build_lgbm_pipeline,
    split_xy as split_xy_lgbm,
)

from bike_demand.evaluation.metrics import (
    regression_metrics,
    permutation_importance_table,
)
from bike_demand.plotting import (
    plot_predicted_vs_actual,
    plot_partial_dependence,
    plot_permutation_importance
)


def run_evaluation(
    *,
    results_dir: Path = Path("results"),
    tweedie_power: float = 1.5,
    random_state: int = 42,
) -> None:
    """
    Run evaluation and interpretation.

    This function evaluates tuned GLM and LGBM models on the
    validation set using the same data split created during
    model training. It computes performance metrics, produces
    diagnostic plots, and generates partial dependence plots
    for the most important features.
    """

    # Setup output directories
    figures_dir = results_dir / "figures"
    pdp_dir = figures_dir / "pdp"
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdp_dir.mkdir(parents=True, exist_ok=True)

    # Load tuned hyperparameters (from model_training)
    glm_params = json.loads((results_dir / "glm_best_params.json").read_text())
    lgbm_params = json.loads((results_dir / "lgbm_best_params.json").read_text())

    # Load data (already split during model_training)
    df = pd.read_parquet("data/processed/seoul_bike_with_split.parquet")

    print(sorted(df.columns.tolist()))

    train_df = df[df["sample"] == "train"]
    val_df   = df[df["sample"] == "validation"]

    # Prepare features and target
    X_train_glm, y_train = split_xy_glm(train_df, spec=None)
    X_val_glm, y_val = split_xy_glm(val_df, spec=None)

    X_train_lgbm, _ = split_xy_lgbm(train_df, spec=None)
    X_val_lgbm, _ = split_xy_lgbm(val_df, spec=None)

    # Rebuild and fit tuned models
    glm = build_glm_pipeline(tweedie_power=tweedie_power)
    glm.set_params(**glm_params)
    glm.fit(X_train_glm, y_train)

    lgbm = build_lgbm_pipeline(
        tweedie_power=tweedie_power,
        random_state=random_state,
    )
    lgbm.set_params(**lgbm_params)
    lgbm.fit(X_train_lgbm, y_train)

    # Predictions on validation set
    glm_pred = glm.predict(X_val_glm)
    lgbm_pred = lgbm.predict(X_val_lgbm)

    # Evaluation metrics
    metrics = {
        "glm": regression_metrics(
            y_val.values,
            glm_pred,
            tweedie_power=tweedie_power,
        ),
        "lgbm": regression_metrics(
            y_val.values,
            lgbm_pred,
            tweedie_power=tweedie_power,
        ),
    }

    (results_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    # Predicted vs Actual plots
    plot_predicted_vs_actual(
        y_val.values,
        glm_pred,
        title="GLM: Predicted vs Actual (Validation)",
        save_path=figures_dir / "pred_vs_actual_glm.png",
    )

    plot_predicted_vs_actual(
        y_val.values,
        lgbm_pred,
        title="LGBM: Predicted vs Actual (Validation)",
        save_path=figures_dir / "pred_vs_actual_lgbm.png",
    )

    # Feature importance (Permutation Importance, validation set)
    lgbm_importance = permutation_importance_table(
        lgbm,
        X_val_lgbm,
        y_val,
        tweedie_power=tweedie_power,
    )

    lgbm_importance.to_csv(
        results_dir / "feature_importance_lglm.csv",
        index=False,
    )

    glm_importance = permutation_importance_table(
        glm,
        X_val_glm,
        y_val,
        tweedie_power=tweedie_power,
    )

    glm_importance.to_csv(
        results_dir / "feature_importance_glm.csv",
        index=False,
    )

    plot_permutation_importance(
    lgbm_importance,
    top_n=5,
    title="LGBM Permutation Feature Importance (Validation)",
    save_path=figures_dir / "pfi_lgbm.png",
)
    plot_permutation_importance(
    glm_importance,
    top_n=5,
    title="GLM Permutation Feature Importance (Validation)",
    save_path=figures_dir / "pfi_glm.png",
)



# Partial Dependence Plots (Top 5 features, LGBM)
    # Top 5 features
    top5_features = lgbm_importance["feature"].head(5)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 8),
        constrained_layout=True,
    )

    axes = axes.flatten()

    for i, feat in enumerate(top5_features):
        plot_partial_dependence(
            lgbm,
            X_val_lgbm,
            y_val,
            feature=feat,
            model_name="LGBM",
            ax=axes[i],
        )


    for j in range(len(top5_features), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Partial Dependence Plots (Top 5 Features, LGBM)",
        fontsize=16,
        fontweight="bold",
    )

    fig.savefig(pdp_dir / "pdp_top5_lgbm.png", dpi=300)
    plt.close(fig)

    print("Evaluation completed successfully.")


if __name__ == "__main__":
    run_evaluation()




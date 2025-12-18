from __future__ import annotations
import numpy as np
import seaborn as sns
import pandas as pd
import dalex as dx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List


sns.set_theme(style="whitegrid", context="talk")

##############
#Plots for EDA

def savefig(save_path: str | Path | None) -> None:
    """
    Save the current figure if save_path is provided.
    """
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "rented_bike_count",
    *,
    bins: int = 50,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot the distribution of the target variable using a histogram and boxplot.

    This visualization highlights skewness and potential outliers
    in the target variable.
    """
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    sns.histplot(df[target_col], bins=bins, kde=True, ax=ax1, color="teal")
    ax1.set_title(f"Distribution of {target_col}", fontweight="bold")
    ax1.set_ylabel("Frequency")

    sns.boxplot(x=df[target_col], ax=ax2, color="teal")
    ax2.set_xlabel(target_col)

    plt.tight_layout()
    savefig(save_path)
    return fig

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def plot_combined_explanatory_distributions(
    df: pd.DataFrame,
    columns: str | list[str],
    *,
    bins: int = 30,
    kde: bool = True,
    titles: list[str] | None = None,
    save_path: str | Path | None = None,
    ncols: int = 2,
) -> plt.Figure:
    if isinstance(columns, str):
        columns = [columns]
    
    n_vars = len(columns)
    nrows = (n_vars + ncols - 1) // ncols
    
    fig = plt.figure(figsize=(10 * ncols, 6 * nrows))
    
    # 外层GridSpec：控制不同变量之间的间距（大间距）
    outer_gs = GridSpec(
        nrows, ncols,
        figure=fig,
        hspace=0.4,  # 不同变量之间的大间距
        wspace=0.25,
        top=0.96,
        bottom=0.04,
        left=0.10, 
        right=0.90 
    )
    
    for idx, column in enumerate(columns):
        row = idx // ncols
        col = idx % ncols
        
        # 内层GridSpec：控制histogram和boxplot之间的间距（小间距）
        inner_gs = GridSpecFromSubplotSpec(
            2, 1,  # 2行1列
            subplot_spec=outer_gs[row, col],
            height_ratios=[3, 1],
            hspace=0.05  # histogram和boxplot之间的小间距
        )
        
        # Histogram
        ax1 = fig.add_subplot(inner_gs[0])
        sns.histplot(data=df, x=column, bins=bins, kde=kde, ax=ax1)
        ax1.set_title(titles[idx] if titles else f"Distribution of {column}", pad=8)
        ax1.set_ylabel("Frequency")
        ax1.set_xlabel('')
        ax1.tick_params(labelbottom=False, bottom=False)
        
        # Boxplot
        ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)
        sns.boxplot(data=df, x=column, ax=ax2)
        ax2.set_xlabel(column)
    
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    return fig

def plot_categorical_frequency(
    df: pd.DataFrame,
    col: str | list[str] | list[dict],
    *,
    order: list | None = None,
    palette: str | None = None,
    title: str | list[str] | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    ncols: int = 3,
    figsize_per_plot: tuple[float, float] = (6, 4),
):
    """
    Plot frequency distribution for one or more categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col : str, list[str], or list[dict]
        - str: single column name
        - list[str]: multiple column names
        - list[dict]: list of dicts with keys 'col', 'order', 'title'
    order : list, optional
        Category order (only for single column).
    palette : str, optional
        Seaborn palette.
    title : str or list[str], optional
        Plot title(s).
    ax : matplotlib Axes, optional
        Axis to draw the plot onto (only for single column).
    save_path : str or Path, optional
        Path to save the figure.
    ncols : int, default=3
        Number of columns in grid (for multiple plots).
    figsize_per_plot : tuple, default=(6, 4)
        Size of each plot (for multiple plots).

    Returns
    -------
    matplotlib Axes or Figure
    """
    # Case 1: Single column (original behavior)
    if isinstance(col, str):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
        
        sns.countplot(
            data=df,
            x=col,
            order=order,
            palette=palette,
            ax=ax,
        )
        
        ax.set_ylabel("Frequency")
        ax.set_xlabel(col)
        
        if title:
            ax.set_title(title, fontweight="bold")
        
        if save_path is not None:
            ax.figure.savefig(save_path, dpi=200, bbox_inches="tight")
        
        return ax
    
    # Case 2: Multiple columns
    # Normalize to list of dicts
    if isinstance(col[0], str):
        columns = [{'col': c} for c in col]
    else:
        columns = col
    
    # Handle titles
    if isinstance(title, str):
        titles = [title]
    elif isinstance(title, list):
        titles = title
    else:
        titles = None
    
    n_plots = len(columns)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig_width = figsize_per_plot[0] * ncols
    fig_height = figsize_per_plot[1] * nrows
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, fig_height),
        squeeze=False
    )
    
    axes_flat = axes.flatten()
    
    for idx, config in enumerate(columns):
        ax_current = axes_flat[idx]
        
        # Extract configuration
        col_name = config['col']
        col_order = config.get('order', None)
        col_title = config.get('title', None)
        col_palette = config.get('palette', palette)
        
        # Use provided title or fall back to titles list
        if col_title is None and titles is not None and idx < len(titles):
            col_title = titles[idx]
        
        # Plot
        sns.countplot(
            data=df,
            x=col_name,
            order=col_order,
            palette=col_palette,
            ax=ax_current,
        )
        
        ax_current.set_ylabel("Count")
        ax_current.set_xlabel(col_name)
        
        if col_title:
            ax_current.set_title(col_title, fontweight="bold")
    
    # Hide extra subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    return fig

def plot_target_vs_continuous(
    df: pd.DataFrame,
    features: list[str],
    target: str = "rented_bike_count",
):
    fig, axes = plt.subplots(
        4, 2,                    
        figsize=(12, 18),
        sharey=True,
    )

    axes = axes.ravel()

    for ax, feat in zip(axes, features):
        sns.scatterplot(
            data=df,
            x=feat,
            y=target,
            ax=ax,
            s=40,
            alpha=0.25,
            linewidth=0,
        )
        ax.set_title(f"{target} vs {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel(target)

    fig.suptitle(
        "Rented Bike Count vs Continuous Features",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()

def plot_target_vs_categorical_mean(
    df: pd.DataFrame,
    features: list[str],
    target: str = "rented_bike_count",
    orders: dict[str, list] | None = None,
    ncols: int = 2, 
):

    nrows = (len(features) + ncols - 1) // ncols  
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5 * nrows),
        sharey=True,
    )

    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for i, feat in enumerate(features):
        sns.barplot(
            data=df,
            x=feat,
            y=target,
            estimator="mean",
            order=orders.get(feat) if orders else None,
            ax=axes[i],
        )

        axes[i].set_title(f"Average {target} by {feat}")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel(target)

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)


    plt.tight_layout()
    plt.show()


def plot_hourly_trend_by_season(
    df: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot the average number of rented bikes by hour of day,
    separately for each season.

    This visualises a key interaction between time of day and seasonality.
    """
    hourly_avg = (
        df.groupby(["hour", "seasons"], observed=True)["rented_bike_count"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.lineplot(
        data=hourly_avg,
        x="hour",
        y="rented_bike_count",
        hue="seasons",
        style="seasons",
        markers=True,
        dashes=False,
        palette="viridis",
        linewidth=2.5,
        ax=ax,
    )

    ax.set_title("Average Bike Rentals by Hour and Season", fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Rented Bike Count")
    ax.set_xticks(range(0, 24))
    ax.legend(title="Season", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    savefig(save_path)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    *,
    target_col: str | None = None,         
    exclude_cols: Optional[List[str]] = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 10),    
    annot: bool = True,
) -> pd.DataFrame:
    """
    Plot a correlation heatmap for numerical features.
    Returns the correlation matrix used for plotting.
    """
    data = df.copy()
    if exclude_cols:
        data = data.drop(columns=exclude_cols, errors="ignore")

    numeric_df = data.select_dtypes(include=["number"]).copy()

    if target_col is not None and target_col in numeric_df.columns:
        cols = [target_col] + [c for c in numeric_df.columns if c != target_col]
        numeric_df = numeric_df[cols]

    corr = numeric_df.corr(numeric_only=True)

    sns.set_context("talk")   
    sns.set_style("white")   

    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    sns.heatmap(
        corr,
        cmap="plasma",        
        vmin=-1, vmax=1,     
        center=0,
        square=True,
        linewidths=0.6,
        linecolor="white",
        annot=annot,
        fmt=".2f",
        annot_kws={"size": 9},        
        cbar_kws={"shrink": 0.85, "pad": 0.02},
        ax=ax,
    )

    ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=14)

    ax.tick_params(axis="x", labelrotation=90)
    ax.tick_params(axis="y", labelrotation=0)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return corr




#######################
# Plots for Evaluation

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot predicted values against actual values.

    This diagnostic plot is used to assess calibration,
    bias, and dispersion of model predictions on the
    validation set.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true target values.
    y_pred : np.ndarray
        Array of predicted target values produced by the model.
    title : str
        Title of the plot, typically indicating the model
        and dataset (e.g. "LGBM: Predicted vs Actual (Validation)").
    save_path : str or Path or None, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    sns.scatterplot(
        x=y_true,
        y=y_pred,
        alpha=0.25,
        edgecolor=None,
        ax=ax,
    )

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=2)

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    savefig(save_path)
    return fig

def plot_permutation_importance(
    importance_df: pd.DataFrame,
    *,
    top_n: int = 10,
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot permutation feature importance as a horizontal bar chart.
    """
    df = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df["feature"], df["importance"])
    ax.set_xlabel("Increase in loss after permutation")
    ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    return fig



def plot_partial_dependence(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature: str,
    *,
    model_name: str = "model",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure | None:
    """
    Plot a partial dependence profile.

    If ax is provided, draw on that axis and DO NOT save.
    If ax is None, create a new figure and optionally save.
    """
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in X")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True
    else:
        fig = ax.figure

    # ------------------------------------------------------------------
    # Numeric feature
    # ------------------------------------------------------------------
    if pd.api.types.is_numeric_dtype(X[feature]):
        explainer = dx.Explainer(
            model,
            X,
            y,
            label=model_name,
            verbose=False,
        )

        profile = explainer.model_profile(
            variables=[feature],
            type="partial",
        )

        res = profile.result.copy()

        if "_vname_" in res.columns:
            res = res[res["_vname_"] == feature]
        if "_label_" in res.columns:
            res = res[res["_label_"] == model_name]

        xcol = "_x_" if "_x_" in res.columns else feature
        ycol = "_yhat_" if "_yhat_" in res.columns else "yhat"

        res = res.sort_values(xcol)

        ax.plot(res[xcol].to_numpy(), res[ycol].to_numpy())
        ax.set_xlabel(feature)
        ax.set_ylabel("Mean prediction")
        ax.set_title(feature, fontweight="bold")

    # ------------------------------------------------------------------
    # Categorical feature
    # ------------------------------------------------------------------
    else:
        s = X[feature]
        levels = (
            list(s.cat.categories)
            if pd.api.types.is_categorical_dtype(s)
            else list(pd.Series(s).dropna().unique())
        )

        max_levels = 20
        levels = levels[:max_levels]

        means = []
        for lvl in levels:
            X_tmp = X.copy()
            X_tmp[feature] = lvl
            pred = model.predict(X_tmp)
            means.append(float(pd.Series(pred).mean()))

        out = (
            pd.DataFrame({"level": levels, "mean_pred": means})
            .sort_values("mean_pred", ascending=False)
        )

        ax.bar(out["level"].astype(str), out["mean_pred"])
        ax.set_xlabel(feature)
        ax.set_ylabel("Mean prediction")
        ax.set_title(feature, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

    if created_fig:
        plt.tight_layout()
        savefig(save_path)
        return fig

    return None

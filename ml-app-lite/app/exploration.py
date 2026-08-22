from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import ExperimentConfig


def run_exploration(train_df: pd.DataFrame, feature_cols: list[str], config: ExperimentConfig) -> None:
    out_dir = Path(config.report_dir) / "exploration"
    out_dir.mkdir(parents=True, exist_ok=True)

    #1. Statistiken (gesamt und je Label), zusätzlich als CSV abgelegt
    stats = train_df[feature_cols].describe().T
    print("\nStatistik (gesamt):")
    print(stats)
    stats.to_csv(out_dir / "feature_stats.csv")

    stats_by_label = train_df.groupby(config.label_column)[feature_cols].describe().T
    stats_by_label.to_csv(out_dir / "feature_stats_by_label.csv")

    #2. Fehlende Werte je Feature
    missing = train_df[feature_cols].isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        print("\nFehlende Werte je Feature:")
        print(missing)

    #3. Plots
    _plot_feature_distributions(train_df, feature_cols, config.label_column, out_dir)
    _plot_correlation_matrix(train_df, feature_cols, out_dir)
    print(f"\nExploration-Plots und Statistik-CSVs gespeichert in {out_dir}")


def _plot_feature_distributions(df: pd.DataFrame, feature_cols: list[str], label_col: str,
                                 out_dir: Path, max_cols: int = 4) -> None:
    #Histogramm je Feature, aufgeteilt nach Label
    n = len(feature_cols)
    ncols = min(max_cols, n)
    nrows = -(-n // ncols) #Ceiling-Division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, col in zip(axes, feature_cols):
        for label, group in df.groupby(label_col):
            ax.hist(group[col].dropna(), bins=30, alpha=0.5, label=str(label))
        ax.set_title(col, fontsize=8)
        ax.legend(fontsize=6)
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "feature_distributions.png", dpi=150)
    plt.close(fig)


def _plot_correlation_matrix(df: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> None:
    #Korrelationsmatrix der Features als Heatmap
    corr = df[feature_cols].corr()
    size = max(4.0, 0.3 * len(feature_cols))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels(feature_cols, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_matrix.png", dpi=150)
    plt.close(fig)
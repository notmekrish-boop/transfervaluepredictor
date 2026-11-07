"""
Visualization module using matplotlib to generate publication-ready plots.
"""
import os
import tempfile
# Ensure Matplotlib uses a writable temporary directory on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Position color mapping
POSITION_COLORS = {
    "Forward": "#e63946",     # Crimson
    "Midfielder": "#2a9d8f",  # Teal
    "Defender": "#457b9d",    # Steel Blue
    "Goalkeeper": "#e76f51",  # Orange/Amber
}


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    positions: pd.Series,
    metrics: Dict[str, float],
    save_path: Optional[Path] = None
) -> Path:
    """
    Scatter plot comparing Actual vs. Model-Predicted Transfer Values.
    """
    path = save_path or (config.VISUALIZATIONS_DIR / "actual_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    # Plot points by position
    for pos, color in POSITION_COLORS.items():
        mask = (positions == pos)
        if mask.any():
            ax.scatter(
                y_true[mask],
                y_pred[mask],
                c=color,
                label=pos,
                alpha=0.8,
                s=70,
                edgecolors="w",
                linewidth=0.6,
            )

    # Ideal diagonal line (y = x)
    max_val = max(float(y_true.max()), float(y_pred.max())) + 15
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.6, lw=1.5, label="Perfect Prediction (y = x)")

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Actual Transfer Value (€ Millions)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Transfer Value (€ Millions)", fontsize=12, fontweight="bold")
    ax.set_title("Premier League Transfer Value: Actual vs. Predicted (Linear Regression)", fontsize=14, fontweight="bold", pad=12)

    # Performance stats text box
    stats_text = (
        f"Model Performance:\n"
        f"Test R²: {metrics.get('test_r2', 0):.3f}\n"
        f"Test MAE: €{metrics.get('test_mae', 0):.2f}M\n"
        f"Test RMSE: €{metrics.get('test_rmse', 0):.2f}M"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved Actual vs Predicted plot to {path}")
    return path


def plot_feature_coefficients(
    coefficients: Dict[str, float],
    save_path: Optional[Path] = None
) -> Path:
    """
    Horizontal bar chart showing learned linear regression coefficients (weights).
    """
    path = save_path or (config.VISUALIZATIONS_DIR / "feature_coefficients.png")

    # Clean display labels
    label_map = {
        "goals": "Goals Scored (per goal)",
        "assists": "Assists (per assist)",
        "minutes_played": "Minutes Played (per 1,000 mins)",
        "age": "Age (per year)",
        "position_Forward": "Position: Forward (vs Defender)",
        "position_Midfielder": "Position: Midfielder (vs Defender)",
        "position_Goalkeeper": "Position: Goalkeeper (vs Defender)",
    }

    display_features = []
    display_weights = []

    for feat, weight in coefficients.items():
        display_features.append(label_map.get(feat, feat))
        # Scale minutes played to per 1,000 minutes for intuitive bar chart readability
        if feat == "minutes_played":
            display_weights.append(weight * 1000.0)
        else:
            display_weights.append(weight)

    # Sort by weight
    sorted_indices = np.argsort(display_weights)
    sorted_features = [display_features[i] for i in sorted_indices]
    sorted_weights = [display_weights[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#2a9d8f" if w >= 0 else "#e76f51" for w in sorted_weights]

    bars = ax.barh(sorted_features, sorted_weights, color=colors, edgecolor="black", linewidth=0.5, height=0.6)

    # Add numeric labels to bars
    for bar, val in zip(bars, sorted_weights):
        offset = 0.4 if val >= 0 else -0.4
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"€{val:+.2f}M",
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="bold",
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Impact on Transfer Value (€ Millions)", fontsize=12, fontweight="bold")
    ax.set_title("Linear Regression Coefficients: How Stats Drive Transfer Value", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved Feature Coefficients plot to {path}")
    return path


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None
) -> Path:
    """
    Residual distribution plot (Predicted - Actual) to analyze prediction errors.
    """
    path = save_path or (config.VISUALIZATIONS_DIR / "residuals_distribution.png")
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    n, bins, patches = ax.hist(
        residuals,
        bins=15,
        color="#457b9d",
        edgecolor="white",
        alpha=0.85,
        density=True,
    )

    # Add KDE/Normal curve
    from scipy.stats import norm
    mu, std = norm.fit(residuals)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, "r-", linewidth=2, label=f"Normal Fit (μ={mu:.1f}, σ={std:.1f})")

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero Error Line")
    ax.set_xlabel("Prediction Error: (Predicted - Actual) in € Millions", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.set_title("Distribution of Transfer Value Prediction Residuals", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved Residuals plot to {path}")
    return path


def plot_market_value_relationships(
    df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Path:
    """
    Multi-panel relationship plot: Goals vs Value and Age vs Value across positions.
    """
    path = save_path or (config.VISUALIZATIONS_DIR / "value_relationships.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    for pos, color in POSITION_COLORS.items():
        sub = df[df["position"] == pos]
        if not sub.empty:
            ax1.scatter(sub["goals"], sub[config.TARGET], color=color, label=pos, s=55, alpha=0.75, edgecolors="w")
            ax2.scatter(sub["age"], sub[config.TARGET], color=color, label=pos, s=55, alpha=0.75, edgecolors="w")

    ax1.set_xlabel("Goals Scored in Season", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Transfer Market Value (€M)", fontsize=11, fontweight="bold")
    ax1.set_title("Market Value vs. Goals", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.set_xlabel("Player Age", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Transfer Market Value (€M)", fontsize=11, fontweight="bold")
    ax2.set_title("Market Value vs. Age (Youth Premium & Veteran Decay)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Key Drivers of Premier League Transfer Value", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Value Relationships plot to {path}")
    return path

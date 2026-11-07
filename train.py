"""
Main training script for Premier League Transfer Value Predictor.
Integrates data loading, API syncing (requests), model training, evaluation, and visualizations.
"""
import sys
import io
# Ensure Windows terminal handles UTF-8 safely
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import logging
from pathlib import Path
import pandas as pd

import config
from src.data_loader import load_player_data, sync_with_football_data_api
from src.model import TransferValueModel
from src.visualizer import (
    plot_actual_vs_predicted,
    plot_feature_coefficients,
    plot_residuals,
    plot_market_value_relationships,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("       PREMIER LEAGUE TRANSFER VALUE PREDICTOR - TRAINING PIPELINE")
    print("=" * 80)

    # 1. Check API & Sync if token available
    print("\n[Step 1/5] Checking football-data.org API connection (requests library)...")
    sync_with_football_data_api()

    # 2. Load and Prepare Player Data
    print("\n[Step 2/5] Loading Premier League player dataset...")
    df = load_player_data()
    print(f" Loaded {len(df)} player records across seasons.")
    print(f" Positions represented: {dict(df['position'].value_counts())}")
    print(f" Value range: €{df[config.TARGET].min():.1f}M to €{df[config.TARGET].max():.1f}M (Mean: €{df[config.TARGET].mean():.1f}M)")

    # 3. Train Linear Regression Model
    print("\n[Step 3/5] Training Linear Regression Model (scikit-learn)...")
    model = TransferValueModel()
    metrics, (X_train, X_test, y_train, y_test) = model.train(df, test_size=0.2, random_state=42)

    # 4. Display Results & Regression Formula
    print("\n" + "-" * 50)
    print(" MODEL EVALUATION METRICS")
    print("-" * 50)
    print(f" Train R² Score:         {metrics['train_r2']:.4f}")
    print(f" Test R² Score:          {metrics['test_r2']:.4f}")
    print(f" Test MAE (Mean Abs):    €{metrics['test_mae']:.2f} Million")
    print(f" Test RMSE (Root MSE):   €{metrics['test_rmse']:.2f} Million")
    print(f" Train Samples:          {metrics['train_samples']}")
    print(f" Test Samples:           {metrics['test_samples']}")

    coeff_info = model.get_coefficients()
    intercept = coeff_info["intercept"]
    coefs = coeff_info["coefficients"]

    print("\n" + "-" * 50)
    print(" LEARNED LINEAR REGRESSION FORMULA")
    print("-" * 50)
    print(f" Base Intercept (b_0):    €{intercept:.2f}M")
    print(" Feature Coefficients (Weights):")
    for feat, weight in coefs.items():
        if feat == "minutes_played":
            print(f"   * {feat:<22}: €{weight:+.4f}M per min (or €{weight * 1000:+.2f}M per 1,000 mins)")
        else:
            print(f"   * {feat:<22}: €{weight:+.2f}M")

    # 5. Generate Visualizations (Matplotlib)
    print("\n[Step 4/5] Generating Matplotlib Visualizations...")
    y_test_pred = model.pipeline.predict(X_test)
    test_positions = X_test["position"]

    p1 = plot_actual_vs_predicted(y_test, y_test_pred, test_positions, metrics)
    p2 = plot_feature_coefficients(coefs)
    p3 = plot_residuals(y_test, y_test_pred)
    p4 = plot_market_value_relationships(df)

    print(f"  Saved: {p1.name}")
    print(f"  Saved: {p2.name}")
    print(f"  Saved: {p3.name}")
    print(f"  Saved: {p4.name}")

    # 6. Save Trained Model
    print("\n[Step 5/5] Saving model pipeline...")
    saved_path = model.save()
    print(f" Model saved to: {saved_path}")

    # 7. Sample Test Predictions Table
    print("\n" + "=" * 80)
    print(" SAMPLE TEST PREDICTIONS VS. ACTUAL TRANSFER VALUES")
    print("=" * 80)
    test_indices = X_test.index[:10]
    sample_df = df.loc[test_indices].copy()
    sample_preds = model.pipeline.predict(X_test.loc[test_indices])

    print(f"{'Player':<24} {'Pos':<11} {'Age':<4} {'G':<3} {'A':<3} {'Actual (€M)':<13} {'Predicted':<12} {'Diff (€M)':<10}")
    print("-" * 80)
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        pred = max(0.5, sample_preds[idx])
        actual = row[config.TARGET]
        diff = pred - actual
        print(f"{row['player_name']:<24} {row['position']:<11} {row['age']:<4} {row['goals']:<3} {row['assists']:<3} €{actual:<11.1f} €{pred:<10.1f} {diff:+6.1f}M")

    print("\n Training pipeline completed successfully!")


if __name__ == "__main__":
    main()

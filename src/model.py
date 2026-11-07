"""
Model training, evaluation, and inference using scikit-learn LinearRegression.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TransferValueModel:
    """Linear Regression Pipeline for estimating Premier League player transfer market values."""

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False

    def _build_pipeline(self) -> Pipeline:
        """Construct scikit-learn Pipeline with ColumnTransformer and LinearRegression."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", config.NUMERICAL_FEATURES),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
            ]
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        )
        return pipeline

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[Dict[str, float], Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Train the linear regression model on player data and evaluate on held-out test split.
        Returns evaluation metrics and the (X_train, X_test, y_train, y_test) tuples.
        """
        X = df[config.ALL_FEATURES]
        y = df[config.TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        # Extract transformed feature names
        preprocessor = self.pipeline.named_steps["preprocessor"]
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_features = list(cat_encoder.get_feature_names_out(config.CATEGORICAL_FEATURES))
        self.feature_names = config.NUMERICAL_FEATURES + cat_features

        # Evaluate performance
        y_train_pred = self.pipeline.predict(X_train)
        y_test_pred = self.pipeline.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        metrics = {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        logger.info(
            f"Trained Linear Regression Model | Test R²: {test_r2:.3f} | Test MAE: €{test_mae:.2f}M | Test RMSE: €{test_rmse:.2f}M"
        )
        return metrics, (X_train, X_test, y_train, y_test)

    def get_coefficients(self) -> Dict[str, Any]:
        """
        Return the learned linear coefficients and intercept with human-readable interpretations.
        """
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Model has not been trained yet.")

        regressor: LinearRegression = self.pipeline.named_steps["regressor"]
        coefficients = dict(zip(self.feature_names, regressor.coef_))
        intercept = float(regressor.intercept_)

        return {
            "intercept": intercept,
            "coefficients": coefficients,
        }

    def predict_value(self, player_stats: Dict[str, Any]) -> float:
        """
        Predict market value in Millions EUR for a player dictionary.
        Required keys: 'goals', 'assists', 'minutes_played', 'age', 'position'
        """
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Model is not loaded or trained.")

        # Ensure correct formatting
        sample_df = pd.DataFrame([
            {
                "goals": int(player_stats["goals"]),
                "assists": int(player_stats["assists"]),
                "minutes_played": int(player_stats["minutes_played"]),
                "age": int(player_stats["age"]),
                "position": str(player_stats["position"]).capitalize().strip(),
            }
        ])

        pred = self.pipeline.predict(sample_df)[0]
        # Return bounded estimate (transfer value shouldn't be negative)
        return max(0.5, float(pred))

    def explain_prediction(self, player_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deconstruct a single prediction step-by-step showing feature contributions.
        """
        coeff_data = self.get_coefficients()
        intercept = coeff_data["intercept"]
        coefs = coeff_data["coefficients"]

        pos = str(player_stats["position"]).capitalize().strip()
        goals = int(player_stats["goals"])
        assists = int(player_stats["assists"])
        minutes = int(player_stats["minutes_played"])
        age = int(player_stats["age"])

        contributions = {
            "Base Intercept": intercept,
            f"Goals ({goals} × €{coefs.get('goals', 0):.2f}M)": goals * coefs.get("goals", 0),
            f"Assists ({assists} × €{coefs.get('assists', 0):.2f}M)": assists * coefs.get("assists", 0),
            f"Minutes Played ({minutes}m × €{coefs.get('minutes_played', 0):.4f}M)": minutes * coefs.get("minutes_played", 0),
            f"Age ({age} yrs × €{coefs.get('age', 0):.2f}M)": age * coefs.get("age", 0),
        }

        # Position contribution
        pos_feature = f"position_{pos}"
        if pos_feature in coefs:
            contributions[f"Position Premium ({pos})"] = coefs[pos_feature]
        else:
            # Baseline position dropped by drop='first' (e.g. Defender)
            contributions[f"Position Baseline ({pos})"] = 0.0

        raw_sum = sum(contributions.values())
        final_prediction = max(0.5, raw_sum)

        return {
            "contributions": contributions,
            "raw_sum": raw_sum,
            "final_predicted_value_eur_m": final_prediction,
        }

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save trained pipeline and metadata to disk."""
        path = filepath or config.MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_names": self.feature_names,
                "is_trained": self.is_trained,
            },
            path,
        )
        logger.info(f"Model saved successfully to {path}")
        return path

    def load(self, filepath: Optional[Path] = None) -> "TransferValueModel":
        """Load trained pipeline and metadata from disk."""
        path = filepath or config.MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}. Run train.py first.")
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.feature_names = data["feature_names"]
        self.is_trained = data["is_trained"]
        logger.info(f"Model loaded successfully from {path}")
        return self

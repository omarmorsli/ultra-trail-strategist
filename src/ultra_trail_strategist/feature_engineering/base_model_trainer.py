"""
Base Model Trainer for Pace Prediction.

Trains a gradient boosted model on the Endomondo/FitRec dataset
to predict running velocity from terrain features.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.ensemble import (  # type: ignore[import-untyped]
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (  # type: ignore[import-untyped]
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class BaseModelTrainer:
    """
    Train base pace prediction model on Endomondo dataset.

    Features:
    - grade (%) - primary predictor
    - altitude (m) - for altitude adjustment
    - heart_rate (optional) - for effort estimation

    Target:
    - velocity (m/s)

    Model: Gradient Boosted Trees for:
    - Better handling of non-linear grade/speed relationship
    - Feature importance for interpretability
    - Fast inference for real-time predictions

    Example
    -------
    >>> trainer = BaseModelTrainer()
    >>> model = trainer.train(training_df)
    >>> metrics = trainer.evaluate(model, test_df)
    >>> trainer.save_model(model, "models/endomondo_base.pkl")
    """

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        model_dir: Path = Path("models"),
    ):
        """
        Initialize the trainer.

        Parameters
        ----------
        model_type : str
            Type of model: 'gradient_boosting' or 'random_forest'.
        model_dir : Path
            Directory for saving trained models.
        """
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.feature_columns = ["grade", "altitude"]
        self.target_column = "velocity"

    def prepare_features(
        self,
        df: pl.DataFrame,
        include_heart_rate: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target vector from DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            Training data with grade, velocity, altitude columns.
        include_heart_rate : bool
            Whether to include heart_rate as a feature.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Feature matrix X and target vector y.
        """
        features = self.feature_columns.copy()
        if include_heart_rate and "heart_rate" in df.columns:
            features.append("heart_rate")
            # Filter rows with valid heart rate
            df = df.filter(pl.col("heart_rate").is_not_null())

        # Drop nulls in required columns
        df = df.drop_nulls(subset=features + [self.target_column])

        X = df.select(features).to_numpy()
        y = df.select(self.target_column).to_numpy().flatten()

        return X, y

    def train(
        self,
        data: pl.DataFrame,
        test_size: float = 0.2,
        include_heart_rate: bool = False,
        **model_params: Any,
    ) -> Dict[str, Any]:
        """
        Train the base model on provided data.

        Parameters
        ----------
        data : pl.DataFrame
            Training data with grade, velocity, altitude columns.
        test_size : float
            Fraction of data to use for testing.
        include_heart_rate : bool
            Include heart_rate as a feature.
        **model_params
            Additional parameters passed to the model.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - model: Trained model
            - metrics: Evaluation metrics
            - feature_importance: Feature importance scores
        """
        logger.info(f"Preparing features from {len(data)} samples...")
        X, y = self.prepare_features(data, include_heart_rate)

        logger.info(f"Feature matrix shape: {X.shape}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Create model
        if self.model_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=model_params.get("n_estimators", 200),
                max_depth=model_params.get("max_depth", 6),
                learning_rate=model_params.get("learning_rate", 0.1),
                min_samples_split=model_params.get("min_samples_split", 10),
                random_state=42,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=model_params.get("n_estimators", 100),
                max_depth=model_params.get("max_depth", 10),
                min_samples_split=model_params.get("min_samples_split", 10),
                random_state=42,
                n_jobs=-1,
            )

        logger.info(f"Training {self.model_type} model...")
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)

        # Feature importance
        features = self.feature_columns.copy()
        if include_heart_rate:
            features.append("heart_rate")

        feature_importance = dict(
            zip(features, model.feature_importances_, strict=True)
        )

        logger.info(f"Training complete. R² = {metrics['r2']:.4f}")

        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "feature_columns": features,
        }

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
        }

    def evaluate_by_grade_bucket(
        self,
        model: Any,
        test_data: pl.DataFrame,
        buckets: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance by grade buckets.

        Parameters
        ----------
        model : Any
            Trained model.
        test_data : pl.DataFrame
            Test data.
        buckets : Optional[List[Tuple[float, float]]]
            Grade buckets as (min, max) pairs.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Metrics by grade bucket.
        """
        if buckets is None:
            buckets = [
                (-50, -15),  # Steep downhill
                (-15, -5),   # Moderate downhill
                (-5, 0),     # Slight downhill
                (0, 5),      # Slight uphill
                (5, 15),     # Moderate uphill
                (15, 50),    # Steep uphill
            ]

        results = {}
        X, y = self.prepare_features(test_data)

        for min_grade, max_grade in buckets:
            mask = (test_data["grade"].to_numpy() >= min_grade) & (
                test_data["grade"].to_numpy() < max_grade
            )
            if not np.any(mask):
                continue

            X_bucket = X[mask]
            y_bucket = y[mask]
            y_pred = model.predict(X_bucket)

            bucket_name = f"{min_grade}% to {max_grade}%"
            results[bucket_name] = {
                **self._calculate_metrics(y_bucket, y_pred),
                "n_samples": int(np.sum(mask)),
            }

        return results

    def save_model(
        self,
        model_result: Dict[str, Any],
        model_name: str = "endomondo_base",
    ) -> Path:
        """
        Save trained model and metadata.

        Parameters
        ----------
        model_result : Dict[str, Any]
            Result from train() method.
        model_name : str
            Name for the saved model.

        Returns
        -------
        Path
            Path to saved model file.
        """
        model_path = self.model_dir / f"{model_name}.pkl"
        meta_path = self.model_dir / f"{model_name}_metadata.json"

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model_result["model"], f)

        # Save metadata (non-model data)
        metadata = {
            "model_type": self.model_type,
            "feature_columns": model_result["feature_columns"],
            "metrics": model_result["metrics"],
            "feature_importance": model_result["feature_importance"],
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        return model_path

    def load_model(self, model_name: str = "endomondo_base") -> Tuple[Any, Dict]:
        """
        Load trained model and metadata.

        Parameters
        ----------
        model_name : str
            Name of the saved model.

        Returns
        -------
        Tuple[Any, Dict]
            Loaded model and metadata.
        """
        model_path = self.model_dir / f"{model_name}.pkl"
        meta_path = self.model_dir / f"{model_name}_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        logger.info(f"Loaded model from {model_path}")
        return model, metadata

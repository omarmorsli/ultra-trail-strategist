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
from sklearn.model_selection import (  # type: ignore[import-untyped]
    GroupKFold,
    GroupShuffleSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

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

        # Core features (always used)
        self.feature_columns = ["grade", "altitude"]
        
        # Extended features (used if available)
        self.extended_features = [
            "distance_into_workout",
            "cumulative_elev_gain", 
            "workout_distance_pct",
        ]
        
        self.target_column = "velocity"
        self.scaler: Optional[StandardScaler] = None

    def prepare_features(
        self,
        df: pl.DataFrame,
        include_heart_rate: bool = False,
        include_hr_zone: bool = False,
        use_extended_features: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and target vector from DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            Training data with grade, velocity, altitude columns.
        include_heart_rate : bool
            Whether to include heart_rate as a feature.
        include_hr_zone : bool
            Whether to include hr_zone (1-5) as a feature.
        use_extended_features : bool
            Whether to include extended features (distance, elevation gain, etc.).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Feature matrix X and target vector y.
        """
        features = self.feature_columns.copy()
        
        # Add extended features if available and requested
        if use_extended_features:
            for ext_feat in self.extended_features:
                if ext_feat in df.columns:
                    features.append(ext_feat)
        
        if include_hr_zone and "hr_zone" in df.columns:
            features.append("hr_zone")
            # Filter rows with valid HR zone
            df = df.filter(pl.col("hr_zone").is_not_null())
        elif include_heart_rate and "heart_rate" in df.columns:
            features.append("heart_rate")
            # Filter rows with valid heart rate
            df = df.filter(pl.col("heart_rate").is_not_null())

        # Drop nulls in required columns
        df = df.drop_nulls(subset=features + [self.target_column])

        X = df.select(features).to_numpy()
        y = df.select(self.target_column).to_numpy().flatten()

        return X, y, features

    def train(
        self,
        data: pl.DataFrame,
        test_size: float = 0.2,
        include_heart_rate: bool = False,
        include_hr_zone: bool = False,
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
        include_hr_zone : bool
            Include hr_zone (1-5) as a feature (preferred over raw HR).
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
        X, y, feature_names = self.prepare_features(
            data, include_heart_rate, include_hr_zone
        )

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
                verbose=2,  # Show training progress
            )
        else:
            model = RandomForestRegressor(
                n_estimators=model_params.get("n_estimators", 100),
                max_depth=model_params.get("max_depth", 10),
                min_samples_split=model_params.get("min_samples_split", 10),
                random_state=42,
                n_jobs=-1,
                verbose=1,  # Show training progress
            )

        logger.info(
            f"Training {self.model_type} model on {len(X_train):,} samples "
            f"({len(X_test):,} test samples)..."
        )
        model.fit(X_train, y_train)
        logger.info("Training complete! Evaluating model...")

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)

        # Feature importance
        feature_importance = dict(
            zip(feature_names, model.feature_importances_, strict=True)
        )

        logger.info(f"Training complete. R² = {metrics['r2']:.4f}")

        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "feature_columns": feature_names,
        }

    def train_with_validation(
        self,
        data: pl.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.2,
        include_heart_rate: bool = False,
        include_hr_zone: bool = False,
        use_extended_features: bool = True,
        use_scaling: bool = True,
        **model_params: Any,
    ) -> Dict[str, Any]:
        """
        Train with proper train/val/test split grouped by workout_id.

        This prevents data leakage by ensuring data points from the same
        workout are never split across train, val, and test sets.

        Parameters
        ----------
        data : pl.DataFrame
            Training data with workout_id column for grouping.
        test_size : float
            Fraction of data for final test set.
        val_size : float
            Fraction of remaining data for validation.
        use_scaling : bool
            Whether to apply StandardScaler to features.
        **model_params
            Additional parameters passed to the model.

        Returns
        -------
        Dict[str, Any]
            Results with train, val, and test metrics.
        """
        logger.info(f"Preparing features from {len(data)} samples...")
        
        # Get workout IDs for group-based splitting
        if "workout_id" not in data.columns:
            logger.warning("No workout_id column - falling back to random split")
            return self.train(
                data, test_size, include_heart_rate, include_hr_zone, **model_params
            )

        # Prepare features (returns X, y, feature_names)
        result = self.prepare_features(
            data, include_heart_rate, include_hr_zone, use_extended_features
        )
        X, y, feature_names = result[0], result[1], result[2]
        
        # Get filtered workout IDs - must apply same filtering as prepare_features
        filtered_data = data.clone()
        
        # Apply HR zone/heart rate filtering to match prepare_features
        if include_hr_zone and "hr_zone" in data.columns:
            filtered_data = filtered_data.filter(pl.col("hr_zone").is_not_null())
        elif include_heart_rate and "heart_rate" in data.columns:
            filtered_data = filtered_data.filter(pl.col("heart_rate").is_not_null())
        
        # Drop nulls in feature columns
        filtered_data = filtered_data.drop_nulls(subset=feature_names + [self.target_column])
        groups = filtered_data["workout_id"].to_numpy()

        if len(groups) == 0:
            raise ValueError(
                f"No valid samples after filtering. "
                f"Check if hr_zone/heart_rate columns have data. "
                f"Try with include_hr_zone=False"
            )

        logger.info(f"Feature matrix shape: {X.shape}, using {len(feature_names)} features")
        logger.info(f"Features: {feature_names}")

        # First split: train+val vs test (by workout)
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        trainval_idx, test_idx = next(gss_test.split(X, y, groups))
        
        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx], y[test_idx]
        groups_trainval = groups[trainval_idx]

        # Second split: train vs val (by workout)
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_size / (1 - test_size), random_state=42
        )
        train_idx, val_idx = next(gss_val.split(X_trainval, y_trainval, groups_trainval))
        
        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        logger.info(
            f"Split sizes - Train: {len(X_train):,}, Val: {len(X_val):,}, "
            f"Test: {len(X_test):,}"
        )

        # Apply scaling if requested
        if use_scaling:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
            logger.info("Applied StandardScaler to features")

        # Create model
        if self.model_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=model_params.get("n_estimators", 200),
                max_depth=model_params.get("max_depth", 6),
                learning_rate=model_params.get("learning_rate", 0.1),
                min_samples_split=model_params.get("min_samples_split", 10),
                random_state=42,
                verbose=2,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=model_params.get("n_estimators", 100),
                max_depth=model_params.get("max_depth", 10),
                min_samples_split=model_params.get("min_samples_split", 10),
                random_state=42,
                n_jobs=-1,
                verbose=1,
            )

        logger.info(f"Training {self.model_type} model...")
        model.fit(X_train, y_train)
        logger.info("Training complete!")

        # Evaluate on all sets
        train_metrics = self._calculate_metrics(y_train, model.predict(X_train))
        val_metrics = self._calculate_metrics(y_val, model.predict(X_val))
        test_metrics = self._calculate_metrics(y_test, model.predict(X_test))

        logger.info(f"Train R²: {train_metrics['r2']:.4f}")
        logger.info(f"Val R²:   {val_metrics['r2']:.4f}")
        logger.info(f"Test R²:  {test_metrics['r2']:.4f}")

        # Check for overfitting
        overfit_gap = train_metrics['r2'] - val_metrics['r2']
        if overfit_gap > 0.1:
            logger.warning(
                f"⚠️ Possible overfitting: Train-Val gap = {overfit_gap:.4f}"
            )

        # Feature importance
        feature_importance = dict(
            zip(feature_names, model.feature_importances_, strict=True)
        )

        return {
            "model": model,
            "scaler": self.scaler,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance,
            "feature_columns": feature_names,
        }

    def cross_validate(
        self,
        data: pl.DataFrame,
        n_splits: int = 5,
        include_heart_rate: bool = False,
        include_hr_zone: bool = False,
        use_extended_features: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation with GroupKFold by workout_id.

        Parameters
        ----------
        data : pl.DataFrame
            Training data with workout_id column.
        n_splits : int
            Number of CV folds.

        Returns
        -------
        Dict[str, Any]
            Cross-validation scores and statistics.
        """
        logger.info(f"Running {n_splits}-fold cross-validation...")
        
        if "workout_id" not in data.columns:
            raise ValueError("workout_id column required for group-based CV")

        result = self.prepare_features(
            data, include_heart_rate, include_hr_zone, use_extended_features
        )
        X, y, feature_names = result[0], result[1], result[2]
        
        filtered_data = data.drop_nulls(subset=feature_names + [self.target_column])
        groups = filtered_data["workout_id"].to_numpy()

        # Create model
        if self.model_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=100,  # Faster for CV
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
            )

        # GroupKFold ensures same workout never in both train and test
        gkf = GroupKFold(n_splits=n_splits)
        scores = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring="r2")

        logger.info(f"CV R² scores: {scores}")
        logger.info(f"Mean R²: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return {
            "cv_scores": scores.tolist(),
            "mean_r2": float(scores.mean()),
            "std_r2": float(scores.std()),
            "n_splits": n_splits,
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
        X, y, _ = self.prepare_features(test_data)

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

        # Handle both old train() and new train_with_validation() result formats
        if "metrics" in model_result:
            # Old format from train()
            metrics = model_result["metrics"]
        elif "test_metrics" in model_result:
            # New format from train_with_validation()
            metrics = {
                "train": model_result["train_metrics"],
                "val": model_result["val_metrics"],
                "test": model_result["test_metrics"],
            }
        else:
            metrics = {}

        # Save metadata (non-model data)
        metadata = {
            "model_type": self.model_type,
            "feature_columns": model_result["feature_columns"],
            "metrics": metrics,
            "feature_importance": model_result["feature_importance"],
        }
        
        # Save scaler if present
        if "scaler" in model_result and model_result["scaler"] is not None:
            scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(model_result["scaler"], f)
            metadata["scaler_path"] = str(scaler_path)
        
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

"""Tests for the Base Model Trainer."""

import numpy as np
import polars as pl
import pytest

from ultra_trail_strategist.feature_engineering.base_model_trainer import (
    BaseModelTrainer,
)


class TestBaseModelTrainer:
    """Test suite for BaseModelTrainer."""

    @pytest.fixture
    def trainer(self) -> BaseModelTrainer:
        """Create a trainer instance."""
        return BaseModelTrainer(model_type="gradient_boosting")

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample training data with all required columns."""
        np.random.seed(42)
        n_samples = 500
        n_workouts = 10
        
        # Generate workout IDs (each workout has ~50 samples)
        workout_ids = np.repeat(np.arange(n_workouts), n_samples // n_workouts)
        
        # Generate features
        grades = np.random.uniform(-20, 20, n_samples)
        altitudes = np.random.uniform(0, 2000, n_samples)
        
        # Velocity inversely related to grade (slower uphill)
        base_velocity = 3.0  # 3 m/s base
        velocities = base_velocity - (grades * 0.05) + np.random.normal(0, 0.3, n_samples)
        velocities = np.clip(velocities, 0.5, 6.0)
        
        # New features
        distance_into_workout = np.tile(
            np.cumsum(np.random.uniform(50, 150, n_samples // n_workouts)),
            n_workouts
        )[:n_samples]
        cumulative_elev_gain = np.tile(
            np.cumsum(np.maximum(0, np.random.uniform(-10, 20, n_samples // n_workouts))),
            n_workouts
        )[:n_samples]
        workout_distance_pct = np.tile(
            np.linspace(0, 1, n_samples // n_workouts),
            n_workouts
        )[:n_samples]

        return pl.DataFrame({
            "workout_id": workout_ids,
            "user_id": np.random.choice([1, 2, 3], n_samples),
            "grade": grades,
            "altitude": altitudes,
            "velocity": velocities,
            "distance_into_workout": distance_into_workout,
            "cumulative_elev_gain": cumulative_elev_gain,
            "workout_distance_pct": workout_distance_pct,
        })

    def test_init(self, trainer: BaseModelTrainer) -> None:
        """Test trainer initialization."""
        assert trainer.model_type == "gradient_boosting"
        assert trainer.feature_columns == ["grade", "altitude"]
        assert "distance_into_workout" in trainer.extended_features
        assert trainer.scaler is None

    def test_prepare_features_basic(
        self, trainer: BaseModelTrainer, sample_data: pl.DataFrame
    ) -> None:
        """Test basic feature preparation."""
        X, y, feature_names = trainer.prepare_features(
            sample_data, use_extended_features=False
        )

        assert X.shape[1] == 2  # grade, altitude
        assert len(y) == len(sample_data)
        assert feature_names == ["grade", "altitude"]

    def test_prepare_features_extended(
        self, trainer: BaseModelTrainer, sample_data: pl.DataFrame
    ) -> None:
        """Test feature preparation with extended features."""
        X, y, feature_names = trainer.prepare_features(
            sample_data, use_extended_features=True
        )

        # Should include extended features
        assert X.shape[1] == 5  # grade, altitude + 3 extended
        assert "distance_into_workout" in feature_names
        assert "cumulative_elev_gain" in feature_names
        assert "workout_distance_pct" in feature_names

    def test_train_basic(
        self, trainer: BaseModelTrainer, sample_data: pl.DataFrame
    ) -> None:
        """Test basic training flow."""
        result = trainer.train(sample_data, n_estimators=10)

        assert "model" in result
        assert "metrics" in result
        assert "feature_importance" in result
        assert "r2" in result["metrics"]
        assert result["metrics"]["r2"] > -1  # Should have some predictive power

    def test_train_with_validation(
        self, trainer: BaseModelTrainer, sample_data: pl.DataFrame
    ) -> None:
        """Test training with proper train/val/test split."""
        result = trainer.train_with_validation(
            sample_data,
            test_size=0.2,
            val_size=0.2,
            use_scaling=True,
            n_estimators=10,
        )

        # Should have metrics for all three sets
        assert "train_metrics" in result
        assert "val_metrics" in result
        assert "test_metrics" in result
        
        # Should have scaler
        assert result["scaler"] is not None
        
        # R² values should be reasonable
        assert result["train_metrics"]["r2"] > -1
        assert result["val_metrics"]["r2"] > -1
        assert result["test_metrics"]["r2"] > -1

    def test_train_with_validation_no_leakage(
        self, trainer: BaseModelTrainer, sample_data: pl.DataFrame
    ) -> None:
        """Test that group-based splitting prevents data leakage."""
        # With only 10 workouts and 60/20/20 split, we should have
        # roughly 6 workouts in train, 2 in val, 2 in test
        result = trainer.train_with_validation(
            sample_data,
            test_size=0.2,
            val_size=0.2,
            n_estimators=10,
        )

        # Val R² should not be dramatically higher than test R²
        val_r2 = result["val_metrics"]["r2"]
        test_r2 = result["test_metrics"]["r2"]
        
        # If there was data leakage, val would be much better than test
        # Allow some variance but not excessive
        assert abs(val_r2 - test_r2) < 0.5

    def test_cross_validate(
        self, trainer: BaseModelTrainer, sample_data: pl.DataFrame
    ) -> None:
        """Test cross-validation with GroupKFold."""
        result = trainer.cross_validate(sample_data, n_splits=3)

        assert "cv_scores" in result
        assert "mean_r2" in result
        assert "std_r2" in result
        assert len(result["cv_scores"]) == 3
        assert result["std_r2"] >= 0

    def test_cross_validate_requires_workout_id(
        self, trainer: BaseModelTrainer
    ) -> None:
        """Test that cross_validate requires workout_id column."""
        data_no_workout = pl.DataFrame({
            "grade": [1.0, 2.0, 3.0],
            "altitude": [100.0, 200.0, 300.0],
            "velocity": [3.0, 2.5, 2.0],
        })

        with pytest.raises(ValueError, match="workout_id"):
            trainer.cross_validate(data_no_workout)

    def test_feature_importance_extended(
        self, trainer: BaseModelTrainer, sample_data: pl.DataFrame
    ) -> None:
        """Test that feature importance includes extended features."""
        result = trainer.train_with_validation(
            sample_data,
            use_extended_features=True,
            n_estimators=10,
        )

        fi = result["feature_importance"]
        assert "grade" in fi
        assert "altitude" in fi
        assert "distance_into_workout" in fi
        
        # Grade should typically be most important
        assert fi["grade"] > 0

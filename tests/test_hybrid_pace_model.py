"""Tests for the Hybrid Pace Predictor."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from ultra_trail_strategist.feature_engineering.pace_model import (
    HybridPacePredictor,
    PacePredictor,
)


class TestHybridPacePredictor:
    """Test suite for HybridPacePredictor."""

    @pytest.fixture
    def predictor(self, tmp_path: Path) -> HybridPacePredictor:
        """Create predictor with temporary model directory."""
        return HybridPacePredictor(model_dir=tmp_path)

    @pytest.fixture
    def mock_strava_streams(self) -> List[Dict[str, Any]]:
        """Create mock Strava activity streams."""
        # Create synthetic data: velocity decreases with grade
        grades = np.linspace(-10, 20, 200).tolist()
        velocities = [(3.0 - 0.08 * g) for g in grades]  # ~3 m/s flat, slower uphill
        altitudes = [1000.0] * 200

        return [
            {
                "grade_smooth": {"data": grades},
                "velocity_smooth": {"data": velocities},
                "altitude": {"data": altitudes},
                "moving": {"data": [True] * 200},
            }
        ]

    def test_backwards_compatibility_alias(self) -> None:
        """Test that PacePredictor is an alias for HybridPacePredictor."""
        assert PacePredictor is HybridPacePredictor

    def test_fallback_prediction_flat(self, predictor: HybridPacePredictor) -> None:
        """Test fallback prediction on flat terrain."""
        velocity = predictor.predict(grade=0.0)
        # Should be around 2.78 m/s (6 min/km)
        assert 2.5 < velocity < 3.5

    def test_fallback_prediction_uphill(self, predictor: HybridPacePredictor) -> None:
        """Test fallback prediction slows on uphills."""
        flat_velocity = predictor.predict(grade=0.0)
        uphill_velocity = predictor.predict(grade=10.0)
        assert uphill_velocity < flat_velocity

    def test_fallback_prediction_downhill(self, predictor: HybridPacePredictor) -> None:
        """Test fallback prediction speeds up on downhills."""
        flat_velocity = predictor.predict(grade=0.0)
        downhill_velocity = predictor.predict(grade=-5.0)
        assert downhill_velocity > flat_velocity

    def test_fallback_prediction_extreme_uphill(
        self, predictor: HybridPacePredictor
    ) -> None:
        """Test fallback handles extreme uphills."""
        velocity = predictor.predict(grade=30.0)
        assert velocity > 0.1  # Minimum floor
        # Should be quite slow but not unreasonable
        pace_min_km = (1000 / velocity) / 60
        assert 8 < pace_min_km < 25

    def test_fine_tune_insufficient_data(
        self, predictor: HybridPacePredictor
    ) -> None:
        """Test fine-tuning with insufficient data does not activate personal model."""
        # Small dataset
        small_streams = [
            {
                "grade_smooth": {"data": [0.0, 5.0, 10.0]},
                "velocity_smooth": {"data": [3.0, 2.5, 2.0]},
                "altitude": {"data": [100.0, 110.0, 120.0]},
            }
        ]

        predictor.fine_tune(small_streams)

        status = predictor.get_model_status()
        assert not status["using_personal"]

    def test_fine_tune_sufficient_data(
        self, predictor: HybridPacePredictor, mock_strava_streams: List[Dict[str, Any]]
    ) -> None:
        """Test fine-tuning with sufficient data activates personal model."""
        predictor.personal_data_threshold = 50  # Lower threshold for test

        predictor.fine_tune(mock_strava_streams)

        status = predictor.get_model_status()
        assert status["personal_model_fitted"]
        assert status["personal_data_count"] >= 50

    def test_predict_after_fine_tuning(
        self, predictor: HybridPacePredictor, mock_strava_streams: List[Dict[str, Any]]
    ) -> None:
        """Test predictions after fine-tuning."""
        predictor.personal_data_threshold = 50

        # Get prediction before fine-tuning (fallback)
        before_velocity = predictor.predict(grade=10.0)

        # Fine-tune
        predictor.fine_tune(mock_strava_streams)

        # Get prediction after fine-tuning
        after_velocity = predictor.predict(grade=10.0)

        # Both should be reasonable velocities
        assert 0.5 < before_velocity < 5.0
        assert 0.5 < after_velocity < 5.0

    def test_predict_segment_pace(self, predictor: HybridPacePredictor) -> None:
        """Test segment pace prediction."""
        # Flat terrain
        pace = predictor.predict_segment(avg_grade=0.0)
        assert 4.0 < pace < 10.0  # Reasonable running pace

        # Uphill
        uphill_pace = predictor.predict_segment(avg_grade=15.0)
        assert uphill_pace > pace

    def test_get_confidence_interval(self, predictor: HybridPacePredictor) -> None:
        """Test confidence interval calculation."""
        lower, upper = predictor.get_confidence_interval(grade=5.0)

        assert lower < upper
        assert lower > 0

        # Prediction should be within interval
        prediction = predictor.predict(grade=5.0)
        assert lower <= prediction <= upper

    def test_confidence_interval_extreme_grades(
        self, predictor: HybridPacePredictor
    ) -> None:
        """Test confidence intervals widen for extreme grades."""
        mild_lower, mild_upper = predictor.get_confidence_interval(grade=2.0)
        extreme_lower, extreme_upper = predictor.get_confidence_interval(grade=25.0)

        mild_range = mild_upper - mild_lower
        extreme_range = extreme_upper - extreme_lower

        # Extreme grades should have wider intervals
        assert extreme_range >= mild_range * 0.8  # Allow some variation

    def test_fatigue_adjustment(self, predictor: HybridPacePredictor) -> None:
        """Test that fatigue adjustment slows predictions."""
        # Early in race
        early_velocity = predictor.predict(
            grade=5.0,
            distance_into_race=10.0,
            total_race_distance=100.0,
        )

        # Late in race
        late_velocity = predictor.predict(
            grade=5.0,
            distance_into_race=90.0,
            total_race_distance=100.0,
        )

        assert late_velocity < early_velocity

    def test_night_adjustment(self, predictor: HybridPacePredictor) -> None:
        """Test that night hours slow predictions."""
        # Daytime
        day_velocity = predictor.predict(
            grade=5.0,
            distance_into_race=50.0,
            total_race_distance=100.0,
            hour_of_day=14,
        )

        # Nighttime
        night_velocity = predictor.predict(
            grade=5.0,
            distance_into_race=50.0,
            total_race_distance=100.0,
            hour_of_day=2,
        )

        assert night_velocity < day_velocity

    def test_get_model_status(self, predictor: HybridPacePredictor) -> None:
        """Test model status reporting."""
        status = predictor.get_model_status()

        assert "base_model_loaded" in status
        assert "personal_model_fitted" in status
        assert "personal_data_count" in status
        assert "using_personal" in status

    def test_minimum_velocity_floor(self, predictor: HybridPacePredictor) -> None:
        """Test that predictions never go below minimum floor."""
        # Extreme uphill
        velocity = predictor.predict(grade=50.0)
        assert velocity >= 0.1

        # With high fatigue
        velocity = predictor.predict(
            grade=30.0,
            distance_into_race=99.0,
            total_race_distance=100.0,
        )
        assert velocity >= 0.1

"""Tests for the legacy PacePredictor API compatibility."""

import unittest

import numpy as np

from ultra_trail_strategist.feature_engineering.pace_model import (
    HybridPacePredictor,
    PacePredictor,
)


class TestPacePredictor(unittest.TestCase):
    """Test legacy PacePredictor API compatibility with HybridPacePredictor."""

    def setUp(self) -> None:
        """Set up test predictor."""
        # Use HybridPacePredictor directly (PacePredictor is an alias)
        self.predictor = HybridPacePredictor()

        # Create synthetic Training Data
        # Relation: Velocity = 3.0 - 0.1 * Grade (Flat 3m/s, uphill slows down)
        # Grade 0 -> 3.0 m/s
        # Grade 10 -> 2.0 m/s

        grades = np.linspace(-10, 20, 200).tolist()
        velocities = [(3.0 - 0.1 * g) for g in grades]
        altitudes = [1000.0] * 200

        # Mock Strava Stream structure
        self.mock_streams = [
            {
                "grade_smooth": {"data": grades},
                "velocity_smooth": {"data": velocities},
                "altitude": {"data": altitudes},
                "moving": {"data": [True] * 200},
            }
        ]

    def test_training_and_prediction(self) -> None:
        """Test fine-tuning and prediction."""
        # Set low threshold for testing
        self.predictor.personal_data_threshold = 50

        # 1. Fine-tune (train equivalent)
        self.predictor.fine_tune(self.mock_streams)
        self.assertTrue(self.predictor.is_personal_fitted)

        # 2. Predict Flat (Grade 0)
        # Should be approx 3.0 m/s
        pred_flat = self.predictor.predict(0.0)
        self.assertAlmostEqual(pred_flat, 3.0, delta=0.5)

        # 3. Predict Uphill (Grade 10)
        # Should be approx 2.0 m/s
        pred_uphill = self.predictor.predict(10.0)
        self.assertAlmostEqual(pred_uphill, 2.0, delta=0.5)

    def test_predict_segment_pace(self) -> None:
        """Test segment pace prediction."""
        self.predictor.personal_data_threshold = 50
        self.predictor.fine_tune(self.mock_streams)

        # Grade 0 -> ~3.0 m/s is 1000/3 = 333s/km = 5.55 min/km
        pace = self.predictor.predict_segment(0.0)
        self.assertAlmostEqual(pace, 5.55, delta=0.5)

    def test_fallback_unfitted(self) -> None:
        """Test fallback logic when not fitted."""
        p = HybridPacePredictor()
        # Should rely on fallback logic
        val = p.predict(0.0)
        self.assertGreater(val, 0)

    def test_backwards_compatibility(self) -> None:
        """Test that PacePredictor alias works."""
        self.assertIs(PacePredictor, HybridPacePredictor)


if __name__ == "__main__":
    unittest.main()

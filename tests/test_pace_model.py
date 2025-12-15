import unittest
import numpy as np
from ultra_trail_strategist.feature_engineering.pace_model import PacePredictor

class TestPacePredictor(unittest.TestCase):
    
    def setUp(self):
        self.predictor = PacePredictor(model_type="linear") # Use linear for simpler deterministic testing
        
        # Create synthetic Training Data
        # Relation: Velocity = 3.0 - 0.1 * Grade (Flat 3m/s, uphill slows down)
        # Grade 0 -> 3.0 m/s
        # Grade 10 -> 2.0 m/s
        
        grades = np.linspace(-10, 20, 100).tolist()
        velocities = [(3.0 - 0.1 * g) for g in grades]
        
        # Mock Strava Stream structure
        self.mock_streams = [{
            "grade_smooth": {"data": grades},
            "velocity_smooth": {"data": velocities},
            "moving": {"data": [True] * 100}
        }]

    def test_training_and_prediction(self):
        # 1. Train
        self.predictor.train(self.mock_streams)
        self.assertTrue(self.predictor.is_fitted)
        
        # 2. Predict Flat (Grade 0)
        # Should be approx 3.0 m/s
        pred_flat = self.predictor.predict(0.0)
        self.assertAlmostEqual(pred_flat, 3.0, delta=0.1)
        
        # 3. Predict Uphill (Grade 10)
        # Should be approx 2.0 m/s
        pred_uphill = self.predictor.predict(10.0)
        self.assertAlmostEqual(pred_uphill, 2.0, delta=0.1)

    def test_predict_segment_pace(self):
        self.predictor.train(self.mock_streams)
        
        # Grade 0 -> 3.0 m/s is 1000/3 = 333s/km = 5.55 min/km
        pace = self.predictor.predict_segment(0.0)
        self.assertAlmostEqual(pace, 5.55, delta=0.1)

    def test_fallback_unfitted(self):
        p = PacePredictor()
        # Should rely on fallback logic
        val = p.predict(0.0)
        self.assertGreater(val, 0)

if __name__ == "__main__":
    unittest.main()

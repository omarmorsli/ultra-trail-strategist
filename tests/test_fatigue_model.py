import unittest
from ultra_trail_strategist.feature_engineering.fatigue_model import FatigueModel

class TestFatigueModel(unittest.TestCase):
    
    def test_depletion(self):
        # CP = 5:00 min/km (200 m/min)
        # Capacity = 1000 meters (D')
        model = FatigueModel(critical_pace_min_km=5.0, w_prime_balance=1000)
        
        # Run FAST: 4:00 min/km (250 m/min) for 1 km
        # Excess speed = 50 m/min
        # Time = 4 min
        # Expenditure = 50 * 4 = 200 meters
        
        new_bal = model.update_balance(segment_pace_min_km=4.0, segment_length_km=1.0)
        
        self.assertAlmostEqual(new_bal, 800.0) # 1000 - 200
        self.assertAlmostEqual(model.get_exhaustion_level(), 0.2)
        self.assertEqual(model.get_penalty_factor(), 1.0) # Not tired enough yet

    def test_bonk_penalty(self):
        # CP = 5:00
        model = FatigueModel(critical_pace_min_km=5.0, w_prime_balance=100)
        
        # Run VERY FAST to empty tank
        # 4:00 pace = 50 m/min excess. Need 2 mins to drain 100m. 
        # 2 mins at 4:00 pace is 0.5 km.
        model.update_balance(4.0, 0.5) 
        
        # Balance should be 0 or close
        self.assertLessEqual(model.current_balance, 0.0)
        self.assertAlmostEqual(model.get_exhaustion_level(), 1.0)
        
        # Penalty should be max (1.5)
        self.assertEqual(model.get_penalty_factor(), 1.5)

    def test_recovery(self):
        model = FatigueModel(5.0, 1000)
        model.current_balance = 500 # Start half empty
        
        # Run SLOW: 6:00 min/km (166.6 m/min)
        # CP Speed = 200
        # Recovery Speed = 33.3 m/min
        # Time for 1km = 6 min
        # Potential Recovery = 33.3 * 6 = 200m
        # Factor 0.5 -> 100m recovered
        
        new_bal = model.update_balance(6.0, 1.0)
        self.assertGreater(new_bal, 500)
        self.assertAlmostEqual(new_bal, 600, delta=1.0)

if __name__ == '__main__':
    unittest.main()

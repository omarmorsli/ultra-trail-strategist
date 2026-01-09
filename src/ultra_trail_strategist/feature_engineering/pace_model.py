import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

logger = logging.getLogger(__name__)


class PacePredictor:
    """
    Predicts running pace based on gradient and athlete history using Machine Learning.

    Target:
    - Predict 'moving_velocity' (m/s) based on 'grade_smooth' (%).
    - Can optionally include 'heartrate' if we want to simulate effort.
    """

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        if model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        else:
            self.model = LinearRegression()

        self.is_fitted = False

    def prepare_training_data(
        self, activities_streams: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts raw Strava streams into training features (X) and targets (y).

        X: [grade]
        y: velocity (m/s)
        """
        grades: List[float] = []
        velocities: List[float] = []

        for stream in activities_streams:
            # We expect 'grade_smooth' and 'velocity_smooth' keys.
            # Strava returns { "grade_smooth": { "data": [...] }, ... }
            # Or if pre-processed, just lists. Assuming cleaned dict format here from our client.

            # Handling generic Strava stream structure roughly:
            # Check if keys exist
            g_stream = stream.get("grade_smooth", {}).get("data", [])
            v_stream = stream.get("velocity_smooth", {}).get("data", [])
            m_stream = stream.get("moving", {}).get("data", [])  # boolean moving status

            if not g_stream or not v_stream:
                continue

            # Filter for moving points only
            g_arr = np.array(g_stream)
            v_arr = np.array(v_stream)

            if m_stream:
                m_arr = np.array(m_stream, dtype=bool)
                # Ensure lengths match
                min_len = min(len(g_arr), len(v_arr), len(m_arr))
                g_arr = g_arr[:min_len][m_arr[:min_len]]
                v_arr = v_arr[:min_len][m_arr[:min_len]]
            else:
                min_len = min(len(g_arr), len(v_arr))
                g_arr = g_arr[:min_len]
                v_arr = v_arr[:min_len]

            grades.extend(g_arr)
            velocities.extend(v_arr)

        X = np.array(grades).reshape(-1, 1)
        y = np.array(velocities)

        return X, y

    def train(self, activities_streams: List[Dict[str, Any]]) -> None:
        """
        Trains the model on provided activity streams.
        """
        logger.info(f"Preparing training data from {len(activities_streams)} activities...")
        X, y = self.prepare_training_data(activities_streams)

        if len(X) < 100:
            logger.warning("Not enough data points to train a reliable model.")
            return

        logger.info(f"Training {self.model_type} on {len(X)} points...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Model training complete.")

    def predict(self, grade: float) -> float:
        """
        Predicts velocity (m/s) for a single grade value.

        Parameters
        ----------
        grade : float
            Grade percentage (e.g., 10.0 for 10% uphill, -5.0 for 5% downhill).

        Returns
        -------
        float
            Predicted velocity in m/s.
        """
        if not self.is_fitted:
            # Fallback using Naismith's Rule (refined):
            # Base pace: 6 min/km (10 km/h = 2.78 m/s) for recreational runner
            # Uphill: +1 min/km per 10% grade
            # Downhill: -0.5 min/km per 10% grade (less benefit than climb penalty)
            base_pace_min_km = 6.0  # Recreational runner flat pace

            if grade > 0:
                # Uphill penalty: ~1 min/km per 10% grade
                pace_min_km = base_pace_min_km + (grade * 0.1)
            else:
                # Downhill benefit (capped - steep descents aren't faster)
                # Less benefit than climb penalty (technical terrain)
                benefit = min(abs(grade) * 0.05, 1.5)  # Cap at 1.5 min/km faster
                pace_min_km = base_pace_min_km - benefit

            # Safety bounds: never slower than 20 min/km or faster than 3 min/km
            pace_min_km = max(3.0, min(20.0, pace_min_km))

            # Convert pace (min/km) to velocity (m/s)
            velocity_ms = 1000 / (pace_min_km * 60)
            return float(velocity_ms)

        return float(self.model.predict([[grade]])[0])

    def predict_segment(self, avg_grade: float) -> float:
        """
        Predicts average pace (min/km) for a segment.
        """
        velocity_ms = self.predict(avg_grade)
        if velocity_ms <= 0.1:
            return 30.0  # Cap at 30 min/km for safety

        pace_min_km = (1000 / velocity_ms) / 60
        return pace_min_km

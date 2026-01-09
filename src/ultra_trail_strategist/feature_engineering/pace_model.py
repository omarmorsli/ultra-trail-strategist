"""
Hybrid Pace Predictor.

Combines:
1. Base model: Pre-trained on 253k Endomondo workouts
2. Personal model: Fine-tuned on athlete's Strava data
3. Ultra patterns: Adjusted for ultra-specific fatigue patterns
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor  # type: ignore[import-untyped]

from ultra_trail_strategist.feature_engineering.ultra_fatigue_adjuster import (
    UltraFatigueAdjuster,
)

logger = logging.getLogger(__name__)


class HybridPacePredictor:
    """
    Hybrid pace predictor combining base model, personal fine-tuning, and ultra patterns.

    The predictor uses a three-tier approach:
    1. **Base model**: Pre-trained on Endomondo dataset (253k workouts)
       - Provides robust predictions even with zero personal data
       - Captures general grade/velocity relationships

    2. **Personal model**: Fine-tuned on athlete's Strava activities
       - Activates when enough personal data is available (default: 100 segments)
       - Learns individual strengths (climbing, descending, etc.)

    3. **Ultra fatigue adjustment**: Applies race-specific degradation
       - Models pace slowdown over ultra distances
       - Accounts for night sections, altitude, etc.

    Example
    -------
    >>> predictor = HybridPacePredictor()
    >>> velocity = predictor.predict(grade=10.0)  # Uses base model
    >>> predictor.fine_tune(strava_activities)
    >>> velocity = predictor.predict(grade=10.0)  # Uses personal model
    """

    def __init__(
        self,
        base_model_path: Optional[Path] = None,
        personal_data_threshold: int = 100,
        model_dir: Path = Path("models"),
    ):
        """
        Initialize the hybrid predictor.

        Parameters
        ----------
        base_model_path : Optional[Path]
            Path to pre-trained base model. If None, uses fallback rules.
        personal_data_threshold : int
            Minimum number of data points required to use personal model.
        model_dir : Path
            Directory for model files.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.personal_data_threshold = personal_data_threshold

        # Base model (pre-trained on Endomondo)
        self.base_model: Optional[Any] = None
        self.base_model_features: List[str] = ["grade", "altitude"]

        if base_model_path:
            self._load_base_model(base_model_path)
        else:
            # Try default location
            default_path = self.model_dir / "endomondo_base.pkl"
            if default_path.exists():
                self._load_base_model(default_path)

        # Personal model (fine-tuned on athlete data)
        self.personal_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self.is_personal_fitted = False
        self.personal_data_count = 0

        # Ultra fatigue adjuster
        self.fatigue_adjuster = UltraFatigueAdjuster(model_dir=model_dir)

        # Feature statistics for normalization
        self._feature_stats: Dict[str, Dict[str, float]] = {}

    def _load_base_model(self, model_path: Path) -> None:
        """Load pre-trained base model."""
        try:
            with open(model_path, "rb") as f:
                self.base_model = pickle.load(f)

            # Try to load metadata
            meta_path = model_path.with_suffix("").with_suffix("_metadata.json")
            if meta_path.exists():
                import json
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                    self.base_model_features = metadata.get(
                        "feature_columns", self.base_model_features
                    )

            logger.info(f"Loaded base model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load base model: {e}")
            self.base_model = None

    def predict(
        self,
        grade: float,
        altitude: float = 0.0,
        heart_rate: Optional[float] = None,
        distance_into_race: Optional[float] = None,
        total_race_distance: Optional[float] = None,
        hour_of_day: Optional[int] = None,
    ) -> float:
        """
        Predict velocity (m/s) for given conditions.

        Parameters
        ----------
        grade : float
            Grade percentage (e.g., 10.0 for 10% uphill, -5.0 for 5% downhill).
        altitude : float
            Altitude in meters (for altitude adjustment).
        heart_rate : Optional[float]
            Target heart rate (if available).
        distance_into_race : Optional[float]
            Distance already covered in race (km).
        total_race_distance : Optional[float]
            Total race distance (km).
        hour_of_day : Optional[int]
            Current hour (0-23) for night adjustment.

        Returns
        -------
        float
            Predicted velocity in m/s.
        """
        # Get base prediction
        base_velocity = self._get_model_prediction(grade, altitude, heart_rate)

        # Apply ultra fatigue adjustment if race context provided
        if distance_into_race is not None and total_race_distance is not None:
            fatigue_mult = self.fatigue_adjuster.get_fatigue_multiplier(
                distance_km=distance_into_race,
                total_race_km=total_race_distance,
                hour_of_day=hour_of_day,
            )
            base_velocity /= fatigue_mult

        return max(0.1, base_velocity)  # Safety floor

    def _get_model_prediction(
        self,
        grade: float,
        altitude: float = 0.0,
        heart_rate: Optional[float] = None,
    ) -> float:
        """
        Get prediction from appropriate model.

        Uses personal model if fitted with enough data, otherwise base model,
        otherwise falls back to analytical rules.
        """
        # Priority 1: Personal model if fitted
        if self.is_personal_fitted and self.personal_data_count >= self.personal_data_threshold:
            features = self._prepare_features(grade, altitude, heart_rate)
            return float(self.personal_model.predict([features])[0])

        # Priority 2: Base model if available
        if self.base_model is not None:
            features = self._prepare_features(grade, altitude, heart_rate)
            try:
                return float(self.base_model.predict([features])[0])
            except Exception as e:
                logger.warning(f"Base model prediction failed: {e}")

        # Priority 3: Analytical fallback
        return self._fallback_prediction(grade)

    def _prepare_features(
        self,
        grade: float,
        altitude: float = 0.0,
        heart_rate: Optional[float] = None,
    ) -> List[float]:
        """Prepare feature vector for model input."""
        features = [grade, altitude]
        if heart_rate is not None and "heart_rate" in self.base_model_features:
            features.append(heart_rate)
        return features

    def _fallback_prediction(self, grade: float) -> float:
        """
        Analytical fallback using refined Naismith's Rule.

        This is used when no trained model is available.
        """
        # Base pace: 6 min/km (10 km/h = 2.78 m/s) for recreational runner
        base_pace_min_km = 6.0

        if grade > 0:
            # Uphill penalty: ~1 min/km per 10% grade
            pace_min_km = base_pace_min_km + (grade * 0.1)
        else:
            # Downhill benefit (capped - steep descents aren't faster)
            benefit = min(abs(grade) * 0.05, 1.5)  # Cap at 1.5 min/km faster
            pace_min_km = base_pace_min_km - benefit

        # Safety bounds
        pace_min_km = max(3.0, min(20.0, pace_min_km))

        # Convert pace (min/km) to velocity (m/s)
        velocity_ms = 1000 / (pace_min_km * 60)
        return velocity_ms

    def fine_tune(
        self,
        activities_streams: List[Dict[str, Any]],
        blend_with_base: bool = True,
        blend_ratio: float = 0.7,
    ) -> None:
        """
        Fine-tune the predictor with athlete's personal data.

        Parameters
        ----------
        activities_streams : List[Dict[str, Any]]
            List of Strava activity streams. Each should have:
            - grade_smooth.data: List of grade percentages
            - velocity_smooth.data: List of velocities (m/s)
            - altitude.data: List of altitudes (optional)
            - moving.data: List of moving flags (optional)
        blend_with_base : bool
            Whether to blend predictions with base model.
        blend_ratio : float
            Weight for personal model when blending (0-1).
        """
        logger.info(f"Fine-tuning on {len(activities_streams)} activities...")

        X, y = self._prepare_training_data(activities_streams)

        if len(X) < self.personal_data_threshold:
            logger.warning(
                f"Not enough data ({len(X)} < {self.personal_data_threshold}). "
                "Model will not be fine-tuned."
            )
            self.personal_data_count = len(X)
            return

        # If blending with base model, add base predictions as pseudo-targets
        if blend_with_base and self.base_model is not None:
            base_preds = self.base_model.predict(X)
            # Blend targets: personal data weighted, base model fills gaps
            y = blend_ratio * y + (1 - blend_ratio) * base_preds

        self.personal_model.fit(X, y)
        self.is_personal_fitted = True
        self.personal_data_count = len(X)

        logger.info(f"Fine-tuned on {len(X)} data points. Personal model active.")

    def _prepare_training_data(
        self,
        activities_streams: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert Strava streams into training features and targets.

        Returns X (features) and y (velocities).
        """
        all_grades = []
        all_altitudes = []
        all_velocities = []

        for stream in activities_streams:
            g_stream = stream.get("grade_smooth", {}).get("data", [])
            v_stream = stream.get("velocity_smooth", {}).get("data", [])
            a_stream = stream.get("altitude", {}).get("data", [])
            m_stream = stream.get("moving", {}).get("data", [])

            if not g_stream or not v_stream:
                continue

            g_arr = np.array(g_stream)
            v_arr = np.array(v_stream)
            a_arr = np.array(a_stream) if a_stream else np.zeros_like(g_arr)

            # Ensure lengths match
            min_len = min(len(g_arr), len(v_arr), len(a_arr))

            if m_stream:
                m_arr = np.array(m_stream, dtype=bool)[:min_len]
                g_arr = g_arr[:min_len][m_arr]
                v_arr = v_arr[:min_len][m_arr]
                a_arr = a_arr[:min_len][m_arr]
            else:
                g_arr = g_arr[:min_len]
                v_arr = v_arr[:min_len]
                a_arr = a_arr[:min_len]

            # Filter outliers
            valid_mask = (v_arr > 0.1) & (v_arr < 15) & (np.abs(g_arr) < 50)
            g_arr = g_arr[valid_mask]
            v_arr = v_arr[valid_mask]
            a_arr = a_arr[valid_mask]

            all_grades.extend(g_arr)
            all_altitudes.extend(a_arr)
            all_velocities.extend(v_arr)

        X = np.column_stack([all_grades, all_altitudes])
        y = np.array(all_velocities)

        return X, y

    def get_confidence_interval(
        self,
        grade: float,
        altitude: float = 0.0,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Return confidence interval for prediction.

        Parameters
        ----------
        grade : float
            Grade percentage.
        altitude : float
            Altitude in meters.
        confidence : float
            Confidence level (e.g., 0.95 for 95% CI).

        Returns
        -------
        Tuple[float, float]
            (lower_bound, upper_bound) velocity in m/s.
        """
        base_pred = self.predict(grade=grade, altitude=altitude)

        # Estimate uncertainty based on grade extremity
        # Steeper grades have more uncertainty
        grade_uncertainty = 0.05 + abs(grade) * 0.005  # 5% + 0.5% per grade %

        # If using personal model, reduce uncertainty
        if self.is_personal_fitted:
            grade_uncertainty *= 0.7

        # If using base model, slightly more uncertainty than personal
        elif self.base_model is not None:
            grade_uncertainty *= 0.85

        # Calculate bounds
        lower = base_pred * (1 - grade_uncertainty)
        upper = base_pred * (1 + grade_uncertainty)

        return (max(0.1, lower), upper)

    def predict_segment(self, avg_grade: float, altitude: float = 0.0) -> float:
        """
        Predict average pace (min/km) for a segment.

        Parameters
        ----------
        avg_grade : float
            Average grade percentage for the segment.
        altitude : float
            Average altitude in meters.

        Returns
        -------
        float
            Predicted pace in minutes per kilometer.
        """
        velocity_ms = self.predict(grade=avg_grade, altitude=altitude)
        if velocity_ms <= 0.1:
            return 30.0  # Cap at 30 min/km for safety

        pace_min_km = (1000 / velocity_ms) / 60
        return pace_min_km

    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all models.

        Returns
        -------
        Dict[str, Any]
            Status including which models are active and data counts.
        """
        return {
            "base_model_loaded": self.base_model is not None,
            "personal_model_fitted": self.is_personal_fitted,
            "personal_data_count": self.personal_data_count,
            "personal_threshold": self.personal_data_threshold,
            "using_personal": (
                self.is_personal_fitted and
                self.personal_data_count >= self.personal_data_threshold
            ),
            "feature_columns": self.base_model_features,
        }


# Backwards compatibility alias
PacePredictor = HybridPacePredictor

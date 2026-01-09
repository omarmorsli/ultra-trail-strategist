"""
Ultra Fatigue Adjuster.

Learns and applies pace degradation patterns from real ultra race results.
Based on analysis of UTMB, Western States, and similar ultra-distance events.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy.interpolate import interp1d  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class UltraFatigueAdjuster:
    """
    Models pace degradation patterns from real ultra race results.

    Key insights from ultra race data:
    - Pace drops ~15-25% after 100km
    - Night section paces drop 10-20%
    - Technical descents see 30%+ slower paces late in race
    - Elite vs recreational athletes have different degradation curves

    Example
    -------
    >>> adjuster = UltraFatigueAdjuster()
    >>> adjuster.fit_from_race_data(race_results)
    >>> multiplier = adjuster.get_fatigue_multiplier(distance_km=120, total_race_km=170)
    >>> adjusted_velocity = base_velocity / multiplier
    """

    def __init__(self, model_dir: Path = Path("models")):
        """
        Initialize the fatigue adjuster.

        Parameters
        ----------
        model_dir : Path
            Directory for saving/loading fatigue models.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Default degradation curve (based on research)
        # Format: (progress 0-1, pace_multiplier)
        self._default_curve = np.array([
            [0.0, 1.0],    # Start
            [0.1, 1.02],   # Slight warmup slowdown
            [0.2, 1.0],    # Back to normal
            [0.3, 1.03],   # Beginning of fatigue
            [0.4, 1.06],   # Moderate fatigue
            [0.5, 1.10],   # Significant fatigue
            [0.6, 1.15],   # Deep fatigue zone
            [0.7, 1.22],   # Heavy fatigue
            [0.8, 1.30],   # Severe fatigue
            [0.9, 1.40],   # Extreme fatigue
            [1.0, 1.50],   # Final push (some speed up, but averaged slower)
        ])

        self._degradation_func: Optional[interp1d] = None
        self._skill_adjustments: Dict[str, np.ndarray] = {}
        self._night_penalty = 1.12  # 12% slower at night

        # Load default curve
        self._build_interpolator(self._default_curve)

    def _build_interpolator(self, curve: np.ndarray) -> None:
        """Build interpolation function from curve data."""
        self._degradation_func = interp1d(
            curve[:, 0],
            curve[:, 1],
            kind="cubic",
            fill_value="extrapolate",
        )

    def fit_from_race_data(
        self,
        race_results: pl.DataFrame,
        min_athletes: int = 50,
    ) -> None:
        """
        Fit degradation curve from actual race results.

        Parameters
        ----------
        race_results : pl.DataFrame
            Race results with columns: athlete_id, checkpoint_order, pace_min_km.
        min_athletes : int
            Minimum number of athletes required for fitting.
        """
        if len(race_results) < min_athletes:
            logger.warning(
                f"Not enough data ({len(race_results)} < {min_athletes}), using default curve"
            )
            return

        # Calculate normalized progress for each checkpoint
        max_order = race_results["checkpoint_order"].max()
        results = race_results.with_columns(
            (pl.col("checkpoint_order") / max_order).alias("progress")
        )

        # Group by athlete and calculate their degradation
        athlete_curves = (
            results.group_by("athlete_id")
            .agg([
                pl.col("pace_min_km").first().alias("initial_pace"),
                pl.struct(["progress", "pace_min_km"]).alias("checkpoints"),
            ])
        )

        # Calculate average degradation at each progress point
        # First, normalize each athlete's pace to their initial pace
        progress_points = np.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0
        multipliers = []

        for progress in progress_points:
            # Find nearest checkpoint for each athlete
            nearby = results.filter(
                (pl.col("progress") >= progress - 0.05) &
                (pl.col("progress") <= progress + 0.05)
            )

            if nearby.is_empty():
                multipliers.append(self._default_curve[int(progress * 10), 1])
                continue

            # Join with initial pace to calculate multiplier
            athlete_initials = (
                results.filter(pl.col("progress") <= 0.1)
                .group_by("athlete_id")
                .agg(pl.col("pace_min_km").mean().alias("initial_pace"))
            )

            joined = nearby.join(athlete_initials, on="athlete_id", how="left")
            joined = joined.with_columns(
                (pl.col("pace_min_km") / pl.col("initial_pace")).alias("multiplier")
            )

            avg_multiplier = joined["multiplier"].mean()
            multipliers.append(
                float(avg_multiplier) if avg_multiplier is not None else 1.0  # type: ignore[arg-type]
            )

        # Build new curve
        fitted_curve = np.column_stack([progress_points, multipliers])
        self._build_interpolator(fitted_curve)

        logger.info(f"Fitted degradation curve from {athlete_curves.height} athletes")

    def fit_by_skill_level(
        self,
        race_results: pl.DataFrame,
        n_skill_bins: int = 3,
    ) -> None:
        """
        Fit separate degradation curves for different skill levels.

        Parameters
        ----------
        race_results : pl.DataFrame
            Race results with skill_level column (0=elite, 1=back-of-pack).
        n_skill_bins : int
            Number of skill level bins.
        """
        skill_labels = ["elite", "mid_pack", "back_pack"][:n_skill_bins]

        for i, label in enumerate(skill_labels):
            lower = i / n_skill_bins
            upper = (i + 1) / n_skill_bins

            filtered = race_results.filter(
                (pl.col("skill_level") >= lower) &
                (pl.col("skill_level") < upper)
            )

            if len(filtered) < 10:
                continue

            # Calculate curve for this skill level
            max_order = filtered["checkpoint_order"].max()
            progress_points = np.linspace(0, 1, 11)
            multipliers = []

            for progress in progress_points:
                nearby = filtered.filter(
                    (pl.col("checkpoint_order") / max_order >= progress - 0.05) &
                    (pl.col("checkpoint_order") / max_order <= progress + 0.05)
                )

                avg_fatigue = nearby["fatigue_factor"].mean() if not nearby.is_empty() else 1.0
                multipliers.append(
                    float(avg_fatigue) if avg_fatigue is not None else 1.0  # type: ignore[arg-type]
                )

            self._skill_adjustments[label] = np.column_stack([progress_points, multipliers])
            logger.info(f"Fitted {label} degradation curve")

    def get_fatigue_multiplier(
        self,
        distance_km: float,
        total_race_km: float,
        hour_of_day: Optional[int] = None,
        skill_level: Optional[str] = None,
    ) -> float:
        """
        Get pace multiplier based on fatigue.

        Parameters
        ----------
        distance_km : float
            Current distance into the race.
        total_race_km : float
            Total race distance.
        hour_of_day : Optional[int]
            Hour (0-23) for night penalty calculation.
        skill_level : Optional[str]
            Skill level ('elite', 'mid_pack', 'back_pack').

        Returns
        -------
        float
            Pace multiplier (1.0 = no penalty, 1.25 = 25% slower).
        """
        if total_race_km <= 0:
            return 1.0

        progress = min(1.0, distance_km / total_race_km)

        # Get base fatigue from curve
        if skill_level and skill_level in self._skill_adjustments:
            curve = self._skill_adjustments[skill_level]
            func = interp1d(curve[:, 0], curve[:, 1], kind="cubic", fill_value="extrapolate")
            base_fatigue = float(func(progress))
        elif self._degradation_func is not None:
            base_fatigue = float(self._degradation_func(progress))
        else:
            base_fatigue = 1.0

        # Apply night penalty
        night_factor = 1.0
        if hour_of_day is not None:
            # Night hours: 22:00 - 06:00
            if hour_of_day >= 22 or hour_of_day < 6:
                night_factor = self._night_penalty
            # Twilight: slightly slower
            elif hour_of_day >= 19 or hour_of_day < 7:
                night_factor = (self._night_penalty + 1.0) / 2

        return base_fatigue * night_factor

    def predict_finish_time(
        self,
        segments: List[Dict[str, Any]],
        base_velocities: List[float],
        start_hour: int = 6,
    ) -> Tuple[float, List[float]]:
        """
        Predict total finish time accounting for fatigue.

        Parameters
        ----------
        segments : List[Dict[str, Any]]
            List of segments with 'length' (km) keys.
        base_velocities : List[float]
            Base predicted velocity (m/s) for each segment without fatigue.
        start_hour : int
            Race start hour (24-hour format).

        Returns
        -------
        Tuple[float, List[float]]
            Total time in seconds and list of segment times.
        """
        total_distance = sum(s.get("length", 0) for s in segments)
        cumulative_distance = 0.0
        cumulative_time = 0.0
        segment_times = []

        for segment, base_velocity in zip(segments, base_velocities, strict=True):
            segment_length = segment.get("length", 0)

            # Calculate fatigue at this point
            fatigue_mult = self.get_fatigue_multiplier(
                distance_km=cumulative_distance,
                total_race_km=total_distance,
                hour_of_day=(start_hour + int(cumulative_time / 3600)) % 24,
            )

            # Adjust velocity
            adjusted_velocity = base_velocity / fatigue_mult

            # Calculate time for segment
            if adjusted_velocity > 0.1:
                segment_time = (segment_length * 1000) / adjusted_velocity
            else:
                segment_time = segment_length * 1000  # 1 m/s fallback

            segment_times.append(segment_time)
            cumulative_time += segment_time
            cumulative_distance += segment_length

        return cumulative_time, segment_times

    def save(self, filename: str = "ultra_fatigue_curve.pkl") -> Path:
        """Save the fitted fatigue model."""
        filepath = self.model_dir / filename

        data = {
            "default_curve": self._default_curve,
            "skill_adjustments": self._skill_adjustments,
            "night_penalty": self._night_penalty,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved fatigue model to {filepath}")
        return filepath

    def load(self, filename: str = "ultra_fatigue_curve.pkl") -> None:
        """Load a previously fitted fatigue model."""
        filepath = self.model_dir / filename

        if not filepath.exists():
            logger.warning(f"No fatigue model found at {filepath}, using defaults")
            return

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self._default_curve = data.get("default_curve", self._default_curve)
        self._skill_adjustments = data.get("skill_adjustments", {})
        self._night_penalty = data.get("night_penalty", self._night_penalty)

        self._build_interpolator(self._default_curve)
        logger.info(f"Loaded fatigue model from {filepath}")

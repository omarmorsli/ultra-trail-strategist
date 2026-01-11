"""
Heart Rate Zone Calculator.

Calculates heart rate zones using the Karvonen method (Heart Rate Reserve).
Zones 1-5 represent increasing effort levels from recovery to VO2max.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ZoneInfo:
    """Information about a heart rate zone."""

    zone: int
    name: str
    description: str
    hr_range: Tuple[int, int]


class HRZoneCalculator:
    """
    Calculate heart rate zones using the Karvonen method.

    The Karvonen method uses Heart Rate Reserve (HRR) for more accurate
    zone calculation that accounts for individual fitness levels.

    HRR = Max HR - Resting HR
    Target HR = (HRR × Zone %) + Resting HR

    Zones
    -----
    1 (50-60% HRR): Recovery - Easy effort, conversation pace
    2 (60-70% HRR): Aerobic base - Steady endurance building
    3 (70-80% HRR): Tempo - Comfortably hard, sustainable for hours
    4 (80-90% HRR): Threshold - Race pace, lactate threshold
    5 (90-100% HRR): VO2max - Maximum effort, intervals

    Example
    -------
    >>> calc = HRZoneCalculator(max_hr=185, resting_hr=55)
    >>> calc.get_zone(145)  # Returns zone number
    3
    >>> calc.get_zone_info(145)  # Returns full zone information
    ZoneInfo(zone=3, name='Tempo', ...)
    """

    ZONE_THRESHOLDS: dict[int, Tuple[float, float]] = {
        1: (0.50, 0.60),  # Recovery
        2: (0.60, 0.70),  # Aerobic base
        3: (0.70, 0.80),  # Tempo
        4: (0.80, 0.90),  # Threshold
        5: (0.90, 1.00),  # VO2max
    }

    ZONE_NAMES: dict[int, Tuple[str, str]] = {
        1: ("Recovery", "Easy effort, active recovery"),
        2: ("Aerobic", "Endurance building, fat burning"),
        3: ("Tempo", "Comfortably hard, marathon pace"),
        4: ("Threshold", "Lactate threshold, half-marathon pace"),
        5: ("VO2max", "Maximum effort, sprint intervals"),
    }

    def __init__(self, max_hr: int, resting_hr: int = 60):
        """
        Initialize the zone calculator.

        Parameters
        ----------
        max_hr : int
            Maximum heart rate (can be measured or estimated from age).
        resting_hr : int
            Resting heart rate (measured in the morning).
        """
        if max_hr <= resting_hr:
            raise ValueError("Max HR must be greater than resting HR")
        if max_hr < 100 or max_hr > 220:
            raise ValueError("Max HR should be between 100 and 220")
        if resting_hr < 30 or resting_hr > 100:
            raise ValueError("Resting HR should be between 30 and 100")

        self.max_hr = max_hr
        self.resting_hr = resting_hr
        self.hr_reserve = max_hr - resting_hr

    @classmethod
    def from_age(cls, age: int, resting_hr: int = 60) -> "HRZoneCalculator":
        """
        Create calculator by estimating max HR from age.

        Uses the standard formula: Max HR = 220 - age
        (More accurate formulas exist but this is widely used)

        Parameters
        ----------
        age : int
            Athlete's age in years.
        resting_hr : int
            Resting heart rate.

        Returns
        -------
        HRZoneCalculator
            Configured calculator instance.
        """
        if age < 10 or age > 100:
            raise ValueError("Age should be between 10 and 100")

        max_hr = 220 - age
        return cls(max_hr=max_hr, resting_hr=resting_hr)

    def get_hrr_percentage(self, heart_rate: float) -> float:
        """
        Calculate heart rate as percentage of Heart Rate Reserve.

        Parameters
        ----------
        heart_rate : float
            Current heart rate in bpm.

        Returns
        -------
        float
            Percentage of HRR (0.0 - 1.0+)
        """
        if heart_rate <= self.resting_hr:
            return 0.0
        return (heart_rate - self.resting_hr) / self.hr_reserve

    def get_zone(self, heart_rate: float) -> int:
        """
        Determine heart rate zone (1-5) for given heart rate.

        Parameters
        ----------
        heart_rate : float
            Current heart rate in bpm.

        Returns
        -------
        int
            Zone number 1-5 (returns 1 if below Zone 1, 5 if above Zone 5).
        """
        hrr_pct = self.get_hrr_percentage(heart_rate)

        # Check each zone threshold
        for zone, (low, high) in self.ZONE_THRESHOLDS.items():
            if low <= hrr_pct < high:
                return zone

        # Below Zone 1 or above Zone 5
        if hrr_pct < 0.50:
            return 1
        return 5

    def get_zone_info(self, heart_rate: float) -> ZoneInfo:
        """
        Get detailed zone information for given heart rate.

        Parameters
        ----------
        heart_rate : float
            Current heart rate in bpm.

        Returns
        -------
        ZoneInfo
            Full zone information including name, description, HR range.
        """
        zone = self.get_zone(heart_rate)
        name, description = self.ZONE_NAMES[zone]
        low, high = self.ZONE_THRESHOLDS[zone]

        hr_low = int(self.resting_hr + (self.hr_reserve * low))
        hr_high = int(self.resting_hr + (self.hr_reserve * high))

        return ZoneInfo(
            zone=zone,
            name=name,
            description=description,
            hr_range=(hr_low, hr_high),
        )

    def get_zone_boundaries(self) -> dict[int, Tuple[int, int]]:
        """
        Calculate HR boundaries for all zones.

        Returns
        -------
        dict[int, Tuple[int, int]]
            Zone number to (low_hr, high_hr) mapping.
        """
        boundaries = {}
        for zone, (low_pct, high_pct) in self.ZONE_THRESHOLDS.items():
            hr_low = int(self.resting_hr + (self.hr_reserve * low_pct))
            hr_high = int(self.resting_hr + (self.hr_reserve * high_pct))
            boundaries[zone] = (hr_low, hr_high)
        return boundaries

    def estimate_max_hr_from_workout(
        self,
        heart_rates: list[float],
        buffer_pct: float = 0.05,
    ) -> Optional[int]:
        """
        Estimate max HR from workout heart rate data.

        Takes the maximum observed HR and adds a buffer since most workouts
        don't reach true max HR.

        Parameters
        ----------
        heart_rates : list[float]
            List of heart rate values from a workout.
        buffer_pct : float
            Buffer to add to observed max (default 5%).

        Returns
        -------
        Optional[int]
            Estimated max HR, or None if insufficient data.
        """
        if not heart_rates or len(heart_rates) < 10:
            return None

        # Use 95th percentile to avoid spikes/noise
        sorted_hr = sorted(heart_rates)
        p95_idx = int(len(sorted_hr) * 0.95)
        observed_max = sorted_hr[p95_idx]

        return int(observed_max * (1 + buffer_pct))

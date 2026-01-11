"""Tests for the Heart Rate Zone Calculator."""

import pytest

from ultra_trail_strategist.feature_engineering.hr_zone_calculator import (
    HRZoneCalculator,
    ZoneInfo,
)


class TestHRZoneCalculator:
    """Test suite for HRZoneCalculator."""

    @pytest.fixture
    def calculator(self) -> HRZoneCalculator:
        """Create a standard calculator (max HR 180, resting HR 60)."""
        return HRZoneCalculator(max_hr=180, resting_hr=60)

    def test_init_basic(self, calculator: HRZoneCalculator) -> None:
        """Test basic initialization."""
        assert calculator.max_hr == 180
        assert calculator.resting_hr == 60
        assert calculator.hr_reserve == 120  # 180 - 60

    def test_init_invalid_max_hr_low(self) -> None:
        """Test that low max HR raises error."""
        with pytest.raises(ValueError, match="between 100 and 220"):
            HRZoneCalculator(max_hr=90, resting_hr=60)

    def test_init_invalid_max_hr_high(self) -> None:
        """Test that high max HR raises error."""
        with pytest.raises(ValueError, match="between 100 and 220"):
            HRZoneCalculator(max_hr=250, resting_hr=60)

    def test_init_max_less_than_resting(self) -> None:
        """Test that max HR <= resting HR raises error."""
        with pytest.raises(ValueError, match="must be greater than"):
            HRZoneCalculator(max_hr=120, resting_hr=130)

    def test_from_age(self) -> None:
        """Test creating calculator from age."""
        # 30 year old: max HR = 220 - 30 = 190
        calc = HRZoneCalculator.from_age(age=30, resting_hr=55)
        assert calc.max_hr == 190
        assert calc.resting_hr == 55

    def test_from_age_invalid(self) -> None:
        """Test that invalid age raises error."""
        with pytest.raises(ValueError, match="between 10 and 100"):
            HRZoneCalculator.from_age(age=5)

    def test_get_hrr_percentage(self, calculator: HRZoneCalculator) -> None:
        """Test HRR percentage calculation."""
        # At resting HR, should be 0%
        assert calculator.get_hrr_percentage(60) == 0.0
        
        # At max HR, should be 100%
        assert calculator.get_hrr_percentage(180) == 1.0
        
        # At midpoint (120), should be 50%
        assert calculator.get_hrr_percentage(120) == pytest.approx(0.5)

    def test_get_hrr_percentage_below_resting(self, calculator: HRZoneCalculator) -> None:
        """Test HRR percentage when HR is below resting."""
        assert calculator.get_hrr_percentage(55) == 0.0

    def test_get_zone_zone1(self, calculator: HRZoneCalculator) -> None:
        """Test Zone 1 detection (50-60% HRR)."""
        # Zone 1: 60 + (120 * 0.50) = 120 to 60 + (120 * 0.60) = 132
        assert calculator.get_zone(120) == 1
        assert calculator.get_zone(126) == 1
        assert calculator.get_zone(131) == 1

    def test_get_zone_zone2(self, calculator: HRZoneCalculator) -> None:
        """Test Zone 2 detection (60-70% HRR)."""
        # Zone 2: 132 to 144
        assert calculator.get_zone(133) == 2
        assert calculator.get_zone(140) == 2

    def test_get_zone_zone3(self, calculator: HRZoneCalculator) -> None:
        """Test Zone 3 detection (70-80% HRR)."""
        # Zone 3: 144 to 156
        assert calculator.get_zone(145) == 3
        assert calculator.get_zone(155) == 3

    def test_get_zone_zone4(self, calculator: HRZoneCalculator) -> None:
        """Test Zone 4 detection (80-90% HRR)."""
        # Zone 4: 156 to 168
        assert calculator.get_zone(157) == 4
        assert calculator.get_zone(167) == 4

    def test_get_zone_zone5(self, calculator: HRZoneCalculator) -> None:
        """Test Zone 5 detection (90-100% HRR)."""
        # Zone 5: 168 to 180
        assert calculator.get_zone(169) == 5
        assert calculator.get_zone(180) == 5

    def test_get_zone_below_zone1(self, calculator: HRZoneCalculator) -> None:
        """Test that very low HR returns Zone 1."""
        assert calculator.get_zone(70) == 1
        assert calculator.get_zone(100) == 1

    def test_get_zone_above_zone5(self, calculator: HRZoneCalculator) -> None:
        """Test that very high HR returns Zone 5."""
        assert calculator.get_zone(190) == 5

    def test_get_zone_info(self, calculator: HRZoneCalculator) -> None:
        """Test zone info retrieval."""
        info = calculator.get_zone_info(145)
        
        assert isinstance(info, ZoneInfo)
        assert info.zone == 3
        assert info.name == "Tempo"
        assert "marathon" in info.description.lower() or "comfortably" in info.description.lower()
        assert isinstance(info.hr_range, tuple)
        assert len(info.hr_range) == 2

    def test_get_zone_boundaries(self, calculator: HRZoneCalculator) -> None:
        """Test zone boundary calculation."""
        boundaries = calculator.get_zone_boundaries()
        
        assert len(boundaries) == 5
        assert 1 in boundaries
        assert 5 in boundaries
        
        # Zone 1 should start at 50% HRR
        # 60 + (120 * 0.50) = 120
        assert boundaries[1][0] == 120
        
        # Zone 5 should end at max HR
        assert boundaries[5][1] == 180

    def test_estimate_max_hr_from_workout(self, calculator: HRZoneCalculator) -> None:
        """Test max HR estimation from workout data."""
        # Simulate a workout with max HR around 170
        heart_rates = [100.0, 120.0, 140.0, 155.0, 165.0, 170.0, 165.0, 150.0, 130.0, 110.0]
        
        estimated = calculator.estimate_max_hr_from_workout(heart_rates)
        
        assert estimated is not None
        # Should be roughly 170 * 1.05 = ~178
        assert 175 <= estimated <= 185

    def test_estimate_max_hr_insufficient_data(self, calculator: HRZoneCalculator) -> None:
        """Test that insufficient data returns None."""
        assert calculator.estimate_max_hr_from_workout([120.0, 130.0]) is None
        assert calculator.estimate_max_hr_from_workout([]) is None

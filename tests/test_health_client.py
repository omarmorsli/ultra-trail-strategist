"""
Tests for HealthClient - fetches athlete readiness from Garmin or manual input.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestHealthClient(unittest.TestCase):
    """Tests for HealthClient readiness score logic."""

    @patch.dict("os.environ", {"GARMIN_EMAIL": "", "GARMIN_PASSWORD": ""}, clear=False)
    def test_manual_override_priority(self):
        """Test that manual override takes highest priority."""
        from ultra_trail_strategist.data_ingestion.health_client import HealthClient

        client = HealthClient()
        score = client.get_readiness_score(manual_override=85)
        self.assertEqual(score, 85)

    @patch.dict("os.environ", {"GARMIN_EMAIL": "", "GARMIN_PASSWORD": ""}, clear=False)
    def test_manual_override_clamped_high(self):
        """Test that manual override is clamped to 100."""
        from ultra_trail_strategist.data_ingestion.health_client import HealthClient

        client = HealthClient()
        score = client.get_readiness_score(manual_override=150)
        self.assertEqual(score, 100)

    @patch.dict("os.environ", {"GARMIN_EMAIL": "", "GARMIN_PASSWORD": ""}, clear=False)
    def test_manual_override_clamped_low(self):
        """Test that manual override is clamped to 0."""
        from ultra_trail_strategist.data_ingestion.health_client import HealthClient

        client = HealthClient()
        score = client.get_readiness_score(manual_override=-10)
        self.assertEqual(score, 0)

    @patch.dict("os.environ", {"GARMIN_EMAIL": "", "GARMIN_PASSWORD": ""}, clear=False)
    def test_default_without_garmin(self):
        """Test default value when Garmin is not configured."""
        from ultra_trail_strategist.data_ingestion.health_client import HealthClient

        client = HealthClient()
        score = client.get_readiness_score()
        # Should return 50 (default) when no Garmin
        self.assertEqual(score, 50)

    @patch.dict(
        "os.environ", {"GARMIN_EMAIL": "test@test.com", "GARMIN_PASSWORD": "pass"}, clear=False
    )
    @patch("ultra_trail_strategist.data_ingestion.health_client.garminconnect.Garmin")
    def test_garmin_training_readiness(self, mock_garmin_cls):
        """Test fetching Training Readiness from Garmin."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.get_training_readiness.return_value = {"score": 78}
        mock_garmin_cls.return_value = mock_client

        from ultra_trail_strategist.data_ingestion.health_client import HealthClient

        client = HealthClient()
        score = client.get_readiness_score()

        self.assertEqual(score, 78)

    @patch.dict(
        "os.environ", {"GARMIN_EMAIL": "test@test.com", "GARMIN_PASSWORD": "pass"}, clear=False
    )
    @patch("ultra_trail_strategist.data_ingestion.health_client.garminconnect.Garmin")
    def test_garmin_body_battery_fallback(self, mock_garmin_cls):
        """Test fallback to Body Battery when Training Readiness unavailable."""
        # Setup mock - training readiness fails, body battery works
        mock_client = MagicMock()
        mock_client.get_training_readiness.return_value = None  # No training readiness
        mock_client.get_body_battery.return_value = [{"bodyBattery": 65}]
        mock_garmin_cls.return_value = mock_client

        from ultra_trail_strategist.data_ingestion.health_client import HealthClient

        client = HealthClient()
        score = client.get_readiness_score()

        self.assertEqual(score, 65)

    @patch.dict(
        "os.environ", {"GARMIN_EMAIL": "test@test.com", "GARMIN_PASSWORD": "pass"}, clear=False
    )
    @patch("ultra_trail_strategist.data_ingestion.health_client.garminconnect.Garmin")
    def test_garmin_login_failure(self, mock_garmin_cls):
        """Test handling of Garmin login failure."""
        mock_garmin_cls.return_value.login.side_effect = Exception("Login failed")

        from ultra_trail_strategist.data_ingestion.health_client import HealthClient

        client = HealthClient()
        # Should fall back to default
        score = client.get_readiness_score()

        self.assertEqual(score, 50)

    @patch.dict(
        "os.environ", {"GARMIN_EMAIL": "test@test.com", "GARMIN_PASSWORD": "pass"}, clear=False
    )
    @patch("ultra_trail_strategist.data_ingestion.health_client.garminconnect.Garmin")
    def test_garmin_api_failure_returns_default(self, mock_garmin_cls):
        """Test that API failures return default value."""
        mock_client = MagicMock()
        mock_client.get_training_readiness.side_effect = Exception("API Error")
        mock_client.get_body_battery.side_effect = Exception("API Error")
        mock_client.get_user_summary.side_effect = Exception("API Error")
        mock_garmin_cls.return_value = mock_client

        from ultra_trail_strategist.data_ingestion.health_client import HealthClient

        client = HealthClient()
        score = client.get_readiness_score()

        # Should return default when all APIs fail
        self.assertEqual(score, 50)


if __name__ == "__main__":
    unittest.main()

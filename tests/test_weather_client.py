import unittest
from unittest.mock import Mock

from ultra_trail_strategist.data_ingestion.weather_client import WeatherClient


class TestWeatherClient(unittest.TestCase):
    def test_get_forecast(self):
        # Setup
        client = WeatherClient()
        client.session = Mock()

        # Mock Response
        mock_response = Mock()
        mock_response.json.return_value = {
            "hourly": {"temperature_2m": [10.0, 11.0, 12.0], "precipitation": [0.0, 0.0, 0.5]}
        }
        mock_response.raise_for_status = Mock()
        client.session.get.return_value = mock_response

        # Execute
        data = client.get_forecast(45.0, 6.0)

        # Verify
        self.assertIn("hourly", data)
        self.assertEqual(data["hourly"]["temperature_2m"][0], 10.0)
        client.session.get.assert_called_once()

    def test_get_current_conditions_rainy(self):
        # Setup
        client = WeatherClient()
        client.session = Mock()

        mock_response = Mock()
        mock_response.json.return_value = {
            "hourly": {
                "temperature_2m": [10.0] * 12,
                "precipitation": [1.0] * 12,  # 12mm total > 5mm -> Heavy Rain
            }
        }
        mock_response.raise_for_status = Mock()
        client.session.get.return_value = mock_response

        # Execute
        summary = client.get_current_conditions(45.0, 6.0)

        # Verify
        self.assertIn("Heavy Rain", summary)
        self.assertIn("10.0°C", summary)


if __name__ == "__main__":
    unittest.main()

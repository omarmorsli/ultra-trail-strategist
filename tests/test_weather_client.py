import unittest
from unittest.mock import patch, Mock
from ultra_trail_strategist.data_ingestion.weather_client import WeatherClient

class TestWeatherClient(unittest.TestCase):
    
    @patch("ultra_trail_strategist.data_ingestion.weather_client.requests.get")
    def test_get_forecast(self, mock_get):
        # Mock Response
        mock_response = Mock()
        mock_response.json.return_value = {
            "hourly": {
                "temperature_2m": [10.0, 11.0, 12.0],
                "precipitation": [0.0, 0.0, 0.5]
            }
        }
        mock_get.return_value = mock_response
        
        client = WeatherClient()
        data = client.get_forecast(45.0, 6.0)
        
        self.assertIn("hourly", data)
        self.assertEqual(data["hourly"]["temperature_2m"][0], 10.0)

    @patch("ultra_trail_strategist.data_ingestion.weather_client.requests.get")
    def test_get_current_conditions_rainy(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "hourly": {
                "temperature_2m": [10.0] * 12,
                "precipitation": [1.0] * 12  # 12mm total > 5mm -> Heavy? No, Wait logic is > 5 total 
            }
        }
        mock_get.return_value = mock_response
        
        client = WeatherClient()
        summary = client.get_current_conditions(45.0, 6.0)
        
        self.assertIn("Heavy Rain", summary)
        self.assertIn("10.0°C", summary)

if __name__ == "__main__":
    unittest.main()

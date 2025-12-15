import unittest
from unittest.mock import patch, MagicMock
from ultra_trail_strategist.data_ingestion.strava_client import StravaClient, StravaTokenResponse

class TestStravaClient(unittest.TestCase):

    @patch("ultra_trail_strategist.data_ingestion.strava_client.settings")
    def setUp(self, mock_settings):
        # Mock settings values
        mock_settings.STRAVA_CLIENT_ID = "123"
        mock_settings.STRAVA_CLIENT_SECRET.get_secret_value.return_value = "secret"
        mock_settings.STRAVA_REFRESH_TOKEN.get_secret_value.return_value = "refresh"
        mock_settings.STRAVA_BASE_URL = "http://mock-api.com"
        mock_settings.PAGE_SIZE = 10
        
        self.client = StravaClient()

    @patch("ultra_trail_strategist.data_ingestion.strava_client.requests.post")
    def test_refresh_access_token_success(self, mock_post):
        # Mock successful token refresh response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_at": 1234567890,
            "expires_in": 3600
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.client._refresh_access_token()

        self.assertEqual(self.client.access_token, "new_access_token")
        self.assertEqual(self.client.refresh_token, "new_refresh_token")
        self.assertEqual(self.client.token_expires_at, 1234567890)

    @patch("ultra_trail_strategist.data_ingestion.strava_client.requests.Session.get")
    def test_get_athlete_activities_pagination(self, mock_get):
        self.client.access_token = "valid_token"
        self.client.token_expires_at = 9999999999 # Future
        
        # WE MUST override the global mock settings for this specific test
        # or just assume the code relies on the settings module we patched in setUp.
        # However, setUp patches 'ultra_trail_strategist.data_ingestion.strava_client.settings'
        # Let's adjust that patch object.
        from ultra_trail_strategist.data_ingestion.strava_client import settings
        settings.PAGE_SIZE = 2

        # Mock two pages of results
        mock_response_p1 = MagicMock()
        mock_response_p1.json.return_value = [{"id": 1}, {"id": 2}] # Full page
        mock_response_p1.raise_for_status.return_value = None
        
        mock_response_p2 = MagicMock()
        mock_response_p2.json.return_value = [{"id": 3}] # Partial page
        mock_response_p2.raise_for_status.return_value = None

        mock_response_empty = MagicMock()
        mock_response_empty.json.return_value = []
        mock_response_empty.raise_for_status.return_value = None

        # Side effect to return different responses for subsequent calls
        mock_get.side_effect = [mock_response_p1, mock_response_p2, mock_response_empty]

        activities = list(self.client.get_athlete_activities())
        
        self.assertEqual(len(activities), 3)
        self.assertEqual(activities[0]["id"], 1)
        self.assertEqual(activities[2]["id"], 3)
        # It should call get twice: once for p1 (full), once for p2 (partial -> break)
        self.assertEqual(mock_get.call_count, 2)

if __name__ == "__main__":
    unittest.main()

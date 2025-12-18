import unittest
from unittest.mock import MagicMock, patch

from ultra_trail_strategist.data_ingestion.strava_client import StravaClient


class TestStravaClient(unittest.TestCase):
    def setUp(self):
        # We can just override the settings object directly since it's a singleton
        # But patching is safer to not pollute other tests.
        self.patcher = patch("ultra_trail_strategist.data_ingestion.strava_client.settings")
        self.mock_settings = self.patcher.start()

        self.mock_settings.STRAVA_CLIENT_ID = "123"
        self.mock_settings.STRAVA_CLIENT_SECRET.get_secret_value.return_value = "secret"
        self.mock_settings.STRAVA_REFRESH_TOKEN.get_secret_value.return_value = "refresh"
        self.mock_settings.STRAVA_BASE_URL = "http://mock-api.com"
        self.mock_settings.PAGE_SIZE = 2

        self.client = StravaClient()
        self.client.session = MagicMock()  # Mock the CachedSession

    def tearDown(self):
        self.patcher.stop()

    @patch("ultra_trail_strategist.data_ingestion.strava_client.requests.post")
    def test_refresh_access_token_success(self, mock_post):
        # Mock successful token refresh response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_at": 1234567890,
            "expires_in": 3600,
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.client._refresh_access_token()

        self.assertEqual(self.client.access_token, "new_access_token")
        self.assertEqual(self.client.refresh_token, "new_refresh_token")
        self.assertEqual(self.client.token_expires_at, 1234567890)

    def test_get_athlete_activities_pagination(self):
        self.client.access_token = "valid_token"
        self.client.token_expires_at = 9999999999

        # Mock two pages of results
        mock_response_p1 = MagicMock()
        mock_response_p1.json.return_value = [{"id": 1}, {"id": 2}]
        mock_response_p1.raise_for_status.return_value = None

        mock_response_p2 = MagicMock()
        mock_response_p2.json.return_value = [{"id": 3}]
        mock_response_p2.raise_for_status.return_value = None

        mock_response_empty = MagicMock()
        mock_response_empty.json.return_value = []
        mock_response_empty.raise_for_status.return_value = None

        self.client.session.get.side_effect = [  # type: ignore
            mock_response_p1,
            mock_response_p2,
            mock_response_empty,
        ]

        activities = list(self.client.get_athlete_activities())

        self.assertEqual(len(activities), 3)
        self.assertEqual(activities[0]["id"], 1)
        self.assertEqual(activities[2]["id"], 3)
        self.assertEqual(self.client.session.get.call_count, 2)  # type: ignore


if __name__ == "__main__":
    unittest.main()

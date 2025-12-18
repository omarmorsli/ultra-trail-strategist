import asyncio
import unittest
from unittest.mock import patch

from ultra_trail_strategist.mcp_server import get_recent_activities


class TestStravaMCPServer(unittest.TestCase):
    @patch("ultra_trail_strategist.mcp_server.strava")
    def test_get_recent_activities(self, mock_strava):
        # Mock the underlying StravaClient generator
        mock_activity = {
            "id": 123,
            "name": "Morning Run",
            "distance": 10000,
            "moving_time": 3600,
            "total_elevation_gain": 100,
            "start_date_local": "2023-01-01T10:00:00Z",
            "average_heartrate": 150,
        }
        mock_strava.get_athlete_activities.return_value = iter([mock_activity])

        # Async test runner
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(get_recent_activities(limit=1))
        loop.close()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], 123)
        self.assertEqual(result[0]["distance_km"], 10.0)


if __name__ == "__main__":
    unittest.main()

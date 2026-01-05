"""
Tests for SurfaceClient - queries OSM for surface type information.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestSurfaceClient(unittest.TestCase):
    """Tests for SurfaceClient OSM surface detection."""

    @patch("ultra_trail_strategist.data_ingestion.surface_client.requests_cache.CachedSession")
    def test_surface_from_osm_surface_tag(self, mock_session_cls):
        """Test extracting surface from OSM 'surface' tag."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "elements": [{"tags": {"surface": "gravel", "highway": "track"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        from ultra_trail_strategist.data_ingestion.surface_client import SurfaceClient

        client = SurfaceClient()
        surface = client.get_surface_type(45.92, 6.86)

        self.assertEqual(surface, "gravel")

    @patch("ultra_trail_strategist.data_ingestion.surface_client.requests_cache.CachedSession")
    def test_surface_inferred_from_highway_asphalt(self, mock_session_cls):
        """Test inferring asphalt surface from highway type."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "elements": [{"tags": {"highway": "secondary"}}]  # No surface tag
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        from ultra_trail_strategist.data_ingestion.surface_client import SurfaceClient

        client = SurfaceClient()
        surface = client.get_surface_type(45.92, 6.86)

        self.assertEqual(surface, "asphalt")

    @patch("ultra_trail_strategist.data_ingestion.surface_client.requests_cache.CachedSession")
    def test_surface_inferred_from_highway_trail(self, mock_session_cls):
        """Test inferring trail surface from highway type."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "elements": [{"tags": {"highway": "path"}}]  # No surface tag
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        from ultra_trail_strategist.data_ingestion.surface_client import SurfaceClient

        client = SurfaceClient()
        surface = client.get_surface_type(45.92, 6.86)

        self.assertEqual(surface, "trail")

    @patch("ultra_trail_strategist.data_ingestion.surface_client.requests_cache.CachedSession")
    def test_surface_unknown_when_no_elements(self, mock_session_cls):
        """Test returning 'unknown' when no OSM elements found."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"elements": []}
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        from ultra_trail_strategist.data_ingestion.surface_client import SurfaceClient

        client = SurfaceClient()
        surface = client.get_surface_type(45.92, 6.86)

        self.assertEqual(surface, "unknown")

    @patch("ultra_trail_strategist.data_ingestion.surface_client.requests_cache.CachedSession")
    def test_surface_unknown_on_network_error(self, mock_session_cls):
        """Test returning 'unknown' on network error."""
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("Network error")
        mock_session_cls.return_value = mock_session

        from ultra_trail_strategist.data_ingestion.surface_client import SurfaceClient

        client = SurfaceClient()
        surface = client.get_surface_type(45.92, 6.86)

        self.assertEqual(surface, "unknown")

    @patch("ultra_trail_strategist.data_ingestion.surface_client.requests_cache.CachedSession")
    def test_overpass_query_construction(self, mock_session_cls):
        """Test that Overpass query is constructed correctly."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"elements": []}
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        from ultra_trail_strategist.data_ingestion.surface_client import SurfaceClient

        client = SurfaceClient()
        client.get_surface_type(45.123, 6.789)

        # Verify the query was called with correct parameters
        call_args = mock_session.get.call_args
        self.assertEqual(call_args[0][0], "http://overpass-api.de/api/interpreter")
        query_param = call_args[1]["params"]["data"]
        self.assertIn("45.123", query_param)
        self.assertIn("6.789", query_param)


if __name__ == "__main__":
    unittest.main()

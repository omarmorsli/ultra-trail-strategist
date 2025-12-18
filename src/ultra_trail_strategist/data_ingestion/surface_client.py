import logging

import requests_cache

logger = logging.getLogger(__name__)


class SurfaceClient:
    """
    Client for querying OpenStreetMap (OSM) via the Overpass API
    to determine surface types (e.g., asphalt, gravel, path).
    Uses persistent caching (30 days).
    """

    def __init__(self):
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        # Cache for 30 days (surface types rarely change)
        self.session = requests_cache.CachedSession(
            ".surface_cache", backend="sqlite", expire_after=2592000
        )

    def get_surface_type(self, lat: float, lon: float) -> str:
        """
        Determines the surface type for a given coordinate.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            str: "asphalt", "unpaved", "gravel", "dirt", "grass", "path", or "unknown".
        """
        # strict rounding to cache nearby hits if we were doing grids,
        # but for specific points we just check exact or very close.
        # For MVP, we query per point (slow) or mock.
        # Given 1km segments in a 50km race = 50 queries. Overpass allows this if spaced.
        # Rule of thumb: Don't hammer.

        # Build Overpass Query
        # Search for ways within 10 meters of the point
        query = f"""
        [out:json];
        way(around:10, {lat}, {lon});
        out tags;
        """

        try:
            response = self.session.get(self.overpass_url, params={"data": query}, timeout=5)
            response.raise_for_status()
            data = response.json()

            elements = data.get("elements", [])
            if not elements:
                return "unknown"

            # Check tags of the first found way
            # Priority: 'surface' tag -> 'highway' tag (inference)
            tags = elements[0].get("tags", {})
            surface = tags.get("surface")

            if surface:
                return surface

            highway = tags.get("highway")
            if highway in ["primary", "secondary", "tertiary", "residential"]:
                return "asphalt"
            elif highway in ["path", "track", "footway"]:
                # Default for track/path if no surface specified is often unpaved
                return "trail"

            return "unknown"

        except Exception as e:
            logger.warning(f"Failed to fetch surface for {lat},{lon}: {e}")
            return "unknown"

"""
Protocol interfaces for external clients.

These protocols define the contracts for external service integrations,
enabling easier testing, mocking, and implementation swapping.
"""

from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class WeatherProvider(Protocol):
    """Protocol for weather data providers."""

    def get_forecast(
        self, latitude: float, longitude: float, days: int = 1, date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch weather forecast for a location.

        Parameters
        ----------
        latitude : float
            Latitude of the location.
        longitude : float
            Longitude of the location.
        days : int
            Number of forecast days.
        date : str, optional
            Specific date (YYYY-MM-DD) for forecast.

        Returns
        -------
        Dict[str, Any]
            Weather forecast data.
        """
        ...

    def get_current_conditions(
        self, latitude: float, longitude: float, date: Optional[str] = None
    ) -> str:
        """
        Get simplified current weather conditions.

        Returns
        -------
        str
            Human-readable weather summary.
        """
        ...


@runtime_checkable
class ActivityProvider(Protocol):
    """Protocol for athlete activity data providers (e.g., Strava, Garmin)."""

    def get_athlete_activities(
        self,
        after: Optional[int] = None,
        before: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch athlete activities with pagination.

        Parameters
        ----------
        after : int, optional
            Unix timestamp to filter activities after.
        before : int, optional
            Unix timestamp to filter activities before.
        limit : int, optional
            Maximum number of activities to return.

        Yields
        ------
        Dict[str, Any]
            Individual activity records.
        """
        ...

    def get_activity_streams(
        self, activity_id: int, keys: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch detailed telemetry streams for an activity.

        Parameters
        ----------
        activity_id : int
            ID of the activity.
        keys : List[str], optional
            Stream keys to fetch (e.g., 'time', 'distance', 'altitude').

        Returns
        -------
        List[Dict[str, Any]]
            Stream data for the activity.
        """
        ...


@runtime_checkable
class SurfaceProvider(Protocol):
    """Protocol for surface/terrain type providers."""

    def get_surface_type(self, lat: float, lon: float) -> str:
        """
        Determine surface type at a coordinate.

        Parameters
        ----------
        lat : float
            Latitude.
        lon : float
            Longitude.

        Returns
        -------
        str
            Surface type (e.g., 'asphalt', 'trail', 'gravel', 'unknown').
        """
        ...


@runtime_checkable
class HealthProvider(Protocol):
    """Protocol for athlete health/readiness data providers."""

    def get_readiness_score(self, manual_override: Optional[int] = None) -> int:
        """
        Get athlete readiness score (0-100).

        Parameters
        ----------
        manual_override : int, optional
            Manual override value (takes priority if provided).

        Returns
        -------
        int
            Readiness score between 0 and 100.
        """
        ...

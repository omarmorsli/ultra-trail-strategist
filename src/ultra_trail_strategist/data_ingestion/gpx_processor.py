import logging
from typing import Any, Dict, List, Optional

import gpxpy
import numpy as np
import polars as pl
from scipy.signal import savgol_filter  # type: ignore

logger = logging.getLogger(__name__)


class GPXProcessor:
    """
    Ingests and processes GPX files for the Ultra-Trail Strategist.
    Handles elevation smoothing and initial data cleaning.
    """

    def __init__(self, file_path: Optional[str] = None):
        """
        Args:
            file_path: Absolute path to the GPX file. Optional if loading from string/stream later.
        """
        self.file_path = file_path
        self._raw_gpx: Optional[Any] = None
        self._df: Optional[pl.DataFrame] = None

    def load_from_file(self) -> None:
        """Loads and parses the GPX file from the file_path."""
        if not self.file_path:
            raise ValueError("file_path must be set to load from file")

        logger.info(f"Loading GPX file from {self.file_path}")
        with open(self.file_path, "r") as f:
            self._raw_gpx = gpxpy.parse(f)

    def load_from_string(self, gpx_content: str) -> None:
        """Loads and parses GPX data from a string."""
        self._raw_gpx = gpxpy.parse(gpx_content)

    def to_dataframe(self) -> pl.DataFrame:
        """
        Converts raw GPX points to a Polars DataFrame.
        Calculates cumulative distance if missing.
        """
        if not self._raw_gpx:
            raise ValueError("GPX data not loaded. Call load_* first.")

        data: List[Dict] = []

        for track in self._raw_gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    data.append(
                        {
                            "time": point.time,
                            "latitude": point.latitude,
                            "longitude": point.longitude,
                            "elevation": point.elevation,
                        }
                    )

        if not data:
            logger.warning("No points found in GPX.")
            return pl.DataFrame()

        df = pl.DataFrame(data)

        # Calculate Haversine distance between consecutive points
        # For simplicity and performance, we'll use a rough approximation or iterate.
        # Since accuracy is key for ultra-running, we should use geodesic, but vectorizing it efficiently is tricky slightly.
        # We will use a Haversine approximation within Polars expressions for speed,
        # or pre-calculate it. Given "Phase 1", let's use a numpy vectorization of Haversine.

        df = self._calculate_distance(df)
        self._df = df
        return df

    def _calculate_distance(self, df: pl.DataFrame) -> pl.DataFrame:
        """Adds a 'distance' and 'cumulative_distance' column using Haversine formula."""
        # Shift coords to get next point

        # We'll use simple Euclidean approximation for small steps or vectorised haversine.
        # Let's implement a vectorized Haversine here for Polars.

        # Convert to radians
        lat1 = np.radians(df["latitude"].to_numpy())
        lon1 = np.radians(df["longitude"].to_numpy())
        lat2 = np.radians(df["latitude"].shift(-1).to_numpy())
        lon2 = np.radians(df["longitude"].shift(-1).to_numpy())

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371000  # Earth radius in meters

        dist = c * r
        # Fill NaN (last point) with 0
        dist = np.nan_to_num(dist)

        # Calculate segment distance.
        # Note: 'dist' array contains dist from node i to node i+1.
        # So dist[0] is distance from point 0 to point 1.
        # We want the cumulative distance at point i to be the distance REACHED at point i.
        # Point 0 should have dist 0.
        # Point 1 should have dist[0].

        # Shift the segment distances to the right.
        dist = np.roll(dist, 1)
        dist[0] = 0.0

        # Add to DF
        df = df.with_columns(pl.Series("segment_dist", dist))

        df = df.with_columns(pl.col("segment_dist").cum_sum().alias("distance"))

        return df

    def smooth_elevation(self, window_length: int = 51, polyorder: int = 3) -> pl.DataFrame:
        """
        Applies Savitzky-Golay filter to smooth elevation data.
        """
        if self._df is None:
            raise ValueError("DataFrame not created. Call to_dataframe() first.")

        elevations = self._df["elevation"].to_numpy()

        # Window length must be odd and less than data size
        if len(elevations) < window_length:
            window_length = len(elevations) if len(elevations) % 2 != 0 else len(elevations) - 1
            if window_length < polyorder + 2:
                logger.warning("Not enough data points to smooth elevation.")
                return self._df

        smoothed = savgol_filter(elevations, window_length, polyorder)

        self._df = self._df.with_columns(pl.Series("elevation_smoothed", smoothed))
        return self._df

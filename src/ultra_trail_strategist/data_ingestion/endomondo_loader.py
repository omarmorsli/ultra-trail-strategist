"""
Endomondo/FitRec Dataset Loader.

Loads and processes the FitRec dataset (253k workouts) for training the base pace model.
Dataset source: https://sites.google.com/view/fitrec-project/

Citation:
    Jianmo Ni, Larry Muhlstein, Julian McAuley,
    "Modeling heart rate and activity data for personalized fitness recommendation",
    WWW'19, San Francisco, US, May. 2019.
"""

import ast
import gzip
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import polars as pl
import requests

logger = logging.getLogger(__name__)

# FitRec dataset URLs
FITREC_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/"
DATASET_FILES = {
    "raw": "endomondoHR.json.gz",
    "filtered": "endomondoHR_proper.json",
    "meta": "endomondoMeta.json.gz",
}


class EndomondoLoader:
    """
    Loader for FitRec/Endomondo workout dataset.

    Supports:
    - Downloading dataset files
    - Streaming workouts from gzipped JSON
    - Converting to training-ready Polars DataFrames
    - Caching processed data as Parquet

    Example
    -------
    >>> loader = EndomondoLoader()
    >>> loader.download_dataset()
    >>> df = loader.to_training_data(sport_filter=["run"])
    >>> print(df.shape)
    """

    def __init__(self, data_dir: Path = Path("data/endomondo")):
        """
        Initialize the Endomondo loader.

        Parameters
        ----------
        data_dir : Path
            Directory for storing downloaded and processed data.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(
        self, dataset_type: str = "raw", force: bool = False
    ) -> Path:
        """
        Download the FitRec dataset file if not present.

        Parameters
        ----------
        dataset_type : str
            Type of dataset: 'raw' (253k workouts), 'filtered' (167k cleaned), 'meta'.
        force : bool
            If True, re-download even if file exists.

        Returns
        -------
        Path
            Path to the downloaded file.
        """
        if dataset_type not in DATASET_FILES:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        filename = DATASET_FILES[dataset_type]
        filepath = self.data_dir / filename

        if filepath.exists() and not force:
            logger.info(f"Dataset already exists: {filepath}")
            return filepath

        url = f"{FITREC_BASE_URL}/{filename}"
        logger.info(f"Downloading {url}...")

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / total_size) * 100
                    if downloaded % (10 * 1024 * 1024) < 8192:  # Log every ~10MB
                        logger.info(f"Progress: {pct:.1f}%")

        logger.info(f"Downloaded: {filepath} ({downloaded / 1024 / 1024:.1f} MB)")
        return filepath

    def load_raw(
        self, max_workouts: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream workouts from gzipped JSON file.

        Parameters
        ----------
        max_workouts : Optional[int]
            Maximum number of workouts to yield (for testing/sampling).

        Yields
        ------
        Dict[str, Any]
            Individual workout record with keys:
            - userId, gender, sport, id
            - longitude, latitude, altitude (lists)
            - timestamp, heart_rate, derived_speed, distance (lists)
        """
        filepath = self.data_dir / DATASET_FILES["raw"]
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found: {filepath}. Run download_dataset() first."
            )

        count = 0
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                if max_workouts and count >= max_workouts:
                    break
                try:
                    # Dataset uses Python dict format (single quotes), not JSON
                    workout = ast.literal_eval(line.strip())
                    yield workout
                    count += 1
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Skipping malformed line: {e}")

        logger.info(f"Loaded {count} workouts")

    def load_filtered(
        self, max_workouts: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream workouts from the filtered/cleaned JSON file.

        The filtered version has:
        - Normalized measurements (Z-scores)
        - Derived variables (speed, distance)
        - Cleaned outliers

        Parameters
        ----------
        max_workouts : Optional[int]
            Maximum number of workouts to yield.

        Yields
        ------
        Dict[str, Any]
            Workout record with normalized values.
        """
        filepath = self.data_dir / DATASET_FILES["filtered"]
        if not filepath.exists():
            # Try to download if not present
            self.download_dataset("filtered")

        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if max_workouts and count >= max_workouts:
                    break
                try:
                    # Dataset uses Python dict format (single quotes), not JSON
                    workout = ast.literal_eval(line.strip())
                    yield workout
                    count += 1
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Skipping malformed line: {e}")

        logger.info(f"Loaded {count} filtered workouts")

    def _calculate_distance_from_gps(
        self, latitudes: List[float], longitudes: List[float]
    ) -> List[float]:
        """
        Calculate cumulative distance from GPS coordinates using haversine formula.

        Parameters
        ----------
        latitudes : List[float]
            Latitude values in degrees.
        longitudes : List[float]
            Longitude values in degrees.

        Returns
        -------
        List[float]
            Cumulative distance in meters.
        """
        if len(latitudes) < 2 or len(longitudes) < 2:
            return []

        R = 6371000  # Earth's radius in meters

        distances = [0.0]
        for i in range(1, min(len(latitudes), len(longitudes))):
            lat1 = np.radians(latitudes[i - 1])
            lat2 = np.radians(latitudes[i])
            lon1 = np.radians(longitudes[i - 1])
            lon2 = np.radians(longitudes[i])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            distance = R * c
            distances.append(distances[-1] + distance)

        return distances

    def _calculate_velocity_from_gps(
        self,
        latitudes: List[float],
        longitudes: List[float],
        timestamps: List[int],
    ) -> List[float]:
        """
        Calculate velocity (m/s) from GPS coordinates and timestamps.

        Parameters
        ----------
        latitudes : List[float]
            Latitude values in degrees.
        longitudes : List[float]
            Longitude values in degrees.
        timestamps : List[int]
            Unix timestamps in seconds.

        Returns
        -------
        List[float]
            Velocity in m/s for each point.
        """
        distances = self._calculate_distance_from_gps(latitudes, longitudes)
        if not distances or len(timestamps) < 2:
            return []

        velocities = [0.0]  # First point has no velocity
        for i in range(1, min(len(distances), len(timestamps))):
            dt = timestamps[i] - timestamps[i - 1]
            if dt > 0:
                dd = distances[i] - distances[i - 1]
                velocity = dd / dt  # m/s
                # Clip to reasonable running/cycling speeds (0-15 m/s = 0-54 km/h)
                velocity = min(max(velocity, 0.0), 15.0)
                velocities.append(velocity)
            else:
                velocities.append(velocities[-1] if velocities else 0.0)

        return velocities

    def _calculate_grade_from_workout(
        self, altitude: List[float], distance: List[float]
    ) -> List[float]:
        """
        Calculate grade percentage from altitude and distance arrays.

        Parameters
        ----------
        altitude : List[float]
            Elevation values in meters.
        distance : List[float]
            Cumulative distance values in meters.

        Returns
        -------
        List[float]
            Grade percentage for each point.
        """
        if len(altitude) < 2 or len(distance) < 2:
            return []

        alt_arr = np.array(altitude)
        dist_arr = np.array(distance)

        # Calculate deltas
        d_alt = np.diff(alt_arr)
        d_dist = np.diff(dist_arr)

        # Avoid division by zero
        d_dist = np.where(d_dist < 0.1, 0.1, d_dist)

        # Grade = (elevation change / distance) * 100
        grades = (d_alt / d_dist) * 100

        # Clip extreme grades
        grades = np.clip(grades, -50, 50)

        # Pad to match original length
        grades = np.concatenate([[grades[0]], grades])

        return list(grades.tolist())

    def to_training_data(
        self,
        sport_filter: Optional[List[str]] = None,
        max_workouts: Optional[int] = None,
        use_filtered: bool = False,
    ) -> pl.DataFrame:
        """
        Convert workouts to training-ready DataFrame.

        Parameters
        ----------
        sport_filter : Optional[List[str]]
            Filter by sport type (e.g., ["run", "bike"]). None = all sports.
        max_workouts : Optional[int]
            Maximum workouts to process.
        use_filtered : bool
            Use pre-filtered dataset (normalized values).

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - workout_id: Unique workout identifier
            - user_id: User identifier
            - sport: Sport type
            - grade: Grade percentage
            - velocity: Speed in m/s
            - heart_rate: Heart rate (if available)
            - altitude: Elevation in meters
        """
        cache_path = self.data_dir / "training_data.parquet"
        
        # Check cache
        if cache_path.exists() and not max_workouts:
            logger.info(f"Loading cached training data from {cache_path}")
            df = pl.read_parquet(cache_path)
            if sport_filter:
                df = df.filter(pl.col("sport").is_in(sport_filter))
            return df

        records = []
        workout_iter = (
            self.load_filtered(max_workouts)
            if use_filtered
            else self.load_raw(max_workouts)
        )

        for workout in workout_iter:
            sport = workout.get("sport", "unknown")
            if sport_filter and sport not in sport_filter:
                continue

            workout_id = workout.get("id")
            user_id = workout.get("userId")

            # Get arrays
            altitude = workout.get("altitude", [])
            distance = workout.get("distance", [])
            
            # If distance not available, calculate from GPS
            if not distance:
                latitudes = workout.get("latitude", [])
                longitudes = workout.get("longitude", [])
                if latitudes and longitudes:
                    distance = self._calculate_distance_from_gps(latitudes, longitudes)
            
            # Speed handling - raw vs filtered
            if use_filtered:
                # Filtered dataset has tar_derived_speed (original scale)
                velocity = workout.get("tar_derived_speed", [])
                heart_rate = workout.get("tar_heart_rate", [])
            else:
                # Raw dataset - use 'speed' field (km/h) and convert to m/s
                raw_speed = workout.get("speed", [])
                if raw_speed:
                    velocity = [s / 3.6 for s in raw_speed]  # km/h -> m/s
                else:
                    # Calculate velocity from GPS and timestamps
                    latitudes = workout.get("latitude", [])
                    longitudes = workout.get("longitude", [])
                    timestamps = workout.get("timestamp", [])
                    if latitudes and longitudes and timestamps:
                        velocity = self._calculate_velocity_from_gps(
                            latitudes, longitudes, timestamps
                        )
                    else:
                        velocity = []
                heart_rate = workout.get("heart_rate", [])

            if not altitude or not distance or not velocity:
                continue

            # Calculate grade
            grades = self._calculate_grade_from_workout(altitude, distance)
            if not grades:
                continue

            # Ensure arrays match
            min_len = min(len(grades), len(velocity), len(altitude))
            if heart_rate:
                min_len = min(min_len, len(heart_rate))

            for i in range(min_len):
                record = {
                    "workout_id": workout_id,
                    "user_id": user_id,
                    "sport": sport,
                    "grade": grades[i],
                    "velocity": velocity[i],
                    "altitude": altitude[i],
                }
                if heart_rate and i < len(heart_rate):
                    record["heart_rate"] = heart_rate[i]
                else:
                    record["heart_rate"] = None
                records.append(record)

        df = pl.DataFrame(records)

        # Clean data
        df = df.filter(
            (pl.col("velocity") > 0.1)  # Moving
            & (pl.col("velocity") < 15)  # < 54 km/h (reasonable running max)
            & (pl.col("grade").abs() < 50)  # Reasonable grades
        )

        # Cache if full dataset
        if not max_workouts and len(records) > 1000:
            logger.info(f"Caching training data to {cache_path}")
            df.write_parquet(cache_path)

        logger.info(f"Created training DataFrame: {df.shape}")
        return df

    def get_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the loaded dataset.

        Returns
        -------
        Dict[str, Any]
            Statistics including workout counts, user counts, sport breakdown.
        """
        sports: Dict[str, int] = {}
        total_workouts = 0
        unique_users: set[Any] = set()

        for workout in self.load_raw():
            total_workouts += 1
            unique_users.add(workout.get("userId"))
            sport = workout.get("sport", "unknown")
            sports[sport] = sports.get(sport, 0) + 1

        return {
            "sports": sports,
            "total_workouts": total_workouts,
            "unique_users": len(unique_users),
        }

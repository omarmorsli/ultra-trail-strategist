"""Tests for the Endomondo/FitRec dataset loader."""

import gzip
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ultra_trail_strategist.data_ingestion.endomondo_loader import EndomondoLoader


class TestEndomondoLoader:
    """Test suite for EndomondoLoader."""

    @pytest.fixture
    def loader(self, tmp_path: Path) -> EndomondoLoader:
        """Create loader with temporary directory."""
        return EndomondoLoader(data_dir=tmp_path)

    @pytest.fixture
    def sample_workout(self) -> dict:
        """Create a sample workout record."""
        return {
            "userId": 12345,
            "gender": "male",
            "sport": "run",
            "id": 1001,
            "longitude": [0.0, 0.001, 0.002],
            "latitude": [0.0, 0.0, 0.0],
            "altitude": [100.0, 110.0, 115.0],
            # Timestamps ~10 seconds apart
            "timestamp": [1000000, 1000010, 1000020],
            "heart_rate": [120, 130, 135],
        }

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that initialization creates the data directory."""
        data_dir = tmp_path / "new_dir"
        EndomondoLoader(data_dir=data_dir)
        assert data_dir.exists()

    def test_calculate_grade_from_workout(self, loader: EndomondoLoader) -> None:
        """Test grade calculation from altitude and distance."""
        altitude = [100.0, 110.0, 120.0, 115.0]  # 10m up, 10m up, 5m down
        distance = [0.0, 100.0, 200.0, 300.0]  # 100m segments

        grades = loader._calculate_grade_from_workout(altitude, distance)

        assert len(grades) == 4
        assert grades[1] == pytest.approx(10.0, rel=0.1)  # 10m / 100m = 10%
        assert grades[2] == pytest.approx(10.0, rel=0.1)  # 10m / 100m = 10%
        assert grades[3] == pytest.approx(-5.0, rel=0.1)  # -5m / 100m = -5%

    def test_calculate_grade_empty_arrays(self, loader: EndomondoLoader) -> None:
        """Test grade calculation handles empty arrays."""
        grades = loader._calculate_grade_from_workout([], [])
        assert grades == []

        grades = loader._calculate_grade_from_workout([100.0], [0.0])
        assert grades == []

    def test_load_raw_file_not_found(self, loader: EndomondoLoader) -> None:
        """Test that load_raw raises error when file missing."""
        with pytest.raises(FileNotFoundError):
            list(loader.load_raw())

    def test_load_raw_with_sample_data(
        self, loader: EndomondoLoader, sample_workout: dict, tmp_path: Path
    ) -> None:
        """Test loading raw data from gzipped file."""
        # Create sample gzipped file (use repr for Python dict format)
        gz_path = tmp_path / "endomondoHR.json.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            f.write(repr(sample_workout) + "\n")

        workouts = list(loader.load_raw())
        assert len(workouts) == 1
        assert workouts[0]["userId"] == 12345
        assert workouts[0]["sport"] == "run"

    def test_load_raw_max_workouts(
        self, loader: EndomondoLoader, sample_workout: dict, tmp_path: Path
    ) -> None:
        """Test max_workouts parameter limits results."""
        gz_path = tmp_path / "endomondoHR.json.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            for i in range(10):
                workout = sample_workout.copy()
                workout["id"] = i
                f.write(repr(workout) + "\n")

        workouts = list(loader.load_raw(max_workouts=3))
        assert len(workouts) == 3

    def test_to_training_data_basic(
        self, loader: EndomondoLoader, sample_workout: dict, tmp_path: Path
    ) -> None:
        """Test conversion to training DataFrame."""
        gz_path = tmp_path / "endomondoHR.json.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            f.write(repr(sample_workout) + "\n")

        df = loader.to_training_data(max_workouts=1)

        assert isinstance(df, pl.DataFrame)
        assert "grade" in df.columns
        assert "velocity" in df.columns
        assert "altitude" in df.columns

    def test_to_training_data_sport_filter(
        self, loader: EndomondoLoader, sample_workout: dict, tmp_path: Path
    ) -> None:
        """Test sport filtering in training data."""
        gz_path = tmp_path / "endomondoHR.json.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            # Add run workout
            f.write(repr(sample_workout) + "\n")
            # Add bike workout
            bike_workout = sample_workout.copy()
            bike_workout["sport"] = "bike"
            bike_workout["id"] = 2002
            f.write(repr(bike_workout) + "\n")

        # Filter for run only
        df = loader.to_training_data(sport_filter=["run"], max_workouts=10)
        assert df.filter(pl.col("sport") == "bike").height == 0

    @patch("requests.get")
    def test_download_dataset(
        self, mock_get: MagicMock, loader: EndomondoLoader
    ) -> None:
        """Test dataset download."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content = lambda chunk_size: [b"test data"]
        mock_get.return_value = mock_response

        path = loader.download_dataset("raw")

        assert path.exists()
        mock_get.assert_called_once()

    def test_download_dataset_exists(self, loader: EndomondoLoader) -> None:
        """Test that existing files are not re-downloaded."""
        # Create dummy file
        (loader.data_dir / "endomondoHR.json.gz").touch()

        with patch("requests.get") as mock_get:
            loader.download_dataset("raw", force=False)
            mock_get.assert_not_called()

import logging
from enum import Enum
from typing import List

import numpy as np
import polars as pl
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SegmentType(str, Enum):
    CLIMB = "Climb"
    DESCENT = "Descent"
    FLAT = "Flat"


from ultra_trail_strategist.data_ingestion.surface_client import SurfaceClient

# ...


class Segment(BaseModel):
    """
    Represents a tactical segment of the course.
    """

    start_dist: float
    end_dist: float
    type: SegmentType
    avg_grade: float
    elevation_gain: float
    elevation_loss: float
    length: float
    surface: str = "unknown"


class CourseSegmenter:
    """
    Analyzes course data to identify tactical segments.
    """

    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.climb_threshold = 3.0  # %
        self.descent_threshold = -3.0  # %
        self.min_segment_length = 50.0  # meters (to avoid noise)
        self.surface_client = SurfaceClient()

    def process(self) -> List[Segment]:
        """
        Main pipeline: calculate grade -> classify -> merge.
        Updates self.df with 'grade' and 'segment_type' columns.
        """
        if self.df.is_empty():
            return []

        self.df = self._calculate_grade(self.df)
        self.df = self._classify_points(self.df)
        segments = self._create_segments(self.df)
        return segments

    def _calculate_grade(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates grade (%) for each point based on smoothed elevation.
        Grade = (d_ele / d_dist) * 100
        """
        # Ensure we have smoothed elevation, fallback to raw
        ele_col = "elevation_smoothed" if "elevation_smoothed" in df.columns else "elevation"

        # Shift to calculate deltas
        # Using numpy for robust nan handling and small divisors
        ele = df[ele_col].to_numpy()
        dist = df["segment_dist"].to_numpy()  # segment_dist is delta dist

        # d_dist is already in 'segment_dist' column for the point i (dist from i to i+1 is usually stored,
        # BUT our GPX processor stored dist from i-1 to i in segment_dist due to shift logic?
        # Let's verify GPXProcessor logic:
        # "dist[i] is distance from point i-1 to point i" (after the roll and fix).
        # So we want gradient at point i representing the incoming segment?
        # Or outgoing? Usually grade is instantaneous.
        # Let's say grade at i is grade of segment ending at i.

        d_ele = df[ele_col].diff().fill_null(0.0).to_numpy()
        d_dist = df["segment_dist"].to_numpy()

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            grade = (d_ele / d_dist) * 100.0
            grade = np.nan_to_num(grade, nan=0.0, posinf=0.0, neginf=0.0)

        return df.with_columns(pl.Series("grade", grade))

    def _classify_points(self, df: pl.DataFrame) -> pl.DataFrame:
        """Assigns a SegmentType to each point."""

        def assign_type(g):
            if g > self.climb_threshold:
                return SegmentType.CLIMB.value
            elif g < self.descent_threshold:
                return SegmentType.DESCENT.value
            else:
                return SegmentType.FLAT.value

        # Use map_elements for custom python function (slower but flexible)
        # Or use polars expressions for speed

        # Polars expression way:
        type_expr = (
            pl.when(pl.col("grade") > self.climb_threshold)
            .then(pl.lit(SegmentType.CLIMB.value))
            .when(pl.col("grade") < self.descent_threshold)
            .then(pl.lit(SegmentType.DESCENT.value))
            .otherwise(pl.lit(SegmentType.FLAT.value))
        )

        return df.with_columns(type_expr.alias("segment_type"))

    def _create_segments(self, df: pl.DataFrame) -> List[Segment]:
        """Groups consecutive points of same type into Segments."""

        # Identification of groups: when type changes, new group starts.
        # Shift type to compare

        # Add a group ID
        # (val != val.shift).cum_sum()

        df = df.with_columns(
            (
                pl.col("segment_type")
                != pl.col("segment_type").shift(1).fill_null(pl.col("segment_type").first())
            )
            .cum_sum()
            .alias("group_id")
        )

        # Aggregate by group_id
        # We need start_dist (min of distance), end_dist (max of distance)
        # avg_grade (weighted by distance? or simple avg) -> simple avg of points is okay for now
        # ele gain/loss

        # Also need lat/lon to query surface
        # Assuming df has 'latitude' and 'longitude' (standard names from GPXProcessor?)
        # Let's check GPXProcessor. It produces: lat, lon, elevation, time...
        # Wait, the column names might be 'lat', 'lon' or 'latitude', 'longitude'.
        # Checking GPXProcessor code (not visible now but I should be careful).
        # Usually xml parsing yields 'lat', 'lon'.
        # I'll try to aggregate "lat" and "lon" first().

        columns = df.columns
        lat_col = "lat" if "lat" in columns else "latitude"
        lon_col = "lon" if "lon" in columns else "longitude"

        grouped = df.group_by(["group_id", "segment_type"], maintain_order=True).agg(
            [
                pl.col("distance").min().alias("start_dist"),
                pl.col("distance").max().alias("end_dist"),
                pl.col("grade").mean().alias("avg_grade"),
                pl.col("elevation").first().alias("start_ele"),
                pl.col("elevation").last().alias("end_ele"),
                pl.col(lat_col).first().alias("start_lat"),
                pl.col(lon_col).first().alias("start_lon"),
            ]
        )

        segments = []
        for row in grouped.iter_rows(named=True):
            length = row["end_dist"] - row["start_dist"]
            ele_diff = row["end_ele"] - row["start_ele"]
            gain = max(0.0, ele_diff)
            loss = max(0.0, -ele_diff)

            # Query Surface (only for segments > 100m to check?)
            # querying every segment -> safe rate limit?
            surface = "unknown"
            if length > 50:
                try:
                    surface = self.surface_client.get_surface_type(
                        row["start_lat"], row["start_lon"]
                    )
                except Exception as e:
                    logger.warning(f"Surface lookup failed: {e}")

            segments.append(
                Segment(
                    start_dist=row["start_dist"],
                    end_dist=row["end_dist"],
                    type=SegmentType(row["segment_type"]),
                    avg_grade=float(row["avg_grade"]),
                    elevation_gain=gain,
                    elevation_loss=loss,
                    length=length,
                    surface=surface,
                )
            )

        return segments

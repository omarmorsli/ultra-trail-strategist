import unittest
from unittest.mock import Mock

import polars as pl

from ultra_trail_strategist.feature_engineering.segmenter import CourseSegmenter, SegmentType


class TestCourseSegmenter(unittest.TestCase):
    def setUp(self):
        # Create a synthetic profile: Flat -> Climb -> Descent -> Flat
        # 4 points per section
        # Flat: dist 0-300m, ele 100
        # Climb: dist 300-600m, ele 100 -> 130 (+10% grade)
        # Descent: dist 600-900m, ele 130 -> 100 (-10% grade)
        # Flat: dist 900-1200m, ele 100

        data = []

        # Helper to gen points
        def add_points(start_dist, end_dist, start_ele, end_ele, count=10):
            step = (end_dist - start_dist) / count
            ele_step = (end_ele - start_ele) / count
            for i in range(count):
                d = start_dist + i * step
                e = start_ele + i * ele_step
                # We need column 'segment_dist' (delta) for our segmenter logic
                # For simplicity, let's assume constant step
                data.append(
                    {
                        "distance": d,
                        "elevation": e,
                        "elevation_smoothed": e,
                        "segment_dist": step,
                        "latitude": 45.0 + (i * 0.0001),  # Mock coords
                        "longitude": 6.0 + (i * 0.0001),
                    }
                )

        add_points(0, 300, 100, 100)  # Flat
        add_points(300, 600, 100, 130)  # Climb 10% (30m / 300m)
        add_points(600, 900, 130, 100)  # Descent -10%
        add_points(900, 1200, 100, 100)  # Flat

        self.df = pl.DataFrame(data)
        self.segmenter = CourseSegmenter(self.df)

        # Mock SurfaceClient to prevent network calls
        self.segmenter.surface_client = Mock()
        self.segmenter.surface_client.get_surface_type.return_value = "trail"

    def test_calculate_grade(self):
        df = self.segmenter._calculate_grade(self.df)
        self.assertIn("grade", df.columns)

        # Check simplified grade values
        # Middle of climb section (approx index 15)
        # We constructed it to be ~10%
        climb_grades = df.filter((pl.col("distance") > 350) & (pl.col("distance") < 550))["grade"]
        self.assertTrue(all(g > 5 for g in climb_grades.to_list()))

    def test_segment_creation(self):
        segments = self.segmenter.process()

        # We expect roughly 4 segments
        # Note: transitions might create tiny segments depending on boundary points
        # But broadly: Flat, Climb, Descent, Flat

        types = [s.type for s in segments if s.length > 50]  # Filter noise

        self.assertEqual(types[0], SegmentType.FLAT)
        self.assertEqual(types[1], SegmentType.CLIMB)
        self.assertEqual(types[2], SegmentType.DESCENT)
        self.assertEqual(types[3], SegmentType.FLAT)


if __name__ == "__main__":
    unittest.main()

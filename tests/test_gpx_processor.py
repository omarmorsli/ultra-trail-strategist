import unittest

import polars as pl

from ultra_trail_strategist.data_ingestion.gpx_processor import GPXProcessor


class TestGPXProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = GPXProcessor()
        # Create a simple synthetic GPX content
        self.mock_gpx = (
            '<gpx version="1.1" creator="UltraTrailTest">\n'
            "    <trk>\n"
            "        <name>Test Track</name>\n"
            "        <trkseg>\n"
            '            <trkpt lat="45.0" lon="6.0"><ele>100.0</ele>'
            '                <time>2023-01-01T10:00:00Z</time></trkpt>\n'
            '            <trkpt lat="45.001" lon="6.0"><ele>105.0</ele>'
            '                <time>2023-01-01T10:01:00Z</time></trkpt>\n'
            '            <trkpt lat="45.002" lon="6.0"><ele>102.0</ele>'
            '                <time>2023-01-01T10:02:00Z</time></trkpt>\n'
            '            <trkpt lat="45.003" lon="6.0"><ele>110.0</ele>'
            '                <time>2023-01-01T10:03:00Z</time></trkpt>\n'
            "        </trkseg>\n"
            "    </trk>\n"
            "</gpx>"
        )

    def test_load_and_parse(self):
        self.processor.load_from_string(self.mock_gpx)
        self.assertIsNotNone(self.processor._raw_gpx)
        if self.processor._raw_gpx:
            self.assertEqual(len(self.processor._raw_gpx.tracks), 1)

    def test_to_dataframe_structure(self):
        self.processor.load_from_string(self.mock_gpx)
        df = self.processor.to_dataframe()

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 4)
        expected_cols = ["time", "latitude", "longitude", "elevation", "segment_dist", "distance"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_distance_calculation(self):
        self.processor.load_from_string(self.mock_gpx)
        df = self.processor.to_dataframe()

        # Distance should be increasing
        distances = df["distance"].to_list()
        self.assertEqual(distances[0], 0.0)
        self.assertTrue(distances[-1] > 0)

        # Roughly 111m per 0.001 degree lat
        self.assertTrue(100 < distances[1] < 120)

    def test_elevation_smoothing(self):
        self.processor.load_from_string(self.mock_gpx)
        self.processor.to_dataframe()
        # Use small window for small dataset
        df = self.processor.smooth_elevation(window_length=3, polyorder=1)

        self.assertIn("elevation_smoothed", df.columns)
        # Check that it exists and is same length
        self.assertEqual(len(df["elevation_smoothed"]), 4)


if __name__ == "__main__":
    unittest.main()

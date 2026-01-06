"""
Tests for cache management utilities.
"""

import os
import tempfile
import unittest
from pathlib import Path

from ultra_trail_strategist.utils.cache_manager import CacheManager


class TestCacheManager(unittest.TestCase):
    """Tests for CacheManager."""

    def setUp(self):
        """Create a temporary directory with mock cache files."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = CacheManager(base_dir=self.temp_dir)

        # Create dummy cache files
        for cache_name in [".strava_cache.sqlite", ".weather_cache.sqlite"]:
            cache_path = Path(self.temp_dir) / cache_name
            cache_path.write_bytes(b"x" * 1024)  # 1KB each

    def tearDown(self):
        """Clean up temporary files."""
        for f in Path(self.temp_dir).iterdir():
            f.unlink()
        os.rmdir(self.temp_dir)

    def test_list_caches(self):
        """Test listing cache files."""
        caches = self.manager.list_caches()
        self.assertEqual(len(caches), 3)

        strava_cache = next(c for c in caches if "strava" in c.name)
        self.assertTrue(strava_cache.exists)
        self.assertEqual(strava_cache.size_bytes, 1024)

    def test_get_total_cache_size(self):
        """Test total cache size calculation."""
        total = self.manager.get_total_cache_size()
        # 2 files * 1KB each
        self.assertEqual(total, 2048)

    def test_clear_cache(self):
        """Test clearing a specific cache."""
        result = self.manager.clear_cache(".strava_cache.sqlite")
        self.assertTrue(result)

        # Verify it's gone
        caches = self.manager.list_caches()
        strava_cache = next(c for c in caches if "strava" in c.name)
        self.assertFalse(strava_cache.exists)

    def test_clear_nonexistent_cache(self):
        """Test clearing a cache that doesn't exist."""
        result = self.manager.clear_cache(".surface_cache.sqlite")
        self.assertFalse(result)

    def test_clear_all_caches(self):
        """Test clearing all caches."""
        cleared = self.manager.clear_all_caches()
        self.assertEqual(cleared, 2)

        # Verify all are gone
        total = self.manager.get_total_cache_size()
        self.assertEqual(total, 0)

    def test_get_cache_summary(self):
        """Test cache summary string."""
        summary = self.manager.get_cache_summary()
        self.assertIn("Cache Status:", summary)
        self.assertIn("strava", summary)
        self.assertIn("1.0 KB", summary)


if __name__ == "__main__":
    unittest.main()

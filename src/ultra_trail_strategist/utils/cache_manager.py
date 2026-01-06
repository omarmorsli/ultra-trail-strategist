"""
Cache management utilities for Ultra-Trail Strategist.

Provides unified cache operations across all SQLite cache files.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheInfo:
    """Information about a cache file."""

    name: str
    path: str
    size_bytes: int
    exists: bool


class CacheManager:
    """
    Unified cache management for all application caches.

    Manages:
    - .strava_cache.sqlite
    - .weather_cache.sqlite
    - .surface_cache.sqlite
    """

    DEFAULT_CACHE_FILES = [
        ".strava_cache.sqlite",
        ".weather_cache.sqlite",
        ".surface_cache.sqlite",
    ]

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the cache manager.

        Parameters
        ----------
        base_dir : str, optional
            Base directory containing cache files. Defaults to current directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def list_caches(self) -> List[CacheInfo]:
        """
        List all cache files and their sizes.

        Returns
        -------
        List[CacheInfo]
            Information about each cache file.
        """
        caches = []
        for cache_name in self.DEFAULT_CACHE_FILES:
            cache_path = self.base_dir / cache_name
            exists = cache_path.exists()
            size = cache_path.stat().st_size if exists else 0
            caches.append(
                CacheInfo(
                    name=cache_name,
                    path=str(cache_path),
                    size_bytes=size,
                    exists=exists,
                )
            )
        return caches

    def get_total_cache_size(self) -> int:
        """
        Get total size of all caches in bytes.

        Returns
        -------
        int
            Total cache size in bytes.
        """
        return sum(c.size_bytes for c in self.list_caches())

    def clear_cache(self, cache_name: str) -> bool:
        """
        Clear a specific cache file.

        Parameters
        ----------
        cache_name : str
            Name of the cache file to clear.

        Returns
        -------
        bool
            True if cache was cleared, False if it didn't exist.
        """
        cache_path = self.base_dir / cache_name
        if cache_path.exists():
            os.remove(cache_path)
            logger.info(f"Cleared cache: {cache_name}")
            return True
        return False

    def clear_all_caches(self) -> int:
        """
        Clear all cache files.

        Returns
        -------
        int
            Number of caches cleared.
        """
        cleared = 0
        for cache_name in self.DEFAULT_CACHE_FILES:
            if self.clear_cache(cache_name):
                cleared += 1
        logger.info(f"Cleared {cleared} cache files")
        return cleared

    def get_cache_summary(self) -> str:
        """
        Get a human-readable summary of cache status.

        Returns
        -------
        str
            Summary string with cache sizes.
        """
        caches = self.list_caches()
        lines = ["Cache Status:"]
        for cache in caches:
            status = f"{cache.size_bytes / 1024:.1f} KB" if cache.exists else "Not found"
            lines.append(f"  {cache.name}: {status}")
        total = self.get_total_cache_size()
        lines.append(f"  Total: {total / 1024:.1f} KB")
        return "\n".join(lines)

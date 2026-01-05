"""
Centralized logging configuration for Ultra-Trail Strategist.

Import this module early in application entry points (main.py, dashboard.py)
to configure logging before other modules are imported.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger for the application.

    Parameters
    ----------
    level : int
        Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# Configure on import for convenience
setup_logging()

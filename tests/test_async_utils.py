"""
Tests for async utilities.
"""

import asyncio
import unittest

from ultra_trail_strategist.utils.async_utils import run_in_thread, to_thread


def blocking_operation(x: int, y: int = 0) -> int:
    """A simple blocking function for testing."""
    return x + y


class TestAsyncUtils(unittest.TestCase):
    """Tests for async utilities."""

    def test_to_thread(self):
        """Test to_thread runs blocking function."""

        async def run_test():
            result = await to_thread(blocking_operation, 5, y=3)
            return result

        result = asyncio.run(run_test())
        self.assertEqual(result, 8)

    def test_run_in_thread_decorator(self):
        """Test run_in_thread decorator."""

        @run_in_thread
        def decorated_blocking(x: int) -> int:
            return x * 2

        async def run_test():
            result = await decorated_blocking(10) # type: ignore[misc]
            return result

        result = asyncio.run(run_test())
        self.assertEqual(result, 20)

    def test_to_thread_with_only_args(self):
        """Test to_thread with positional args only."""

        async def run_test():
            result = await to_thread(blocking_operation, 3, 7)
            return result

        result = asyncio.run(run_test())
        self.assertEqual(result, 10)


if __name__ == "__main__":
    unittest.main()

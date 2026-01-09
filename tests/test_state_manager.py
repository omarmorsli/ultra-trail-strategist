"""
Tests for RaceStateManager - manages live race state persistence.
"""

import json
import os
import tempfile
import unittest

from ultra_trail_strategist.state_manager import RaceState, RaceStateManager


class TestRaceState(unittest.TestCase):
    """Tests for the RaceState Pydantic model."""

    def test_default_state(self):
        """Test RaceState default values."""
        state = RaceState()
        self.assertIsNone(state.start_time)
        self.assertEqual(state.actual_splits, {})
        self.assertEqual(state.current_segment_index, 0)

    def test_state_with_data(self):
        """Test RaceState with provided values."""
        state = RaceState(
            start_time="2025-01-05T08:00:00",
            actual_splits={0: 30.0, 1: 65.5},
            current_segment_index=2,
        )
        self.assertEqual(state.start_time, "2025-01-05T08:00:00")
        self.assertEqual(state.actual_splits[0], 30.0)
        self.assertEqual(state.current_segment_index, 2)

    def test_state_ignores_extra_fields(self):
        """Test that extra fields are ignored (ConfigDict extra='ignore')."""
        state = RaceState(unknown_field="ignored", current_segment_index=5) # type: ignore[call-arg]
        self.assertEqual(state.current_segment_index, 5)
        self.assertFalse(hasattr(state, "unknown_field"))


class TestRaceStateManager(unittest.TestCase):
    """Tests for RaceStateManager persistence logic."""

    def setUp(self):
        """Create a temporary file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.temp_dir, "test_race_state.json")
        self.manager = RaceStateManager(state_file=self.state_file)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        os.rmdir(self.temp_dir)

    def test_get_state_empty_file(self):
        """Test get_state returns default when no file exists."""
        state = self.manager.get_state()
        self.assertIsNone(state.start_time)
        self.assertEqual(state.actual_splits, {})
        self.assertEqual(state.current_segment_index, 0)

    def test_save_and_get_state(self):
        """Test saving and retrieving state."""
        state = RaceState(
            start_time="2025-01-05T06:00:00",
            actual_splits={0: 25.0},
            current_segment_index=1,
        )
        self.manager.save_state(state)

        # Verify file was created
        self.assertTrue(os.path.exists(self.state_file))

        # Retrieve and verify
        loaded_state = self.manager.get_state()
        self.assertEqual(loaded_state.start_time, "2025-01-05T06:00:00")
        self.assertEqual(loaded_state.actual_splits, {0: 25.0})
        self.assertEqual(loaded_state.current_segment_index, 1)

    def test_update_checkpoint(self):
        """Test updating a checkpoint."""
        self.manager.update_checkpoint(0, 30.0)

        state = self.manager.get_state()
        self.assertEqual(state.actual_splits[0], 30.0)
        self.assertEqual(state.current_segment_index, 1)

    def test_update_multiple_checkpoints(self):
        """Test updating multiple checkpoints sequentially."""
        self.manager.update_checkpoint(0, 28.5)
        self.manager.update_checkpoint(1, 55.0)
        self.manager.update_checkpoint(2, 90.0)

        state = self.manager.get_state()
        self.assertEqual(len(state.actual_splits), 3)
        self.assertEqual(state.actual_splits[2], 90.0)
        self.assertEqual(state.current_segment_index, 3)

    def test_set_start_time(self):
        """Test setting race start time."""
        self.manager.set_start_time("2025-01-05T07:00:00")

        state = self.manager.get_state()
        self.assertEqual(state.start_time, "2025-01-05T07:00:00")

    def test_reset_race(self):
        """Test resetting race state."""
        # First, create some state
        self.manager.update_checkpoint(0, 30.0)
        self.assertTrue(os.path.exists(self.state_file))

        # Reset
        self.manager.reset_race()

        # Verify file is gone
        self.assertFalse(os.path.exists(self.state_file))

        # Verify get_state returns defaults
        state = self.manager.get_state()
        self.assertEqual(state.actual_splits, {})

    def test_corrupted_json_file(self):
        """Test handling of corrupted JSON file."""
        # Write invalid JSON
        with open(self.state_file, "w") as f:
            f.write("{ invalid json }")

        # Should return default state, not crash
        state = self.manager.get_state()
        self.assertEqual(state.current_segment_index, 0)

    def test_json_key_type_conversion(self):
        """Test that JSON string keys are converted to integers."""
        # JSON stores dict keys as strings
        with open(self.state_file, "w") as f:
            json.dump({"actual_splits": {"0": 30.0, "1": 60.0}, "current_segment_index": 2}, f)

        # Update checkpoint should handle string-to-int conversion
        self.manager.update_checkpoint(2, 95.0)

        state = self.manager.get_state()
        # All keys should be integers
        self.assertIn(0, state.actual_splits)
        self.assertIn(1, state.actual_splits)
        self.assertIn(2, state.actual_splits)


if __name__ == "__main__":
    unittest.main()

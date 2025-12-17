import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, ConfigDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATE_FILE = "live_race_state.json"

class RaceState(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    start_time: Optional[str] = None
    actual_splits: Dict[int, float] = {}  # segment_index -> minutes from start
    current_segment_index: int = 0
    
    # We might want to store the original plan too, but for now let's keep it simple.
    
class RaceStateManager:
    """
    Manages persistence of the live race state to a local JSON file.
    This ensures data is not lost if the dashboard/server restarts.
    """
    
    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file

    def _load_raw_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_file):
            return {}
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load race state: {e}")
            return {}

    def get_state(self) -> RaceState:
        data = self._load_raw_state()
        return RaceState(**data)

    def save_state(self, state: RaceState):
        try:
            # Pydantic v2 dump
            data = state.model_dump()
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save race state: {e}")

    def update_checkpoint(self, segment_index: int, arrival_time_minutes: float):
        """
        Updates the state with a new checkpoint arrival time.
        """
        state = self.get_state()
        # Ensure dictionary keys are integers (JSON loads them as strings sometimes)
        state.actual_splits = {int(k): v for k, v in state.actual_splits.items()}
        
        state.actual_splits[segment_index] = arrival_time_minutes
        
        # Assume valid sequential progression? Not necessarily, but let's update current index
        # to be the max index checked in so far + 1
        max_checked_in = max(state.actual_splits.keys()) if state.actual_splits else -1
        state.current_segment_index = max_checked_in + 1
        
        self.save_state(state)
        logger.info(f"Checkpoint updated: Segment {segment_index} at {arrival_time_minutes} min")

    def set_start_time(self, start_time_iso: str):
        state = self.get_state()
        state.start_time = start_time_iso
        self.save_state(state)

    def reset_race(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
            logger.info("Race state reset.")

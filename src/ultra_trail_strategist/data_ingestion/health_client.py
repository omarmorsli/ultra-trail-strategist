import logging
import os
from datetime import date
from typing import Optional

import garminconnect  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class HealthClient:
    """
    Client for fetching athlete health and readiness metrics.
    Integrates with Garmin Connect if credentials are provided.
    Falls back to mock/manual values.
    """

    def __init__(self):
        self.garmin_client = None
        self.email = os.getenv("GARMIN_EMAIL")
        self.password = os.getenv("GARMIN_PASSWORD")
        self._init_garmin()

    def _init_garmin(self):
        """Attempts to initialize and log in to Garmin Connect."""
        if self.email and self.password:
            try:
                self.garmin_client = garminconnect.Garmin(self.email, self.password)
                self.garmin_client.login()
                logger.info("✅ Successfully logged into Garmin Connect.")
            except Exception as e:
                logger.error(f"❌ Failed to login to Garmin Connect: {e}")
                self.garmin_client = None
        else:
            logger.info("ℹ️ No Garmin credentials found (GARMIN_EMAIL/PASSWORD). Using manual mode.")

    def get_readiness_score(self, manual_override: Optional[int] = None) -> int:
        """
        Returns the athlete's daily readiness score (0-100).
        Priority:
        1. Manual Override (if provided via UI)
        2. Garmin Training Readiness
        3. Garmin Body Battery
        4. Default (80)

        Args:
            manual_override: If provided (e.g. from UI), use this high priority.

        Returns:
            int: 0-100 score.
        """
        # 1. Manual Override
        if manual_override is not None:
            return max(0, min(100, manual_override))

        # 2. Garmin Integration
        if self.garmin_client:
            try:
                today = date.today().isoformat()

                # --- A. Training Readiness ---
                try:
                    # Generic method based on library common patterns
                    if hasattr(self.garmin_client, "get_training_readiness"):
                        tr_data = self.garmin_client.get_training_readiness(today)
                        # Structure typically: {'score': 85, ...} or {'trainingReadiness': 85}
                        if tr_data:
                            if isinstance(tr_data, dict):
                                if "score" in tr_data:
                                    val = int(tr_data["score"])
                                    logger.info(f"✅ Garmin: Found Training Readiness Score: {val}")
                                    return val
                                if "trainingReadiness" in tr_data:
                                    val = int(tr_data["trainingReadiness"])
                                    logger.info(f"✅ Garmin: Found Training Readiness: {val}")
                                    return val
                            # If it's a list (rare for single day readiness but possible)
                            if isinstance(tr_data, list) and len(tr_data) > 0:
                                if "score" in tr_data[-1]:
                                    val = int(tr_data[-1]["score"])
                                    logger.info(
                                        f"✅ Garmin: Found Training Readiness (List): {val}"
                                    )
                                    return val
                except Exception as e_tr:
                    logger.warning(f"Garmin Readiness fetch failed: {e_tr}")

                # --- B. Body Battery (Fallback) ---
                try:
                    # Method: get_body_battery(date)
                    if hasattr(self.garmin_client, "get_body_battery"):
                        bb_data = self.garmin_client.get_body_battery(today)
                        # Typically returns a list of dictionaries for 15min intervals
                        # [{'date':..., 'bodyBatteryValues': {'charged': 80, ...}}, ...]
                        # Or a summary dict.

                        if isinstance(bb_data, list) and len(bb_data) > 0:
                            # Get the latest data point
                            last_data = bb_data[-1]
                            if "bodyBatteryValues" in last_data:  # Detailed structure
                                vals = last_data["bodyBatteryValues"]
                                # 'charged' or 'value' usually present
                                for key in ["charged", "value", "mostRecentBodyBattery"]:
                                    if key in vals and vals[key] is not None:
                                        val = int(vals[key])
                                        logger.info(f"✅ Garmin: Found Body Battery ({key}): {val}")
                                        return val

                            # Flattened structure check
                            if "bodyBattery" in last_data:
                                val = int(last_data["bodyBattery"])
                                logger.info(f"✅ Garmin: Found Body Battery (flat): {val}")
                                return val

                    # --- C. User Summary (Fallback 2) ---
                    # get_user_summary(date) is very standard
                    summary = self.garmin_client.get_user_summary(today)
                    if summary:
                        # Check for various keys known to appear in summaries
                        for key in [
                            "bodyBattery",
                            "averageBodyBattery",
                            "maxBodyBattery",
                            "endOfDayBodyBattery",
                        ]:
                            if key in summary and summary[key] is not None:
                                val = int(summary[key])
                                logger.info(f"✅ Garmin: Found User Summary ({key}): {val}")
                                return val

                except Exception as e_bb:
                    logger.warning(f"Garmin Body Battery fetch failed: {e_bb}")

            except Exception as e:
                logger.error(f"Failed to fetch Garmin data: {e}")

        # 4. Default
        logger.info("ℹ️ Garmin: No data found, returning default 50.")
        return 50

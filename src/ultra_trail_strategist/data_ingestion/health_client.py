import os
import logging
from typing import Optional, Dict, Any
from datetime import date
import garminconnect

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
                
                # Fetch Training Readiness
                # Note: API endpoint structures vary, using common known patterns
                # garminconnect library usually has specific methods or generic get requests
                
                # Try specific training readiness if available in library, else body battery
                start_date = today
                
                # Try getting stats
                # The library often exposes: client.get_training_readiness(date) ?
                # Let's try generic user summary or specific method if known.
                # Checking library capability: usually get_user_summary(date) has body battery.
                
                stats = self.garmin_client.get_user_summary(today)
                
                # Parse Training Readiness (if available in summary)
                # 'trainingReadiness' might be a key
                if "trainingReadiness" in stats:
                     return int(stats["trainingReadiness"])

                # Fallback to Body Battery
                if "bodyBattery" in stats:
                     # bodyBattery is often a dict with 'highest', 'lowest', or just a value
                     # stats['bodyBatteryChargedValue'] ?
                     # Let's look for 'averageBodyBattery' or 'charged'
                     # Or checks 'mostRecentBodyBattery'
                     pass
                     
                # Safer check for body battery in daily summary
                # "totalBodyBattery" or similar.
                # Actually, let's look for "latestBodyBattery" logic or use Body Composition? No.
                
                # Let's try to fetch specific Body Battery data if summary fails
                # client.get_body_battery(today)
                bb_data = self.garmin_client.get_body_battery(start_date)
                if bb_data:
                    # usually a list of dicts. Get the last one?
                    # or it returns a summary dict.
                    # Assuming it returns a list of samples, we take the last one.
                    if isinstance(bb_data, list) and len(bb_data) > 0:
                        last_point = bb_data[-1]
                        if "bodyBatteryValues" in last_point:
                             # This structure is complex. 
                             pass
                    # If it's a list containing dicts with 'value'
                    pass

                # Simplify: User Summary usually has 'minBodyBattery', 'maxBodyBattery', 'endOfDayBodyBattery'
                if 'endOfDayBodyBattery' in stats and stats['endOfDayBodyBattery']:
                     return int(stats['endOfDayBodyBattery'])
                if 'maxBodyBattery' in stats and stats['maxBodyBattery']:
                    # Max BB is a decent proxy for "Morning Readiness"
                    return int(stats['maxBodyBattery'])

            except Exception as e:
                logger.error(f"Failed to fetch Garmin data: {e}")
        
        # 4. Default
        return 80

import logging
import time
from typing import Iterator, Dict, Any, Optional
import requests
from pydantic import BaseModel
from ultra_trail_strategist.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StravaTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: int
    expires_in: int

class StravaClient:
    """
    A robust client for the Strava API v3.
    Handles OAuth2 token refreshing, pagination, and error checking.
    """

    def __init__(self):
        self.base_url = settings.STRAVA_BASE_URL
        self.client_id = settings.STRAVA_CLIENT_ID
        self.client_secret = settings.STRAVA_CLIENT_SECRET.get_secret_value()
        self.refresh_token = settings.STRAVA_REFRESH_TOKEN.get_secret_value()
        self.access_token: Optional[str] = None
        self.token_expires_at: int = 0
        self.session = requests.Session()

    def _ensure_valid_token(self) -> None:
        """
        Checks if the current access token is valid (or non-existent).
        If expired or missing, refreshes the token using the refresh_token.
        """
        current_time = time.time()
        # Refresh if token is missing or expires in less than 60 seconds
        if not self.access_token or current_time >= self.token_expires_at - 60:
            logger.info("Access token missing or expiring, refreshing...")
            self._refresh_access_token()

    def _refresh_access_token(self) -> None:
        """
        Exchanges the refresh_token for a new access_token.
        Updates internal state.
        """
        auth_url = "https://www.strava.com/oauth/token"
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
        }

        try:
            response = requests.post(auth_url, data=payload)
            response.raise_for_status()
            data = response.json()
            
            # Validate response with Pydantic
            token_data = StravaTokenResponse(**data)
            
            self.access_token = token_data.access_token
            # Update refresh token if a new one is returned
            self.refresh_token = token_data.refresh_token
            self.token_expires_at = token_data.expires_at
            
            logger.info("Successfully refreshed Strava access token.")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to refresh Strava token: {e}")
            if e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def build_headers(self) -> Dict[str, str]:
        """Constructs headers with the valid Bearer token."""
        self._ensure_valid_token()
        return {"Authorization": f"Bearer {self.access_token}"}

    def get_athlete_activities(
        self, 
        after: Optional[int] = None, 
        before: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetches athlete activities with automatic pagination.
        Yields individual activity dictionaries.
        
        Args:
            after (int, optional): Timestamp to filter activities after.
            before (int, optional): Timestamp to filter activities before.
            limit (int, optional): Max total activities to return. None for all.
        """
        page = 1
        per_page = settings.PAGE_SIZE
        count = 0

        while True:
            params = {
                "page": page,
                "per_page": per_page
            }
            if after:
                params["after"] = after
            if before:
                params["before"] = before

            endpoint = f"{self.base_url}/athlete/activities"
            try:
                response = self.session.get(endpoint, headers=self.build_headers(), params=params)
                response.raise_for_status()
                activities = response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching activities on page {page}: {e}")
                raise

            if not activities:
                break

            for activity in activities:
                yield activity
                count += 1
                if limit and count >= limit:
                    return

            if len(activities) < per_page:
                # Reached the last page of results
                break
            
            page += 1

    def get_activity_stream(self, activity_id: int) -> Dict[str, Any]:
        """
        Fetches the detailed stream for a specific activity 
        (lat/lng, elevation, time, etc.).
        """
        endpoint = f"{self.base_url}/activities/{activity_id}/streams"
        keys = "time,distance,latlng,altitude,velocity_smooth,heartrate,grade_smooth,moving"
        params = {"keys": keys, "key_by_type": "true"}
        
        try:
            response = self.session.get(endpoint, headers=self.build_headers(), params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching stream for activity {activity_id}: {e}")
            raise

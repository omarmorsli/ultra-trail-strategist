import os

# Inject dummy credentials BEFORE any application imports happen
# This prevents Pydantic ValidationError during collection
# when tests run in CI environments without real .env files.
os.environ.setdefault("STRAVA_CLIENT_ID", "dummy_id")
os.environ.setdefault("STRAVA_CLIENT_SECRET", "dummy_secret_123")
os.environ.setdefault("STRAVA_REFRESH_TOKEN", "dummy_refresh_token")

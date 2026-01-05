from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings managed by Pydantic Settings.
    Reads from environment variables and .env file.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Strava API Configuration (Required)
    STRAVA_CLIENT_ID: str
    STRAVA_CLIENT_SECRET: SecretStr
    STRAVA_REFRESH_TOKEN: SecretStr
    STRAVA_BASE_URL: str = "https://www.strava.com/api/v3"

    # OpenAI Configuration (Optional - uses dummy key for testing if not provided)
    OPENAI_API_KEY: Optional[SecretStr] = None

    # Telegram Bot (Optional)
    TELEGRAM_BOT_TOKEN: Optional[str] = None

    # Garmin Connect (Optional)
    GARMIN_EMAIL: Optional[str] = None
    GARMIN_PASSWORD: Optional[SecretStr] = None

    # Application Defaults
    PAGE_SIZE: int = 100


settings = Settings()  # type: ignore[call-arg]


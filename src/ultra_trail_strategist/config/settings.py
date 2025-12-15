from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    """
    Application settings managed by Pydantic Settings.
    Reads from environment variables and .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Strava API Configuration
    STRAVA_CLIENT_ID: str
    STRAVA_CLIENT_SECRET: SecretStr
    STRAVA_REFRESH_TOKEN: SecretStr
    STRAVA_BASE_URL: str = "https://www.strava.com/api/v3"

    # Application Defaults
    PAGE_SIZE: int = 100

settings = Settings()

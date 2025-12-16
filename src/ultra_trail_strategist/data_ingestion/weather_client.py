import logging
from typing import Dict, Any, Optional
import requests
import requests_cache

logger = logging.getLogger(__name__)

class WeatherClient:
    """
    Client for OpenMeteo API to fetch weather forecasts.
    Uses persistent caching.
    """
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self):
        # Cache for 2 hours
        self.session = requests_cache.CachedSession(
            '.weather_cache', 
            backend='sqlite', 
            expire_after=7200
        )

    def get_forecast(self, latitude: float, longitude: float, days: int = 1, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches hourly forecast for the given location.
        If date is provided (YYYY-MM-DD), fetches forecast for that specific day.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,weather_code",
            "timezone": "auto"
        }
        
        if date:
            params["start_date"] = date
            params["end_date"] = date
        else:
            params["forecast_days"] = days
        
        try:
            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch weather: {e}")
            return {}

    def get_current_conditions(self, latitude: float, longitude: float, date: Optional[str] = None) -> str:
        """
        Returns a simplified string summary of current/forecasted weather.
        """
        data = self.get_forecast(latitude, longitude, days=1, date=date)
        if not data or "hourly" not in data:
            return "Weather unavailable."
            
        # Analyze first 12 hours roughly
        temps = data["hourly"]["temperature_2m"][:12]
        precip = data["hourly"]["precipitation"][:12]
        
        avg_temp = sum(temps) / len(temps) if temps else 0
        total_rain = sum(precip) if precip else 0
        
        condition = "Clear"
        if total_rain > 5.0:
            condition = "Heavy Rain"
        elif total_rain > 0.5:
            condition = "Rainy"
            
        return f"{condition}, Avg Temp: {avg_temp:.1f}°C"

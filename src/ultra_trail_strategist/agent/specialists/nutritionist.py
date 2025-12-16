import logging
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from ultra_trail_strategist.mcp_server import get_race_forecast

logger = logging.getLogger(__name__)

class NutritionistAgent:
    """
    Specialist Agent focused on Nutrition and Hydration.
    Uses Weather data to adjust intake recommendations.
    """
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
    async def generate_nutrition_plan(self, segments: List[Any], lat: float, lon: float, race_date: Optional[str] = None) -> str:
        """
        Generates a nutrition plan based on course duration and weather.
        """
        # 1. Fetch Weather
        try:
            weather_report = await get_race_forecast(lat, lon, date=race_date)
        except Exception:
            weather_report = "Weather data unavailable."
            
        # 2. Calculate Context
        total_dist_km = sum(s.length for s in segments) / 1000
        # Rough estimation if pacing not available yet: 10 min/km default for ultras
        est_duration_hours = (total_dist_km * 10) / 60 
        
        # 3. LLM Advice
        system_prompt = """You are a precision Sports Nutritionist for ultra-endurance athletes.
        Rules:
        - Base carb intake: 60-90g/hr.
        - Base fluid: 500ml/hr.
        - IF HOT (>20°C): Increase fluid to 700-800ml/hr + electrolytes.
        - IF COLD (<5°C): Remind athlete to drink even if not thirsty.
        - IF LONG (>4 hours): Solid food strategy is mandatory.
        """
        
        human_template = """
        CONTEXT:
        Distance: {dist:.1f} km
        Est Duration: {dur:.1f} hours
        Weather: {weather}
        
        Please provide a concise Nutrition & Hydration Protocol.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_template)
        ])
        
        chain = prompt | self.llm
        result = chain.invoke({
            "dist": total_dist_km,
            "dur": est_duration_hours,
            "weather": weather_report
        })
        
        return f"NUTRITION REPORT:\n{result.content}"

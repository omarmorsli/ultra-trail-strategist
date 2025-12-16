import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from ultra_trail_strategist.agent.specialists.pacer import PacerAgent
from ultra_trail_strategist.agent.specialists.nutritionist import NutritionistAgent
from ultra_trail_strategist.feature_engineering.segmenter import Segment, SegmentType

class TestSpecialists(unittest.TestCase):
    
    @patch("ultra_trail_strategist.agent.specialists.pacer.get_activity_streams", new_callable=AsyncMock)
    def test_pacer_flow(self, mock_streams):
        # Setup Fake LLM
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda
        fake_llm = RunnableLambda(lambda x: AIMessage(content="Pace yourself."))
        
        mock_streams.return_value = [] # Empty streams
        
        agent = PacerAgent(fake_llm)
        
        segments = [Segment(start_dist=0, end_dist=1000, type=SegmentType.FLAT, avg_grade=0, elevation_gain=0, elevation_loss=0, length=1000)]
        history = [{"id": 1}]
        
        # Invoke
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(agent.generate_pacing_plan(segments, history))
        loop.close()
        
        self.assertIn("Insufficient data", result["report"])

    @patch("ultra_trail_strategist.agent.specialists.nutritionist.get_race_forecast", new_callable=AsyncMock)
    def test_nutritionist_flow(self, mock_weather):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda
        fake_llm = RunnableLambda(lambda x: AIMessage(content="Drink water."))
        
        mock_weather.return_value = "Sunny, 25C"
        
        agent = NutritionistAgent(fake_llm)
        segments = [Segment(start_dist=0, end_dist=10000, type=SegmentType.FLAT, avg_grade=0, elevation_gain=0, elevation_loss=0, length=10000)]
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(agent.generate_nutrition_plan(segments, 45.0, 6.0))
        loop.close()
        
        self.assertIn("NUTRITION REPORT", result)
        self.assertIn("Drink water", result)

if __name__ == "__main__":
    unittest.main()

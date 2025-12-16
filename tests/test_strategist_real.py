import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from ultra_trail_strategist.agent.strategist import StrategistAgent, RaceState
from ultra_trail_strategist.feature_engineering.segmenter import Segment, SegmentType

class TestStrategistAgent(unittest.TestCase):
    
    @patch("ultra_trail_strategist.agent.strategist.ChatOpenAI")
    @patch("ultra_trail_strategist.agent.strategist.get_recent_activities", new_callable=AsyncMock)
    @patch("ultra_trail_strategist.agent.strategist.get_activity_streams", new_callable=AsyncMock)
    def test_full_agent_flow(self, mock_get_streams, mock_get_activities, mock_llm_cls):
        
        agent = StrategistAgent()
        
        # Replace LLM with a RunnableLambda to satisfy LangChain type checks
        agent.llm = RunnableLambda(lambda x: AIMessage(content="Start slow, finish strong."))
        
        # Mock Tools
        mock_get_activities.return_value = [{"name": "Long Run", "id": 123, "distance_km": 20}]
        mock_get_streams.return_value = [] # Return empty list to test graceful degradation
        
        # Dummy Segments
        segments = [
            Segment(start_dist=0, end_dist=1000, type=SegmentType.FLAT, avg_grade=0, elevation_gain=0, elevation_loss=0, length=1000)
        ]
        
        initial_state = {
            "segments": segments,
            "athlete_history": [],
            "course_analysis": "",
            "pacing_plan": "",
            "final_strategy": ""
        }
        
        # Test Async Graph Execution
        async def run_test():
            return await agent.workflow.ainvoke(initial_state)
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_test())
        loop.close()
        
        # Assertions
        self.assertIn("Course Analysis", result["course_analysis"])
        self.assertEqual(result["athlete_history"][0]["name"], "Long Run")
        # Check that we handled empty streams gracefully (or check pacing logic if proper streams provided)
        self.assertIn("Could not fetch stream data", result["pacing_plan"])
        self.assertIn("Start slow", result["final_strategy"])

if __name__ == "__main__":
    unittest.main()

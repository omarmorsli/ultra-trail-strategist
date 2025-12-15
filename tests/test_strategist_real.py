import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from ultra_trail_strategist.agent.strategist import StrategistAgent, RaceState
from ultra_trail_strategist.feature_engineering.segmenter import Segment, SegmentType

class TestStrategistAgent(unittest.TestCase):
    
    @patch("ultra_trail_strategist.agent.strategist.ChatOpenAI")
    @patch("ultra_trail_strategist.agent.strategist.get_recent_activities", new_callable=AsyncMock)
    def test_full_agent_flow(self, mock_get_activities, mock_llm_cls):
        # Mock LLM
        mock_llm_instance = mock_llm_cls.return_value
        from langchain_core.messages import AIMessage
        mock_llm_instance.invoke.return_value = AIMessage(content="Start slow, finish strong.")
        
        # Mock Tool
        mock_get_activities.return_value = [{"name": "Long Run", "distance_km": 20}]
        
        agent = StrategistAgent()
        
        # Replace LLM with a RunnableLambda to satisfy LangChain type checks
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda
        
        agent.llm = RunnableLambda(lambda x: AIMessage(content="Start slow, finish strong."))
        
        # Dummy Segments
        segments = [
            Segment(start_dist=0, end_dist=1000, type=SegmentType.FLAT, avg_grade=0, elevation_gain=0, elevation_loss=0, length=1000)
        ]
        
        initial_state = {
            "segments": segments,
            "athlete_history": [],
            "course_analysis": "",
            "final_strategy": ""
        }
        
        # Test Async Graph Execution
        async def run_test():
            return await agent.workflow.ainvoke(initial_state)
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_test())
        loop.close()
        
        self.assertIn("Course Analysis", result["course_analysis"])
        self.assertEqual(result["athlete_history"][0]["name"], "Long Run")
        self.assertIn("Start slow", result["final_strategy"])

if __name__ == "__main__":
    unittest.main()

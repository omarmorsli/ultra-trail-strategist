import unittest
import asyncio
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from ultra_trail_strategist.pipeline import RaceDataPipeline
from ultra_trail_strategist.agent.strategist import StrategistAgent, RaceState
from ultra_trail_strategist.feature_engineering.segmenter import Segment, SegmentType

class TestIntegration(unittest.TestCase):

    @patch("ultra_trail_strategist.pipeline.GPXProcessor")
    @patch("ultra_trail_strategist.pipeline.CourseSegmenter")
    def test_pipeline_flow(self, mock_segmenter_cls, mock_gpx_cls):
        # Setup mocks
        mock_gpx = mock_gpx_cls.return_value
        mock_segmenter = mock_segmenter_cls.return_value
        
        # Mock segments return
        mock_segment = Segment(
            start_dist=0, end_dist=100, type=SegmentType.FLAT, 
            avg_grade=0, elevation_gain=0, elevation_loss=0, length=100
        )
        mock_segmenter.process.return_value = [mock_segment]
        
        pipeline = RaceDataPipeline("dummy.gpx")
        segments = pipeline.run()
        
        # Verify calls
        mock_gpx.load_from_file.assert_called_once()
        mock_gpx.to_dataframe.assert_called_once()
        mock_gpx.smooth_elevation.assert_called_once()
        mock_segmenter.process.assert_called_once()
        
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].type, SegmentType.FLAT)

    def test_agent_graph(self):
        agent = StrategistAgent()
        
        # Mock LLM to avoid API calls and ensure deterministic output
        agent.llm = RunnableLambda(lambda x: AIMessage(content="STRATEGY: Go fast."))
        
        # IMPORTANT: Propagate mock LLM to sub-agents
        agent.pacer.llm = agent.llm
        agent.nutritionist.llm = agent.llm

        # Create dummy segments
        segments = [
            Segment(
                start_dist=0, end_dist=1000, type=SegmentType.CLIMB,
                avg_grade=5.0, elevation_gain=50, elevation_loss=0, length=1000
            )
        ]
        
        initial_state = {
            "segments": segments,
            "athlete_history": [],
            "course_analysis": "",
            "pacing_report": "",
            "nutrition_report": "",
            "final_strategy": ""
        }
        
        # Run graph asynchronously
        async def run_test():
            return await agent.workflow.ainvoke(initial_state)
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_test())
        loop.close()
        
        self.assertIn("Course Stats", result["course_analysis"])
        self.assertIn("1.0km", result["course_analysis"])
        self.assertIn("STRATEGY", result["final_strategy"])

if __name__ == "__main__":
    unittest.main()

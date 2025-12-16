import os
import asyncio
from typing import List, Dict, Any, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from ultra_trail_strategist.feature_engineering.segmenter import Segment
from ultra_trail_strategist.feature_engineering.pace_model import PacePredictor
# Import tool functions directly
from ultra_trail_strategist.mcp_server import get_recent_activities, get_activity_streams

class RaceState(TypedDict):
    """
    State for the Strategist Agent.
    """
    segments: List[Segment]
    athlete_history: List[Dict[str, Any]]
    course_analysis: str
    pacing_plan: str  # New field for ML output
    final_strategy: str

class StrategistAgent:
    """
    Agentic workflow to generate race strategy using Course Data + Strava History + ML Pacing.
    """
    def __init__(self):
        # simple check to avoid crash if env var is missing during tests
        api_key = os.getenv("OPENAI_API_KEY", "dummy-key-for-testing")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=api_key)
        self.workflow = self._build_graph()
        self.pace_model = PacePredictor()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(RaceState)
        
        # Add nodes
        workflow.add_node("analyze_course", self.analyze_course)
        workflow.add_node("fetch_athlete_history", self.fetch_athlete_history)
        workflow.add_node("generate_pacing", self.generate_pacing)
        workflow.add_node("generate_strategy", self.generate_strategy)
        
        # Define edges
        workflow.set_entry_point("analyze_course")
        workflow.add_edge("analyze_course", "fetch_athlete_history")
        workflow.add_edge("fetch_athlete_history", "generate_pacing")
        workflow.add_edge("generate_pacing", "generate_strategy")
        workflow.add_edge("generate_strategy", END)
        
        return workflow.compile()

    def analyze_course(self, state: RaceState) -> Dict[str, Any]:
        """
        Analyzes the segments to extract key course stats.
        """
        segments = state["segments"]
        total_dist = sum(s.length for s in segments)
        total_gain = sum(s.elevation_gain for s in segments)
        
        # Create a condensed string representation of segments for the LLM
        segment_summary = "\n".join([
            f"- {s.type.value} ({s.length/1000:.1f}km, Avg Grade: {s.avg_grade:.1f}%)"
            for s in segments
        ])

        analysis = (
            f"Course Analysis:\n"
            f"Total Distance: {total_dist/1000:.2f} km\n"
            f"Total Elevation Gain: {total_gain:.0f} m\n\n"
            f"Key Segments:\n{segment_summary}"
        )
        
        return {"course_analysis": analysis}

    async def fetch_athlete_history(self, state: RaceState) -> Dict[str, Any]:
        """
        Fetches athlete history using the MCP tool.
        """
        try:
            history = await get_recent_activities(limit=5)
        except Exception:
            history = [] 
            
        return {"athlete_history": history}

    async def generate_pacing(self, state: RaceState) -> Dict[str, Any]:
        """
        Trains ML model on history and predicts pace for course segments.
        """
        history = state.get("athlete_history", [])
        segments = state.get("segments", [])
        
        if not history:
            return {"pacing_plan": "No history available for ML pacing."}

        # 1. Fetch Streams for Training
        activity_ids = [a["id"] for a in history if a.get("id")]
        try:
            # Limit to recent 5 to save API calls/time
            streams = await get_activity_streams(activity_ids[:5])
        except Exception:
            streams = []

        if not streams:
            return {"pacing_plan": "Could not fetch stream data for pacing model."}

        # 2. Train Model
        self.pace_model.train(streams)
        
        # 3. Predict for each segment
        predicted_splits = []
        total_predicted_time_min = 0
        
        for i, seg in enumerate(segments):
            pace_min_km = self.pace_model.predict_segment(seg.avg_grade)
            segment_time_min = pace_min_km * (seg.length / 1000)
            total_predicted_time_min += segment_time_min
            
            predicted_splits.append(
                f"Seg {i+1} ({seg.type.value}, {seg.length/1000:.1f}km): "
                f"{pace_min_km:.1f} min/km -> {segment_time_min:.0f} min"
            )

        total_hours = total_predicted_time_min / 60
        pacing_summary = (
            f"ML PACING PREDICTION (Based on {len(streams)} activities):\n"
            f"Estimated Finish Time: {total_hours:.1f} hours\n\n"
            "Splits:\n" + "\n".join(predicted_splits)
        )
        
        return {"pacing_plan": pacing_summary}

    def generate_strategy(self, state: RaceState) -> Dict[str, Any]:
        """
        Generates the narrative strategy using the gathered context and ML pacing.
        """
        course_context = state["course_analysis"]
        history_context = state["athlete_history"]
        pacing_context = state.get("pacing_plan", "No pacing plan generated.")
        
        # Construct Prompt
        system_prompt = """You are an expert Ultra-Trail Coach and Strategist.
        Your goal is to create a segment-by-segment race strategy based on the course profile, athlete's history, and ML-generated pacing predictions.
        
        Guidelines:
        1. Analyze the course vertically (climbs/descents).
        2. Integrate the ML Pacing estimates into your advice (e.g., "The model predicts 15 min for this climb, try to hold steady HR").
        3. Be concise but encouraging.
        """
        
        human_template = """
        COURSE DATA:
        {course}
        
        ATHLETE RECENT HISTORY:
        {history}
        
        ML PACING PREDICTIONS:
        {pacing}
        
        Please generate the Race Strategy.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_template)
        ])
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "course": course_context,
            "history": str(history_context),
            "pacing": pacing_context
        })
        
        return {"final_strategy": response.content}

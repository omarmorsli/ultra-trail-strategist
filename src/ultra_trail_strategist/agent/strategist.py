import os
from typing import List, Dict, Any, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from ultra_trail_strategist.feature_engineering.segmenter import Segment
# Import tool functions directly since we are in the same python environment
# In a distributed MCP setup, we would use an MCP Client to discover these.
from ultra_trail_strategist.mcp_server import get_recent_activities

class RaceState(TypedDict):
    """
    State for the Strategist Agent.
    """
    segments: List[Segment]
    athlete_history: List[Dict[str, Any]]
    course_analysis: str
    final_strategy: str

class StrategistAgent:
    """
    Agentic workflow to generate race strategy using Course Data + Strava History.
    """
    def __init__(self):
        # simple check to avoid crash if env var is missing during tests
        api_key = os.getenv("OPENAI_API_KEY", "dummy-key-for-testing")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=api_key)
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(RaceState)
        
        # Add nodes
        workflow.add_node("analyze_course", self.analyze_course)
        workflow.add_node("fetch_athlete_history", self.fetch_athlete_history)
        workflow.add_node("generate_strategy", self.generate_strategy)
        
        # Define edges
        workflow.set_entry_point("analyze_course")
        workflow.add_edge("analyze_course", "fetch_athlete_history")
        workflow.add_edge("fetch_athlete_history", "generate_strategy")
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
        # In this local version, we call the async function directly.
        # Ideally, the LLM would decide to call this, but for this fixed pipeline
        # we enforce fetching history to provide context.
        try:
            history = await get_recent_activities(limit=5)
        except Exception:
            history = [] # Fallback if no creds or error
            
        return {"athlete_history": history}

    def generate_strategy(self, state: RaceState) -> Dict[str, Any]:
        """
        Generates the narrative strategy using the gathered context.
        """
        course_context = state["course_analysis"]
        history_context = state["athlete_history"]
        
        # Construct Prompt
        system_prompt = """You are an expert Ultra-Trail Coach and Strategist.
        Your goal is to create a segment-by-segment race strategy based on the course profile and the athlete's recent training status.
        
        Guidelines:
        1. Analyze the course vertically (climbs/descents).
        2. Reference the athlete's recent volume/fitness to gauge aggressiveness.
        3. Provide specific advice for each major segment (nutrition, pacing, mental focus).
        4. Be concise but encouraging.
        """
        
        human_template = """
        COURSE DATA:
        {course}
        
        ATHLETE RECENT HISTORY:
        {history}
        
        Please generate the Race Strategy.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_template)
        ])
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "course": course_context,
            "history": str(history_context)
        })
        
        return {"final_strategy": response.content}

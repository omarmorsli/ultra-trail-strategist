import os
import asyncio
from typing import List, Dict, Any, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from ultra_trail_strategist.feature_engineering.segmenter import Segment
from ultra_trail_strategist.mcp_server import get_recent_activities
from ultra_trail_strategist.agent.specialists.pacer import PacerAgent
from ultra_trail_strategist.agent.specialists.nutritionist import NutritionistAgent

class RaceState(TypedDict):
    """
    State for the Multi-Agent System.
    """
    segments: List[Segment]
    athlete_history: List[Dict[str, Any]]
    course_analysis: str
    pacing_report: str      # Output from Pacer
    nutrition_report: str   # Output from Nutritionist
    final_strategy: str     # Output from Principal

class StrategistAgent:
    """
    Principal Agent that orchestrates the Pacer and Nutritionist specialists.
    """
    def __init__(self):
        # simple check to avoid crash if env var is missing during tests
        api_key = os.getenv("OPENAI_API_KEY", "dummy-key-for-testing")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=api_key)
        
        # Specialists
        self.pacer = PacerAgent(self.llm)
        self.nutritionist = NutritionistAgent(self.llm)
        
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(RaceState)
        
        # Nodes
        workflow.add_node("analyze_course", self.analyze_course)
        workflow.add_node("fetch_history", self.fetch_history)
        workflow.add_node("run_pacer", self.run_pacer)
        workflow.add_node("run_nutritionist", self.run_nutritionist)
        workflow.add_node("write_final_strategy", self.write_final_strategy)
        
        # Edges
        workflow.set_entry_point("analyze_course")
        workflow.add_edge("analyze_course", "fetch_history")
        
        # Parallel Execution: After history is fetched, run Pacer and Nutritionist
        # Note: LangGraph doesn't do true async parallel by default without explicit fan-out, 
        # but structurally they are independent. We'll run them sequentially in graph for simplicity 
        # or use map-reduce if needed. Here: Sequential for clarity (History -> Pacer -> Nutrition -> Final)
        workflow.add_edge("fetch_history", "run_pacer")
        workflow.add_edge("run_pacer", "run_nutritionist")
        workflow.add_edge("run_nutritionist", "write_final_strategy")
        workflow.add_edge("write_final_strategy", END)
        
        return workflow.compile()

    def analyze_course(self, state: RaceState) -> Dict[str, Any]:
        """Summarize course stats."""
        segments = state["segments"]
        total_dist = sum(s.length for s in segments)
        total_gain = sum(s.elevation_gain for s in segments)
        
        analysis = (
            f"Course Stats: {total_dist/1000:.1f}km, {total_gain:.0f}m gain.\n"
            f"Segments: {len(segments)} segments detected."
        )
        return {"course_analysis": analysis}

    async def fetch_history(self, state: RaceState) -> Dict[str, Any]:
        """Get athlete history for Pacer."""
        try:
            history = await get_recent_activities(limit=5)
        except Exception:
            history = []
        return {"athlete_history": history}

    async def run_pacer(self, state: RaceState) -> Dict[str, Any]:
        """Delegate to PacerAgent."""
        # Need segments and history
        report = await self.pacer.generate_pacing_plan(
            state["segments"], 
            state["athlete_history"]
        )
        return {"pacing_report": report}

    async def run_nutritionist(self, state: RaceState) -> Dict[str, Any]:
        """Delegate to NutritionistAgent."""
        # Hardcoded Lat/Lon for now (Chamonix approx) or extract from GPX metadata if improved
        # Future TODO: Extract lat/lon from first segment point
        report = await self.nutritionist.generate_nutrition_plan(
            state["segments"], 
            45.92, 6.86 # Chamonix
        )
        return {"nutrition_report": report}

    def write_final_strategy(self, state: RaceState) -> Dict[str, Any]:
        """
        Principal Agent synthesizes the specialist reports.
        """
        system_prompt = """You are the Principal Ultra-Trail Coach.
        You have received reports from your specialist team (Pacer and Nutritionist).
        Your job is to synthesize these into a single, cohesive, motivating race strategy document for the athlete.
        
        Do not just copy-paste. Weave the nutrition advice into the pacing strategy.
        """
        
        human_template = """
        COURSE: {course}
        
        PACER REPORT:
        {pacing}
        
        NUTRITIONIST REPORT:
        {nutrition}
        
        Generate the Final Strategy.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_template)
        ])
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "course": state["course_analysis"],
            "pacing": state["pacing_report"],
            "nutrition": state["nutrition_report"]
        })
        
        return {"final_strategy": response.content}

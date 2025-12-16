import logging
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ultra_trail_strategist.feature_engineering.pace_model import PacePredictor
from ultra_trail_strategist.feature_engineering.fatigue_model import FatigueModel
from ultra_trail_strategist.mcp_server import get_activity_streams

logger = logging.getLogger(__name__)

class PacerAgent:
    """
    Specialist Agent focused on Race Pacing.
    Uses ML models to predict split times.
    """
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.pace_model = PacePredictor()
        
    async def generate_pacing_plan(self, segments: List[Any], athlete_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes the pacing workflow:
        1. Fetch Streams for recent history.
        2. Train ML Model.
        3. Predict paces for all segments.
        4. (Optional) Use LLM to summarize/contextualize the data table.
        """
        # 1. Fetch Streams
        activity_ids = [a["id"] for a in athlete_history if a.get("id")]
        try:
            streams = await get_activity_streams(activity_ids[:5])
        except Exception as e:
            logger.error(f"Pacer failed to fetch streams: {e}")
            streams = []

        if not streams:
            return {"report": "PACING PLAN: Insufficient data to generate ML predictions.", "data": []}

        # 2. Train Model
        self.pace_model.train(streams)
        
        # 3. Predict Segments
        predicted_splits = []
        structured_data = [] # For Dashboard Plotting
        total_time_min = 0
        
        # Initialize Physiology Model (Default CP 4:30 min/km for now)
        fatigue_model = FatigueModel(critical_pace_min_km=4.5)
        
        for i, seg in enumerate(segments):
            # 1. Base ML Prediction (Fresh State)
            base_pace = self.pace_model.predict_segment(seg.avg_grade)
            
            # 2. Check Fatigue Status
            penalty = fatigue_model.get_penalty_factor()
            
            # 3. Apply Penalty (Slow down if exhausted)
            pace_min_km = base_pace * penalty
            
            # 4. Update Physiological State based on actual effort
            seg_len_km = seg.length / 1000
            fatigue_model.update_balance(pace_min_km, seg_len_km)
            
            segment_time_min = pace_min_km * seg_len_km
            total_time_min += segment_time_min
            
            # Formatting note: Add * if penalty applied
            note = " (BONK!)" if penalty > 1.1 else ""
            
            predicted_splits.append(
                f"- Seg {i+1} ({seg.type.value}, {seg_len_km:.1f}km, {seg.avg_grade:.1f}%): "
                f"{pace_min_km:.1f} min/km | {segment_time_min:.0f} min{note}"
            )
            
            structured_data.append({
                "segment_index": i,
                "start_dist": seg.start_dist,
                "end_dist": seg.end_dist,
                "pace_min_km": pace_min_km,
                "time_min": segment_time_min,
                "fatigue_level": fatigue_model.get_exhaustion_level() # New data point!
            })

        total_hours = total_time_min / 60
        
        # 4. LLM Summary (Optional, but adds "flavor")
        data_block = "\n".join(predicted_splits)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional Race Pacer. Summarize this split table into a concise pacing directive."),
            ("human", f"Estimated Finish: {total_hours:.1f} hours.\nSplits:\n{data_block}")
        ])
        
        chain = prompt | self.llm
        result = chain.invoke({})
        
        report_str = f"PACING REPORT:\n{result.content}\n\nDETAILED SPLITS:\n{data_block}"
        
        return {
            "report": report_str,
            "data": structured_data
        }

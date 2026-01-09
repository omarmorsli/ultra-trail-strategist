import logging
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ultra_trail_strategist.feature_engineering.drift_analyzer import DriftAnalyzer
from ultra_trail_strategist.feature_engineering.fatigue_model import FatigueModel
from ultra_trail_strategist.feature_engineering.pace_model import PacePredictor
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

    async def generate_pacing_plan(
        self,
        segments: List[Any],
        athlete_history: List[Dict[str, Any]],
        actual_splits: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the pacing workflow:
        1. Fetch Streams for recent history.
        2. Train ML Model.
        3. Predict paces for all segments.
        4. (Optional) Use LLM to summarize/contextualize the data table.
        """
        if actual_splits is None:
            actual_splits = {}

        # 1. Fetch Streams
        activity_ids = [a["id"] for a in athlete_history if a.get("id")]
        try:
            streams = await get_activity_streams(activity_ids[:5])
        except Exception as e:
            logger.error(f"Pacer failed to fetch streams: {e}")
            streams = []

        if not streams:
            return {
                "report": "PACING PLAN: Insufficient data to generate ML predictions.",
                "data": [],
            }

        # 2. Train Model
        self.pace_model.train(streams)

        # 3. Calculate athlete-specific Critical Pace (CP) from activity history
        # CP = threshold pace below which you can sustain effort indefinitely
        # Use median pace from recent activities as a proxy
        athlete_paces = []
        for activity in athlete_history:
            avg_speed = activity.get("average_speed")  # m/s from Strava
            if avg_speed and avg_speed > 0:
                pace_min_km = (1000 / avg_speed) / 60  # Convert to min/km
                athlete_paces.append(pace_min_km)

        if athlete_paces:
            # Use 75th percentile as CP (most activities are easier than race pace)
            critical_pace = float(np.percentile(athlete_paces, 75))
            logger.info(f"Calculated Critical Pace from athlete data: {critical_pace:.2f} min/km")
        else:
            # Fallback for recreational runner (6 min/km is ~10 km/h)
            critical_pace = 6.0
            logger.warning("No pace data available, using default CP: 6.0 min/km")

        # 4. Analyze Cardiac Drift from activity streams
        drift_analyzer = DriftAnalyzer()
        endurance_factor = drift_analyzer.calculate_endurance_factor(streams)
        # Convert endurance factor to hourly drift penalty
        # If endurance_factor is 0.95, that means 5% total drift expected
        drift_penalty_base = (1.0 - endurance_factor) / 3.0  # Spread over ~3 hours
        logger.info(
            f"Endurance factor: {endurance_factor:.2f}, "
            f"drift penalty: {drift_penalty_base:.1%}/hour after hour 3"
        )

        # 5. Predict Segments
        predicted_splits = []
        structured_data = []  # For Dashboard Plotting
        total_time_min = 0.0

        # Initialize Physiology Model with athlete-specific CP
        fatigue_model = FatigueModel(critical_pace_min_km=critical_pace)

        for i, seg in enumerate(segments):
            seg_len_km = seg.length / 1000

            # Check if we have an ACTUAL split for this segment
            # If so, use it directly and update state without applying penalties.
            if i in actual_splits:
                segment_time_min = actual_splits[i]
                pace_min_km = segment_time_min / seg_len_km if seg_len_km > 0 else 0

                # Update Fatigue Model state even for past segments to track accumulation
                fatigue_model.update_balance(pace_min_km, seg_len_km)

                total_time_min += segment_time_min

                predicted_splits.append(
                    f"- Seg {i + 1} ({seg.type.value}, {seg_len_km:.1f}km): "
                    f"ACTUAL {segment_time_min:.0f} min ({pace_min_km:.1f} min/km) ✅"
                )

                structured_data.append(
                    {
                        "segment_index": i,
                        "start_dist": seg.start_dist,
                        "end_dist": seg.end_dist,
                        "pace_min_km": pace_min_km,
                        "time_min": segment_time_min,
                        "fatigue_level": fatigue_model.get_exhaustion_level(),
                        "is_actual": True,
                        "cumulative_time_min": total_time_min,
                    }
                )
                continue

            # --- Future Prediction Logic ---

            # 1. Base ML Prediction (Fresh State)
            base_pace = self.pace_model.predict_segment(seg.avg_grade)

            # 2. Check Fatigue Status (W' Balance)
            w_prime_penalty = fatigue_model.get_penalty_factor()

            # 3. Check Cardiac Drift (Endurance Decay)
            # Apply after hour 3
            current_hours = total_time_min / 60
            drift_penalty = 1.0
            if current_hours > 3.0:
                # Decay per hour after hour 3
                excess_hours = current_hours - 3.0
                drift_penalty = 1.0 + (drift_penalty_base * excess_hours)

            # 3b. Check Surface Drag
            surface_penalties = {
                "asphalt": 1.0,
                "concrete": 1.0,
                "unpaved": 1.05,
                "gravel": 1.05,
                "dirt": 1.05,
                "grass": 1.08,
                "path": 1.10,
                "track": 1.10,
                "trail": 1.10,
                "alpine": 1.25,  # High alpine technical
                "unknown": 1.05,  # Conservative default for "unknown" in wild areas
            }
            # Fallback to key containing substring if direct match fails? No, simpler:
            surface = getattr(seg, "surface", "unknown").lower()
            surface_penalty = surface_penalties.get(surface, 1.05)

            total_penalty = w_prime_penalty * drift_penalty * surface_penalty

            # 4. Apply Penalty
            pace_min_km = base_pace * total_penalty

            # 5. Update Physiological State based on actual effort
            fatigue_model.update_balance(pace_min_km, seg_len_km)

            segment_time_min = pace_min_km * seg_len_km
            total_time_min += segment_time_min

            # Formatting note
            notes = []
            if w_prime_penalty > 1.1:
                notes.append("BONK")
            if drift_penalty > 1.05:
                notes.append("DRIFT")
            note_str = " (" + ", ".join(notes) + "!)" if notes else ""

            predicted_splits.append(
                f"- Seg {i + 1} ({seg.type.value}, {seg_len_km:.1f}km, {seg.avg_grade:.1f}%): "
                f"{pace_min_km:.1f} min/km | {segment_time_min:.0f} min{note_str}"
            )

            structured_data.append(
                {
                    "segment_index": i,
                    "start_dist": seg.start_dist,
                    "end_dist": seg.end_dist,
                    "pace_min_km": pace_min_km,
                    "time_min": segment_time_min,
                    "fatigue_level": fatigue_model.get_exhaustion_level(),
                    "is_actual": False,
                    "cumulative_time_min": total_time_min,
                }
            )

        total_hours = total_time_min / 60

        # 4. LLM Summary (Optional, but adds "flavor")
        data_block = "\n".join(predicted_splits)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a professional Race Pacer. Summarize this split table into a concise pacing directive.",  # noqa: E501
                ),
                ("human", f"Estimated Finish: {total_hours:.1f} hours.\nSplits:\n{data_block}"),
            ]
        )

        chain = prompt | self.llm
        result = chain.invoke({})

        report_str = f"PACING REPORT:\n{result.content}\n\nDETAILED SPLITS:\n{data_block}"

        return {"report": report_str, "data": structured_data}

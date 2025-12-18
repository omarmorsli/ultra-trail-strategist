import asyncio
import logging
import os

import streamlit as st
from dotenv import load_dotenv

from ultra_trail_strategist.agent.strategist import StrategistAgent
from ultra_trail_strategist.data_ingestion.health_client import HealthClient
from ultra_trail_strategist.pipeline import RaceDataPipeline
from ultra_trail_strategist.state_manager import RaceStateManager
from ultra_trail_strategist.ui.components import (
    render_elevation_profile,
    render_pacing_charts,
    render_race_mode,
    render_sidebar,
)
from ultra_trail_strategist.ui.map_renderer import render_3d_course, render_course_map

# Config Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Load Env
load_dotenv()

# App Title & Config
st.set_page_config(page_title="Ultra-Trail Strategist", page_icon="🏃‍♂️", layout="wide")


# --- Cached Loaders ---
@st.cache_resource
def load_agent():
    return StrategistAgent()


# --- Main App ---
def main():
    st.title("🏃‍♂️ Ultra-Trail Strategist AI")

    # Init Components
    health_client = HealthClient()
    state_manager = RaceStateManager()

    # --- Sidebar ---
    uploaded_file, race_date, demo_mode, readiness = render_sidebar(health_client)
    mode = st.sidebar.radio("App Mode", ["Planning 📝", "Live Race ⏱️"])

    # --- Data Loading ---
    gpx_path = None
    if demo_mode:
        gpx_path = "data/dummy.gpx"
    elif uploaded_file:
        # Save uploaded file
        # Ensure data dir exists
        os.makedirs("data", exist_ok=True)
        gpx_path = f"data/{uploaded_file.name}"
        with open(gpx_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    if not gpx_path:
        st.info("👈 Please upload a GPX file or select Demo Mode to start.")
        st.stop()

    # --- Initialize Pipeline ---
    pipeline = RaceDataPipeline(gpx_path)

    # Simple caching of pipeline results in session state to avoid re-parsing on every interaction
    if "pipeline_gpx" not in st.session_state or st.session_state["pipeline_gpx"] != gpx_path:
        with st.spinner("Processing Course Data..."):
            pipeline.run()
            st.session_state["pipeline_obj"] = pipeline
            st.session_state["pipeline_gpx"] = gpx_path

    pipeline = st.session_state["pipeline_obj"]
    segments = pipeline.get_segments()
    course_df = pipeline.get_dataframe()

    # --- Live Race Mode ---
    if mode == "Live Race ⏱️":
        render_race_mode(state_manager, segments)

        st.divider()
        st.subheader("Map Visualization")

        map_tab1, map_tab2 = st.tabs(["🗺️ 2D Map & Radar", "🏔️ 3D Flyover"])

        with map_tab1:
            show_radar = st.toggle("Show Weather Radar 🌧️", value=True, key="radar_live")
            render_course_map(course_df, show_radar=show_radar)

        with map_tab2:
            st.info("💡 Hold right-click to rotate/tilt the view.")
            render_3d_course(course_df)

        st.divider()
        st.subheader("Updated Projections")

        if st.button("🔄 Re-Calculate Strategy"):
            # Load Agent
            agent = load_agent()
            race_state = state_manager.get_state()

            initial_state = {
                "segments": segments,
                "athlete_history": [],
                "course_analysis": "",
                "pacing_report": "",
                "pacing_data": [],
                "nutrition_report": "",
                "readiness": readiness,
                "race_date": str(race_date) if race_date else None,
                "final_strategy": "",
                "actual_splits": race_state.actual_splits,
            }

            with st.spinner("Re-Forecasting based on live splits..."):
                final_state = asyncio.run(agent.workflow.ainvoke(initial_state))
                st.session_state["final_state"] = final_state
                st.session_state["last_run_mode"] = "live"
                st.rerun()

    # --- Planning Mode ---
    else:
        # Show Course Profile
        col1, col2 = st.columns([3, 1])
        with col1:
            render_elevation_profile(course_df)
        with col2:
            st.metric("Total Distance", f"{course_df['distance'].max() / 1000:.1f} km")
            st.metric("Total Segments", len(segments))

        st.subheader("Course Map")

        map_tab1, map_tab2 = st.tabs(["🗺️ 2D Map & Radar", "🏔️ 3D Flyover"])

        with map_tab1:
            show_radar = st.toggle("Show Weather Radar 🌧️", key="radar_plan")
            render_course_map(course_df, show_radar=show_radar)

        with map_tab2:
            st.info("💡 Hold right-click to rotate/tilt the view.")
            render_3d_course(course_df)

        if st.button("🚀 Generate Race Strategy"):
            agent = load_agent()
            initial_state = {
                "segments": segments,
                "athlete_history": [],
                "course_analysis": "",
                "pacing_report": "",
                "pacing_data": [],
                "nutrition_report": "",
                "readiness": readiness,
                "race_date": str(race_date) if race_date else None,
                "final_strategy": "",
                "actual_splits": {},  # Planning implies no actuals yet
            }

            with st.spinner("Agents are analyzing segment by segment..."):
                final_state = asyncio.run(agent.workflow.ainvoke(initial_state))
                st.session_state["final_state"] = final_state
                st.session_state["last_run_mode"] = "planning"

    # --- Result Display (Common) ---
    if "final_state" in st.session_state:
        result = st.session_state["final_state"]

        st.divider()
        st.success("Strategy Ready!")

        tab1, tab2, tab3 = st.tabs(["🏁 Coach Strategy", "⏱ Pacing Plan", "🥗 Nutrition"])

        with tab1:
            st.markdown(result["final_strategy"])

        with tab2:
            st.markdown("### ML Pacing Predictions")
            # If live mode, maybe highlight actuals vs predicted?
            # The chart component logic in components.py should handle it.
            render_pacing_charts(result.get("pacing_data"))
            st.text(result["pacing_report"])

        with tab3:
            st.markdown("### Weather-Adaptive Nutrition")
            st.text(result["nutrition_report"])


if __name__ == "__main__":
    main()

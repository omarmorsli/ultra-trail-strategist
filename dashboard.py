import streamlit as st
import os
import asyncio
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

# App Title & Config
st.set_page_config(page_title="Ultra-Trail Strategist", page_icon="🏃‍♂️", layout="wide")

# Load Env
load_dotenv()

# Config Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Imports (cached)
@st.cache_resource
def load_agent():
    # Lazy import to avoid loading heavy libs until needed
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
    from ultra_trail_strategist.pipeline import RaceDataPipeline
    from ultra_trail_strategist.agent.strategist import StrategistAgent
    return StrategistAgent()

@st.cache_data
def process_gpx(file_content):
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
    from ultra_trail_strategist.pipeline import RaceDataPipeline
    
    # Save to temp file because pipeline expects path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gpx") as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name
        
    pipeline = RaceDataPipeline(tmp_path)
    segments = pipeline.run()
    
    # Also get raw df for plotting
    df = pipeline.df.to_pandas()
    
    return segments, df, tmp_path

# --- UI ---
st.title("🏃‍♂️ Ultra-Trail Strategist AI")
from ultra_trail_strategist.data_ingestion.health_client import HealthClient
st.markdown("### Intelligent Race Strategy & Pacing")

# Init Clients
health_client = HealthClient()

# Import UI Components
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from ultra_trail_strategist.ui.components import render_sidebar, render_elevation_profile, render_pacing_charts

# Sidebar
uploaded_file, race_date, demo_mode, readiness = render_sidebar(health_client)

if uploaded_file or demo_mode:
    with st.spinner("Analyzing Course Topography..."):
        if demo_mode and not uploaded_file:
             # Dummy Data for Demo
            dates = pd.date_range(start='1/1/2024', periods=100, freq='T')
            df_dummy = pd.DataFrame({
                'distance': [i*100 for i in range(100)],
                'elevation': [1000 + (i*10 if i<50 else (100-i)*10) for i in range(100)],
                'grade': [10 if i<50 else -10 for i in range(100)]
            })
            segments, df, tmp_path = [], df_dummy, "dummy"
            # We skip actual agent processing in pure demo UI if not really running
            st.warning("Running in pure Demo Visual Check Mode (No AI)")
        else:
            segments, df, tmp_path = process_gpx(uploaded_file.getvalue())

    # 1. Course Visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Elevation Profile")
        render_elevation_profile(df)
        
    with col2:
        st.subheader("Stats")
        total_dist = df["distance"].max() / 1000 if not df.empty else 0
        total_gain = df[df["grade"] > 0]["elevation"].diff().sum() if not df.empty else 0 # Rough approx
        
        st.metric("Distance", f"{total_dist:.1f} km")
        st.metric("Segments", len(segments))
        
    # 2. AI Strategy
    if st.button("Generate AI Strategy"):
        if demo_mode:
            st.error("Please upload a real GPX to generate a real strategy.")
        else:
            try:
                agent = load_agent()
                
                initial_state = {
                    "segments": segments,
                    "athlete_history": [],
                    "course_analysis": "",
                    "pacing_report": "",
                    "pacing_data": [],
                    "nutrition_report": "",
                    "readiness": readiness,
                    "race_date": str(race_date),
                    "final_strategy": ""
                }
                
                with st.spinner("AI Agents working... (Fetching History, Weather, Calculating Splits)"):
                    # Async wrapper for Streamlit
                    async def run_agent():
                        return await agent.workflow.ainvoke(initial_state)
                    
                    # Run in separate thread to avoid "loop already running" errors in Streamlit
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(asyncio.run, run_agent())
                        result = future.result()
                
                st.success("Strategy Generated!")
                
                tab1, tab2, tab3 = st.tabs(["🏁 Coach Strategy", "⏱ Pacing Plan", "🥗 Nutrition"])
                
                with tab1:
                    st.markdown(result["final_strategy"])
                    
                with tab2:
                    st.markdown("### ML Pacing Predictions")
                    render_pacing_charts(result.get("pacing_data"))
                    st.text(result["pacing_report"])
                    
                with tab3:
                    st.markdown("### Weather-Adaptive Nutrition")
                    st.text(result["nutrition_report"])
                    
            except Exception as e:
                st.error(f"Error generating strategy: {e}")

else:
    st.info("👆 Upload a GPX file to get started.")


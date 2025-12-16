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

# Sidebar
with st.sidebar:
    st.image("assets/logo_UTS.png", width=200)
    st.title("Settings")
    st.header("Input")
    uploaded_file = st.file_uploader("Upload GPX File", type=["gpx"])
    race_date = st.date_input("Race Date")
    demo_mode = st.checkbox("Use Demo Data", value=False)
    
    if os.getenv("STRAVA_CLIENT_ID"):
        st.success("✅ Strava Connected")
    else:
        st.error("❌ Strava Config Missing")
        
    # Garmin Check
    if health_client.garmin_client:
        st.success("✅ Garmin Connected")
        auto_readiness = health_client.get_readiness_score()
        readiness_source = "Garmin (Auto)"
    else:
        st.warning("⚠️ Garmin Disconnected")
        auto_readiness = 50
        readiness_source = "Manual"

    st.divider()
    st.header("Bio-Metrics")
    
    # Slider with auto-value as default
    readiness = st.slider(
        f"Readiness Score ({readiness_source})", 
        0, 100, 
        auto_readiness, 
        help="0=Exhausted, 100=Peak. Auto-fetched from Garmin if connected."
    )

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
        fig = px.area(df, x="distance", y="elevation", title="Course Elevation (m)")
        fig.update_layout(showlegend=False)
        fig.update_traces(line_color='#FF4B4B', fillcolor='rgba(255, 75, 75, 0.1)')
        st.plotly_chart(fig, use_container_width=True)
        
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
                    
                    # Visualize Pacing
                    if result.get("pacing_data"):
                        pace_data = result["pacing_data"]
                        # Create plotting data for step chart
                        plot_data = []
                        for const in pace_data:
                            # Step logic: same pace from start to end of segment
                            plot_data.append({"distance": const["start_dist"], "pace": const["pace_min_km"]})
                            plot_data.append({"distance": const["end_dist"], "pace": const["pace_min_km"]})
                        
                        df_pace = pd.DataFrame(plot_data)
                        
                        fig_pace = px.line(df_pace, x="distance", y="pace", title="Projected Pace (min/km)")
                        fig_pace.update_yaxes(autorange="reversed") # Faster pace is lower number usually, but for min/km higher is slower. Usually runners like lower is higher Y? Or standard inverting. 
                        # Let's keep standard (higher Y = slower). 
                        fig_pace.update_layout(yaxis_title="Pace (min/km)")
                        st.plotly_chart(fig_pace, use_container_width=True)

                        # Visualize Physiological Battery (W' Balance)
                        battery_data = []
                        for const in pace_data:
                            # Invert exhaustion (0.0 -> 1.0) to Battery (100% -> 0%)
                            exhaustion = const.get("fatigue_level", 0.0)
                            battery = (1.0 - exhaustion) * 100.0
                            
                            battery_data.append({"distance": const["start_dist"], "battery": battery})
                            battery_data.append({"distance": const["end_dist"], "battery": battery})
                            
                        df_battery = pd.DataFrame(battery_data)
                        fig_batt = px.area(df_battery, x="distance", y="battery", title="Physiological Battery (W' Balance)")
                        fig_batt.update_yaxes(range=[0, 100], title="Energy Reserves (%)")
                        fig_batt.update_traces(line_color='#2ECC71', fillcolor='rgba(46, 204, 113, 0.2)')
                        st.plotly_chart(fig_batt, use_container_width=True)

                    st.text(result["pacing_report"])
                    
                with tab3:
                    st.markdown("### Weather-Adaptive Nutrition")
                    st.text(result["nutrition_report"])
                    
            except Exception as e:
                st.error(f"Error generating strategy: {e}")

else:
    st.info("👆 Upload a GPX file to get started.")


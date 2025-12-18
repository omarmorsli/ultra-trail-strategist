import os

import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import streamlit as st


def render_sidebar(health_client):
    """
    Renders the sidebar and returns user inputs.
    """
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
            0,
            100,
            auto_readiness,
            help="0=Exhausted, 100=Peak. Auto-fetched from Garmin if connected.",
        )

    return uploaded_file, race_date, demo_mode, readiness


def render_elevation_profile(df):
    """Renders the course elevation profile."""
    fig = px.area(df, x="distance", y="elevation", title="Course Elevation (m)")
    fig.update_layout(showlegend=False)
    fig.update_traces(line_color="#FF4B4B", fillcolor="rgba(255, 75, 75, 0.1)")
    st.plotly_chart(fig, use_container_width=True)


def render_pacing_charts(pacing_data):
    """Renders the Pace and Battery charts."""
    if not pacing_data:
        return

    # 1. Pacing Chart
    plot_data = []
    for const in pacing_data:
        plot_data.append({"distance": const["start_dist"], "pace": const["pace_min_km"]})
        plot_data.append({"distance": const["end_dist"], "pace": const["pace_min_km"]})

    df_pace = pd.DataFrame(plot_data)

    fig_pace = px.line(df_pace, x="distance", y="pace", title="Projected Pace (min/km)")
    fig_pace.update_yaxes(autorange="reversed")
    fig_pace.update_layout(yaxis_title="Pace (min/km)")
    st.plotly_chart(fig_pace, use_container_width=True)

    # 2. Battery Chart
    battery_data = []
    for const in pacing_data:
        exhaustion = const.get("fatigue_level", 0.0)
        battery = (1.0 - exhaustion) * 100.0

        battery_data.append({"distance": const["start_dist"], "battery": battery})
        battery_data.append({"distance": const["end_dist"], "battery": battery})

    df_battery = pd.DataFrame(battery_data)
    fig_batt = px.area(
        df_battery, x="distance", y="battery", title="Physiological Battery (W' Balance)"
    )
    fig_batt.update_yaxes(range=[0, 100], title="Energy Reserves (%)")
    fig_batt.update_traces(line_color="#2ECC71", fillcolor="rgba(46, 204, 113, 0.2)")
    st.plotly_chart(fig_batt, use_container_width=True)


def render_race_mode(state_manager, segments):
    """
    Renders the Live Race interface.
    """
    st.markdown("## ⏱️ Live Race Companion")

    state = state_manager.get_state()
    current_idx = state.current_segment_index

    # Progress Bar
    progress = min(current_idx / len(segments), 1.0) if segments else 0
    st.progress(progress, text=f"Progress: {current_idx}/{len(segments)} Segments")

    # Display Segments Checklist
    st.subheader("Checkpoints")

    if not segments:
        st.info("No course loaded.")
        return

    # Just show the next few and the last few
    # Or scrollable container? Let's do a simple expander list

    for i, seg in enumerate(segments):
        is_completed = i < current_idx
        is_next = i == current_idx

        # Determine status icon
        if i in state.actual_splits:
            status = f"✅ Done ({state.actual_splits[i]:.0f} min)"
        elif is_next:
            status = "🏃 IN PROGRESS"
        else:
            status = "Upcoming"

        with st.expander(
            f"Segment {i + 1}: {seg.length / 1000:.1f}km ({seg.type.value}) - {status}",
            expanded=is_next,
        ):
            if is_completed:
                st.write(f"Completed at: {state.actual_splits.get(i)} mins from start")
            else:
                st.write(f"Distance: {seg.end_dist / 1000:.1f} km mark")

                # Check-in form
                with st.form(key=f"checkin_{i}"):
                    # Default: mock value based on length/avg pace?
                    # Simpler: User types minutes from start.
                    # V2: User types Time of Day (e.g. 14:30).
                    # For MVP V3 Phase 2: Input is "Minutes Elapsed".
                    arrival_input = st.number_input(
                        "Elapsed Minutes since Start",
                        min_value=0.0,
                        value=state.actual_splits.get(i - 1, 0.0) + 30.0,  # Guess
                        step=1.0,
                    )

                    if st.form_submit_button("📍 Check In Here"):
                        state_manager.update_checkpoint(i, arrival_input)
                        st.rerun()

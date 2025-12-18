import folium
import pandas as pd  # type: ignore
import pydeck as pdk  # type: ignore
import streamlit as st
from streamlit_folium import st_folium  # type: ignore


def render_3d_course(course_df: pd.DataFrame):
    """
    Renders the course in 3D using PyDeck.
    """
    if course_df.empty:
        st.warning("No data for 3D map.")
        return

    # Prepare data for PathLayer
    # We need a list of coordinates [lon, lat, ele]
    # Filter valid
    valid = course_df.dropna(subset=["latitude", "longitude", "elevation"])
    if valid.empty:
        st.warning("No valid coordinate data.")
        return

    path_data = [
        [row["longitude"], row["latitude"], row["elevation"]] for _, row in valid.iterrows()
    ]

    # Create a DataFrame for the layer (PyDeck likes dataframes or lists of dicts)
    layer_data = pd.DataFrame(
        {
            "path": [path_data],
            "name": ["Race Course"],
            "color": [[255, 0, 0]],  # Red
        }
    )

    # Calculate view state
    mid_idx = len(path_data) // 2
    # Start view
    start_lon, start_lat, _ = path_data[0]
    mid_lon, mid_lat, _ = path_data[mid_idx]

    view_state = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=11,
        pitch=50,  # 3D angle
        bearing=0,
    )

    # Define Layer
    layer = pdk.Layer(
        "PathLayer",
        data=layer_data,
        pickable=True,
        get_color="color",
        width_scale=20,
        width_min_pixels=2,
        get_path="path",
        get_width=5,
    )

    # Tooltip
    tooltip = {
        "html": "<b>Race Course</b>",
        "style": {
            "background": "grey",
            "color": "white",
            "font-family": "sans-serif",
            "z-index": "10000",
        },
    }

    # Render
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None,  # Usage of Mapbox styles requires a token. Defaulting to Carto (None) for "out of box" support.
    )

    st.pydeck_chart(r)


def render_course_map(course_df: pd.DataFrame, show_radar: bool = False):
    """
    Renders the course map using Folium with an optional Weather Radar overlay.
    """
    if course_df.empty:
        st.warning("No coordinates to map.")
        return

    # Filter invalid coordinates
    valid_points = course_df.dropna(subset=["latitude", "longitude"])
    if valid_points.empty:
        st.warning("GPX contains no valid latitude/longitude data.")
        return

    # Calculate center
    avg_lat = valid_points["latitude"].mean()
    avg_lon = valid_points["longitude"].mean()

    # Initialize Map
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11, tiles="OpenStreetMap")

    # 1. Plot Course Track
    points = list(zip(valid_points["latitude"], valid_points["longitude"]))
    folium.PolyLine(points, color="red", weight=4, opacity=0.8, tooltip="Course Track").add_to(m)

    # 2. Add Segments Markers (Start/End)
    # We ideally need 'segment_index' in course_df or pass segments list.
    # For now, just Start and Finish markers.
    folium.Marker(points[0], popup="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(points[-1], popup="Finish", icon=folium.Icon(color="black", icon="flag")).add_to(
        m
    )

    # 3. Weather Radar Layer
    if show_radar:
        # RainViewer Tile Layer
        # URL format: https://tile.rainviewer.com{/ts}/{size}/{z}/{x}/{y}/{color}/{options}.png
        # We need the latest timestamp. For MVP we use 'now' or fetch it?
        # Standard generic "now" doesn't work well with XYZ tiles often, better to fetch valid TS.
        # But RainViewer documentation says we can iterate timestamps.
        # Simplest free approach: Use a specific tile server or just embed one consistent layer.

        # Let's try to fetch the latest available timestamp from their API for correctness.
        # If that fails, we skip or warn.
        import requests

        try:
            # Get maps metadata
            rv_resp = requests.get("https://api.rainviewer.com/public/weather-maps.json")
            if rv_resp.status_code == 200:
                data = rv_resp.json()

                # Handle V2 API format
                if "radar" in data and "past" in data["radar"]:
                    past_data = data["radar"]["past"]
                    if past_data:
                        latest_entry = past_data[-1]
                        latest_ts = latest_entry["time"]

                        # Add Radar Layer
                        # V2 URL scheme might be different?
                        # Docs say: https://tile.rainviewer.com{/ts}/{size}/{z}/{x}/{y}/{color}/{options}.png
                        tile_url = f"https://tile.rainviewer.com/{latest_ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png"

                        folium.TileLayer(
                            tiles=tile_url,
                            attr="RainViewer",
                            name="Weather Radar",
                            overlay=True,
                            control=True,
                            opacity=0.6,
                            show=True,
                        ).add_to(m)

                        # Debug info for user
                        from datetime import datetime

                        dt = datetime.fromtimestamp(latest_ts)
                        st.toast(f"📡 Radar Loaded: {dt.strftime('%H:%M')}", icon="🌧️")
                        st.caption(f"Radar Data: RainViewer ({dt.strftime('%H:%M')})")
                    else:
                        st.warning("No radar data available.")
                else:
                    st.warning("Unexpected RainViewer API format.")
        except Exception as e:
            st.warning(f"Could not load Weather Radar: {e}")

    # Render
    st_folium(m, width=None, height=500)

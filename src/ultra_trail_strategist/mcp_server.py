import logging
import asyncio
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

from ultra_trail_strategist.data_ingestion.strava_client import StravaClient

# Initialize FastMCP Server
mcp = FastMCP("Strava MCP Server")

# Instantiate functionality
strava = StravaClient()
logger = logging.getLogger("strava_mcp")

@mcp.tool()
async def get_recent_activities(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch the athlete's most recent activities from Strava.
    Useful for understanding current fitness and training volume.
    
    Args:
        limit: Number of activities to retrieve (default 5).
    """
    activities = []
    # StravaClient is synchronous, so we treat it as such or wrap in executor if needed for high load.
    # For now, direct call is fine as FastMCP handles async wrapper.
    iterator = strava.get_athlete_activities(limit=limit)
    for act in iterator:
        # Simplify output for LLM context window efficiency
        activities.append({
            "id": act.get("id"),
            "name": act.get("name"),
            "distance_km": act.get("distance", 0) / 1000,
            "moving_time_min": act.get("moving_time", 0) / 60,
            "total_elevation_gain": act.get("total_elevation_gain", 0),
            "start_date": act.get("start_date_local"),
            "average_heartrate": act.get("average_heartrate", "N/A")
        })
    return activities

@mcp.tool()
async def get_activity_analysis(activity_id: int) -> str:
    """
    Analyze a specific past activity to extract performance metrics 
    like GAP details or heart rate drift. 
    (Currently returns raw stream summary).
    """
    # This is a placeholder for more complex analysis using get_activity_stream
    # We would fetch streams and compute GAP/HR ratio.
    try:
        streams = strava.get_activity_stream(activity_id)
        # Summarize just to show connection works
        return f"Fetched streams for {activity_id}: Found keys {list(streams.keys())}"
    except Exception as e:
        return f"Error fetching activity {activity_id}: {str(e)}"

@mcp.tool()
async def get_activity_streams(activity_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Fetch detailed streams (telemetry) for a list of activities.
    Used for Machine Learning model training.
    """
    streams = []
    # Strava API rate limits apply. Sequential fetching to be safe.
    for aid in activity_ids:
        try:
            # We skip errors to ensure we return as much data as possible
            s = strava.get_activity_stream(aid)
            streams.append(s)
        except Exception as e:
            logger.error(f"Failed to fetch stream for {aid}: {e}")
            continue
    return streams

# Entry point for running the server directly
if __name__ == "__main__":
    mcp.run()

# Ultra-Trail Strategist 🏃‍♂️⛰️

An Agentic AI application that generates professional-grade, segment-by-segment race strategies for ultra-distance runners. It uniquely combines geospatial course analysis with physiological athlete history to provide tactical advice.

## 🌟 Key Features

*   **Smart Course Segmentation**: Automatically breaks down GPX files into tactical segments: **Climbs** (>3%), **Descents** (<-3%), and **Flats**.
*   **Physiological Context**: Connects to **Strava** to analyze your recent training volume and fitness status.
*   **Agentic Reasoning**: Uses a **LangGraph**-based AI agent to synthesize data and act as a "Principal Coach."
*   **MCP Server**: Implements the **Model Context Protocol (MCP)** to expose Strava data as standard tools for the AI agent.

## 🏗️ Architecture

1.  **Data Ingestion**:
    *   `GPXProcessor`: Polars-based parsing and Savitzky-Golay elevation smoothing.
    *   `StravaClient`: OAuth2-handled API client with pagination.
2.  **Feature Engineering**:
    *   `CourseSegmenter`: Logic for detecting topographical features.
3.  **Agent Logic**:
    *   `StrategistAgent`: Orchestrates the workflow (Course Analysis -> Tool Usage -> Strategy Generation).

## 🚀 Getting Started

### Prerequisites

*   Python 3.10+
*   A Strava Account (with API Application created)
*   OpenAI API Key

### Installation

1.  **Clone and Install**:
    ```bash
    git clone https://github.com/yourusername/ultra-trail-strategist.git
    cd ultra-trail-strategist
    pip install .
    ```

2.  **Configuration**:
    Create a `.env` file in the root directory (copy `.env.example`):
    ```ini
    STRAVA_CLIENT_ID=your_id
    STRAVA_CLIENT_SECRET=your_secret
    STRAVA_REFRESH_TOKEN=your_refresh_token
    OPENAI_API_KEY=sk-...
    ```
    *Note: Ensure your Strava Refresh Token has `activity:read_all` scope.*

### Usage

Run the main application entry point:

```bash
python main.py
```

1.  It will prompt you for a **GPX file path**.
2.  (Optional) Press **Enter** to run in **Demo Mode** with synthetic data.
3.  The AI will analyze the course, fetch your recent training history, and print a detailed **Race Strategy**.

### Running Tests

To verify the installation and logic:

```bash
python -m unittest discover tests
```

## 🛠️ MCP Server

To run the standalone Strava MCP Server (for use with Claude Desktop or other MCP clients):

```bash
python src/ultra_trail_strategist/mcp_server.py
```

## 📄 License

MIT

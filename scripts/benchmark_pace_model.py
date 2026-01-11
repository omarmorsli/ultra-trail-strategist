#!/usr/bin/env python
"""
Benchmark Pace Model Script.

Compares different pace prediction approaches on a GPX course:
1. Fallback (Naismith's Rule) - analytical baseline
2. Base Model (Endomondo pre-trained) - cold-start prediction
3. Hybrid Model (with personal fine-tuning) - full hybrid predictor

Usage:
    pdm run python scripts/benchmark_pace_model.py --gpx assets/demo.gpx
    pdm run python scripts/benchmark_pace_model.py --gpx path/to/course.gpx --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultra_trail_strategist.data_ingestion.gpx_processor import GPXProcessor
from ultra_trail_strategist.feature_engineering.pace_model import HybridPacePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_grade(elevations: list[float], distances: list[float]) -> list[float]:
    """Calculate grade percentages between points."""
    grades = []
    for i in range(1, min(len(elevations), len(distances))):
        d_elev = elevations[i] - elevations[i - 1]
        d_dist = distances[i] - distances[i - 1]
        if d_dist > 0:
            grade = (d_elev / d_dist) * 100
            # Clip to reasonable range
            grade = max(-50, min(50, grade))
        else:
            grade = 0.0
        grades.append(grade)
    # Prepend 0 for first point
    return [0.0] + grades


def benchmark_models(
    gpx_path: Path,
    verbose: bool = False,
) -> dict:
    """
    Run benchmark comparison of pace models on a GPX course.

    Parameters
    ----------
    gpx_path : Path
        Path to GPX file.
    verbose : bool
        Print detailed output.

    Returns
    -------
    dict
        Benchmark results including times and statistics.
    """
    logger.info(f"Loading GPX: {gpx_path}")

    # Load GPX data
    processor = GPXProcessor(str(gpx_path))
    processor.load_from_file()
    df = processor.to_dataframe()
    
    if df.is_empty():
        logger.error("No points found in GPX file")
        return {}

    logger.info(f"Loaded {len(df)} GPS points")

    # Extract data from DataFrame
    distances = df["distance"].to_list()
    elevations = df["elevation"].to_list()
    
    # Smooth elevation
    processor.smooth_elevation()
    if processor._df is not None and "elevation_smoothed" in processor._df.columns:
        elevations = processor._df["elevation_smoothed"].to_list()

    # Calculate grades
    grades = calculate_grade(elevations, distances)

    total_distance_km = distances[-1] / 1000 if distances else 0
    total_elevation_gain = sum(
        max(0, elevations[i] - elevations[i - 1])
        for i in range(1, len(elevations))
    )
    total_elevation_loss = sum(
        abs(min(0, elevations[i] - elevations[i - 1]))
        for i in range(1, len(elevations))
    )

    logger.info(f"Course: {total_distance_km:.2f} km")
    logger.info(f"Elevation: +{total_elevation_gain:.0f}m / -{total_elevation_loss:.0f}m")

    # Create simple fixed-distance segments (500m each)
    segment_length_m = 500
    segments = []
    start_idx = 0
    
    for i, dist in enumerate(distances):
        if (dist - distances[start_idx]) >= segment_length_m or i == len(distances) - 1:
            # Calculate average grade for this segment
            segment_grades = (
                grades[start_idx:i+1] if i > start_idx else [grades[start_idx]]
            )
            segment_elevations = (
                elevations[start_idx:i+1] if i > start_idx else [elevations[start_idx]]
            )
            
            avg_grade = (
                sum(segment_grades) / len(segment_grades) if segment_grades else 0.0
            )
            avg_elevation = (
                sum(segment_elevations) / len(segment_elevations)
                if segment_elevations else 0.0
            )
            
            segments.append({
                "start_distance": distances[start_idx],
                "end_distance": dist,
                "grade": avg_grade,
                "avg_elevation": avg_elevation,
                "length": dist - distances[start_idx],
            })
            start_idx = i

    logger.info(f"Created {len(segments)} segments")

    # Initialize predictor
    predictor = HybridPacePredictor()

    # Model comparison results
    results = {
        "course": {
            "name": gpx_path.stem,
            "distance_km": total_distance_km,
            "elevation_gain_m": total_elevation_gain,
            "elevation_loss_m": total_elevation_loss,
            "num_segments": len(segments),
        },
        "models": {},
    }

    # === Benchmark Fallback Model ===
    logger.info("\n--- Fallback Model (Naismith's Rule) ---")
    fallback_times = []
    for seg in segments:
        # Use private method for fallback
        velocity = predictor._fallback_prediction(seg["grade"])
        segment_length = seg["end_distance"] - seg["start_distance"]
        time_seconds = segment_length / velocity if velocity > 0 else 0
        fallback_times.append(time_seconds)

    fallback_total = sum(fallback_times)
    results["models"]["fallback"] = {
        "name": "Naismith's Rule",
        "total_time_seconds": fallback_total,
        "total_time_formatted": format_time(fallback_total),
        "avg_pace_min_km": (fallback_total / 60) / total_distance_km,
    }
    logger.info(f"Total time: {format_time(fallback_total)}")
    logger.info(f"Avg pace: {results['models']['fallback']['avg_pace_min_km']:.1f} min/km")

    # === Benchmark Base Model ===
    logger.info("\n--- Base Model (Endomondo Pre-trained) ---")
    base_times = []
    for seg in segments:
        velocity = predictor.predict(
            grade=seg["grade"],
            altitude=seg.get("avg_elevation", 0.0),
        )
        segment_length = seg["end_distance"] - seg["start_distance"]
        time_seconds = segment_length / velocity if velocity > 0 else 0
        base_times.append(time_seconds)

    base_total = sum(base_times)
    results["models"]["base"] = {
        "name": "Endomondo Base Model",
        "total_time_seconds": base_total,
        "total_time_formatted": format_time(base_total),
        "avg_pace_min_km": (base_total / 60) / total_distance_km,
        "has_trained_model": predictor.base_model is not None,
    }
    logger.info(f"Total time: {format_time(base_total)}")
    logger.info(f"Avg pace: {results['models']['base']['avg_pace_min_km']:.1f} min/km")

    # === Benchmark Hybrid with Ultra Fatigue ===
    logger.info("\n--- Hybrid Model (with Ultra Fatigue Adjustment) ---")
    hybrid_times = []
    cumulative_km = 0.0

    for seg in segments:
        segment_length = seg["end_distance"] - seg["start_distance"]
        segment_km = segment_length / 1000

        velocity = predictor.predict(
            grade=seg["grade"],
            altitude=seg.get("avg_elevation", 0.0),
            distance_into_race=cumulative_km,
            total_race_distance=total_distance_km,
            hour_of_day=10,  # Daytime assumption
        )

        time_seconds = segment_length / velocity if velocity > 0 else 0
        hybrid_times.append(time_seconds)
        cumulative_km += segment_km

    hybrid_total = sum(hybrid_times)
    results["models"]["hybrid"] = {
        "name": "Hybrid (Fatigue Adjusted)",
        "total_time_seconds": hybrid_total,
        "total_time_formatted": format_time(hybrid_total),
        "avg_pace_min_km": (hybrid_total / 60) / total_distance_km,
    }
    logger.info(f"Total time: {format_time(hybrid_total)}")
    logger.info(f"Avg pace: {results['models']['hybrid']['avg_pace_min_km']:.1f} min/km")

    # === Summary ===
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Course: {gpx_path.stem}")
    print(f"Distance: {total_distance_km:.2f} km | D+: {total_elevation_gain:.0f}m")
    print("-" * 60)
    print(f"{'Model':<30} {'Time':>12} {'Pace':>10}")
    print("-" * 60)

    for _model_key, model_data in results["models"].items():
        print(
            f"{model_data['name']:<30} "
            f"{model_data['total_time_formatted']:>12} "
            f"{model_data['avg_pace_min_km']:.1f} min/km"
        )

    print("=" * 60)

    # Calculate relative differences
    if fallback_total > 0:
        base_diff = ((base_total - fallback_total) / fallback_total) * 100
        hybrid_diff = ((hybrid_total - fallback_total) / fallback_total) * 100
        print(f"\nBase vs Fallback: {base_diff:+.1f}%")
        print(f"Hybrid vs Fallback: {hybrid_diff:+.1f}%")

    return results


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pace prediction models on a GPX course"
    )
    parser.add_argument(
        "--gpx",
        type=Path,
        default=Path("assets/demo.gpx"),
        help="Path to GPX file (default: assets/demo.gpx)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if not args.gpx.exists():
        logger.error(f"GPX file not found: {args.gpx}")
        sys.exit(1)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results = benchmark_models(args.gpx, args.verbose)

    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()

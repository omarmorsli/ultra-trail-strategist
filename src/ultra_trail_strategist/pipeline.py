import logging
import polars as pl
from typing import List
from ultra_trail_strategist.data_ingestion.gpx_processor import GPXProcessor
from ultra_trail_strategist.feature_engineering.segmenter import CourseSegmenter, Segment

logger = logging.getLogger(__name__)

class RaceDataPipeline:
    """
    Orchestrates the ingestion and processing of race data.
    """
    
    def __init__(self, gpx_file_path: str):
        self.gpx_file_path = gpx_file_path
        self.processor = GPXProcessor(file_path=gpx_file_path)
        self.segments: List[Segment] = []
        self.df: pl.DataFrame = pl.DataFrame()

    def run(self) -> List[Segment]:
        """
        Executes the pipeline: Load GPX -> Smooth -> Segment.
        """
        logger.info(f"Starting pipeline for {self.gpx_file_path}")
        
        # 1. Load and Process GPX
        self.processor.load_from_file()
        self.processor.to_dataframe()
        self.df = self.processor.smooth_elevation()
        
        # 2. Segmentation
        segmenter = CourseSegmenter(self.df)
        self.segments = segmenter.process()
        
        logger.info(f"Pipeline complete. Generated {len(self.segments)} segments.")
        return self.segments

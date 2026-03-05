"""
core/pipeline/__init__.py
==========================
Public exports for the pipeline package.
"""

from core.pipeline.video_pipeline  import VideoPipeline, PipelineConfig, PipelineFrame
from core.pipeline.frame_producer  import FrameProducer
from core.pipeline.frame_consumer  import FrameConsumer

__all__ = [
    "VideoPipeline",
    "PipelineConfig",
    "PipelineFrame",
    "FrameProducer",
    "FrameConsumer",
]

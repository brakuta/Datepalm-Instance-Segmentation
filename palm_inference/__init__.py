"""Country-scale date palm inference pipeline.

Public API:
    from palm_inference import InferenceConfig, run_pipeline, merge_and_dedup
"""
from .config import InferenceConfig
from .pipeline import run_pipeline
from .postprocess import merge_and_dedup, merge_outputs, dedup_overlaps, emit_density_raster

__all__ = [
    "InferenceConfig",
    "run_pipeline",
    "merge_and_dedup",
    "merge_outputs",
    "dedup_overlaps",
    "emit_density_raster",
]

# nlpv2/pipeline/__init__.py
# All models and data are loaded at module startup inside their respective files.
# Import run_pipeline as the public API entry point.

from .pipeline import run_pipeline, confidence_level

__all__ = ["run_pipeline", "confidence_level"]

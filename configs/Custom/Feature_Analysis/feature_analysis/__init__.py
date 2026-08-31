# -*- coding: utf-8 -*-
"""Feature-analysis suite for the date palm multi-resolution benchmark.

Computes representation-level evidence for cross-resolution generalisation
across CNN, Transformer, and state-space (Mamba) backbones, and renders
publication-grade figures. See README.md for the scientific design and usage.
"""

__version__ = "2.0.0"

from .config import AnalysisConfig, make_config  # noqa: F401

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launcher for the feature-analysis suite.

Resolves the local package directory and dispatches to the CLI, so the suite
can be invoked from any working directory without installation, e.g.:

    python configs/Custom/Feature_Analysis/run_feature_analysis.py \
        run --config configs/Custom/Feature_Analysis/config_feature_analysis.json
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from feature_analysis.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

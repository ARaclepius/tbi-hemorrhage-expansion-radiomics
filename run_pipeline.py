#!/usr/bin/env python3
"""Run the CI and interpretable-ML TBI hemorrhage-expansion workflow.

See scripts/run_ci_interpretable_pipeline.py --help for all options.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "run_ci_interpretable_pipeline.py"
    runpy.run_path(str(script), run_name="__main__")

"""
Universal experiment runner that loads and executes experiments from TOML configs.

Usage:
    python run_experiment.py [config_path]

If no config_path is provided, defaults to workspace/experiment.toml.

Examples:
    python run_experiment.py                          # Uses workspace/experiment.toml
    python run_experiment.py workspace/experiment.toml
    python run_experiment.py configs/baseline.toml
"""

import sys
from pathlib import Path

# Add src to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils.runners import run_experiment


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "workspace/experiment.toml"
    run_experiment(config_path)

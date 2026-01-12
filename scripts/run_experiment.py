"""
Universal experiment runner that loads and executes experiments from TOML configs.

Usage:
    python scripts/run_experiment.py [config_path] [--no-commit]

If no config_path is provided, defaults to workspace/experiment.toml.
Use --no-commit to skip git status checks (useful for development/debugging).

Examples:
    python scripts/run_experiment.py                          # Uses workspace/experiment.toml
    python scripts/run_experiment.py workspace/experiment.toml
    python scripts/run_experiment.py configs/baseline.toml
    python scripts/run_experiment.py --no-commit              # Skip git checks with default config
    python scripts/run_experiment.py config.toml --no-commit  # Skip git checks with specified config
"""

import sys
from pathlib import Path

# Add src to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.experiment_runners import run_experiment


if __name__ == "__main__":
    # Parse arguments
    config_path = "workspace/experiment.toml"  # default
    skip_git_check = False

    for arg in sys.argv[1:]:
        if arg == "--no-commit":
            skip_git_check = True
        elif not arg.startswith("--"):
            config_path = arg

    run_experiment(config_path, skip_git_check=skip_git_check)

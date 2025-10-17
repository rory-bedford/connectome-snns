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
import importlib.util
from pathlib import Path

# Add src to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils.reproducibility import ExperimentTracker


def run_experiment(config_path=None):
    """
    Load and run an experiment from a TOML config file.

    Args:
        config_path: Path to experiment.toml file. Defaults to workspace/experiment.toml.
    """
    # Initialize tracker (validates config)
    tracker = ExperimentTracker(config_path=config_path)

    # Get the script to run from config
    script_path = Path(tracker.config["script"])
    if not script_path.is_absolute():
        # Make relative to repo root
        repo_root = Path(__file__).resolve().parent
        script_path = (repo_root / script_path).resolve()

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        sys.exit(1)

    print(f"Loading experiment script: {script_path}")

    # Load the script as a module
    spec = importlib.util.spec_from_file_location("experiment_script", script_path)
    module = importlib.util.module_from_spec(spec)

    # Run within tracker context
    with tracker:
        # Execute the script's main() function
        spec.loader.exec_module(module)
        if hasattr(module, "main"):
            # Pass output_dir and params_csv directly to the script
            module.main(output_dir=tracker.output_dir, params_csv=tracker.params_csv)
        else:
            print(f"ERROR: Script {script_path} has no main() function")
            sys.exit(1)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_experiment(config_path)

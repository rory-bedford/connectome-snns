"""
Simple parallel grid search with GPU assignment and custom config modification.

Run:
    ./run --grid path/to/your/experiment.toml

Edit CUDA_VISIBLE_DEVICES below to match your available GPUs.
"""

import sys
from copy import deepcopy
from pathlib import Path


# Add src to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.experiment_runners import run_custom_search

CUDA_VISIBLE_DEVICES = [0]  # Edit with available GPU IDs


def custom_config_generator(base_params):
    """
    Define your custom grid search here.

    Args:
        base_params: Loaded parameters TOML as dict

    Yields:
        (params_dict, description_string) tuples
    """
    # Altering noise level from 0.05 to 0.4
    for seed in range(42, 52):
        params = deepcopy(base_params)
        params["simulation"]["seed"] = float(seed)
        yield params, f"seed-{seed:.2f}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "workspace/experiment.toml"
    results = run_custom_search(
        experiment_config_path=config_path,
        config_generator=custom_config_generator,
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )

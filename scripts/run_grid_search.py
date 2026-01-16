"""
Simple parallel grid search with GPU assignment and custom config modification.

Setup:
    1. Copy this file to workspace/run_grid_search.py
    2. Edit the custom_config_generator function to define your parameter sweep
    3. Edit CUDA_VISIBLE_DEVICES to match your available GPUs

Run:
    ./run --grid path/to/your/experiment.toml

The ./run script automatically uses workspace/run_grid_search.py if it exists,
otherwise falls back to this template in scripts/.
"""

import sys
from copy import deepcopy
from pathlib import Path

# Add src to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.experiment_runners import run_custom_search

CUDA_VISIBLE_DEVICES = [0, 1]  # Edit with available GPU IDs


def custom_config_generator(base_params):
    """
    Define your custom grid search here.

    Args:
        base_params: Loaded parameters TOML as dict

    Yields:
        (params_dict, description_string) tuples
    """
    # Example 1: Simple grid
    for dt in [0.5, 1.0, 2.0]:
        for num_neurons in [5000, 10000]:
            params = deepcopy(base_params)
            params["simulation"]["dt"] = dt
            params["connectome"]["topology"]["num_neurons"] = num_neurons

            yield params, f"dt={dt}_neurons={num_neurons}"

    # Example 2: Complex modifications
    for scale in [0.001, 0.002, 0.004]:
        for assemblies in [10, 20, 40]:
            params = deepcopy(base_params)
            params["optimisation"]["scaling_factors"][0][0] = scale
            params["connectome"]["topology"]["num_assemblies"] = assemblies

            # Conditional logic
            if scale > 0.002:
                params["hyperparameters"]["surrgrad_scale"] = 200.0

            yield params, f"scale={scale}_asm={assemblies}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "workspace/experiment.toml"
    results = run_custom_search(
        experiment_config_path=config_path,
        config_generator=custom_config_generator,
        cuda_devices=CUDA_VISIBLE_DEVICES,
    )

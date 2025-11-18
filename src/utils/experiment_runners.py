import toml
import subprocess
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys
from datetime import datetime
import importlib.util
import shutil
from utils.reproducibility import ExperimentTracker
from tqdm import tqdm  # Add tqdm for progress bar


def get_repo_root():
    """Get the repository root directory."""
    # Navigate up from this file (src/utils/experiment_runners.py) to repo root
    return Path(__file__).resolve().parent.parent.parent


def resolve_relative_to_repo(path_str):
    """
    Resolve a path relative to the repository root if it's not absolute.

    Args:
        path_str: String path that may be relative or absolute

    Returns:
        Path object resolved to absolute path
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    else:
        return (get_repo_root() / path).resolve()


def run_experiment(config_path=None, skip_git_check=False):
    """
    Load and run an experiment from a TOML config file.

    Args:
        config_path: Path to experiment.toml file. Defaults to workspace/experiment.toml.
        skip_git_check: If True, skip git status validation (useful for development).
    """
    # Initialize tracker (validates config)
    tracker = ExperimentTracker(config_path=config_path, skip_git_check=skip_git_check)

    # Get the script to run from config
    script_path = resolve_relative_to_repo(tracker.config["script"])

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
            # Pass input_dir, output_dir and params_file, and wandb_config only if enabled
            kwargs = {
                "input_dir": tracker.input_dir,
                "output_dir": tracker.output_dir,
                "params_file": tracker.params_file,
            }
            if tracker.wandb_config and tracker.wandb_config.get("enabled", False):
                # Remove 'enabled' key before passing to script
                wandb_config = {
                    k: v for k, v in tracker.wandb_config.items() if k != "enabled"
                }
                kwargs["wandb_config"] = wandb_config

            module.main(**kwargs)
        else:
            print(f"ERROR: Script {script_path} has no main() function")
            sys.exit(1)


def run_on_gpu(args):
    """Run experiment on specific GPU."""
    experiment_config_path, gpu_id, description = args
    print(f"Running: {description} on GPU {gpu_id}")
    env = {**subprocess.os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    result = subprocess.run(
        ["python", "run_experiment.py", str(experiment_config_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    return {
        "config": experiment_config_path.stem,
        "gpu": gpu_id,
        "description": description,
        "success": result.returncode == 0,
        "output": result.stdout if result.returncode == 0 else result.stderr,
    }


def run_custom_search(experiment_config_path, config_generator, cuda_devices):
    """
    Run custom grid search in parallel across GPUs.

    Args:
        experiment_config_path: Path to experiment.toml
        config_generator: Generator function(base_params) that yields (params_dict, description)
        cuda_devices: List of GPU IDs (e.g., [0, 1])
    """
    # Load experiment config and base parameters
    experiment_config = toml.load(experiment_config_path)
    base_params = toml.load(experiment_config["parameters_file"])

    # Use the parent folder name directly from the config file
    parent_output = Path(experiment_config["output_dir"])
    grid_parent = parent_output.parent / parent_output.name
    grid_parent.mkdir(parents=True, exist_ok=True)

    # Copy the run_grid_search.py script to the target directory
    repo_root = get_repo_root()
    grid_search_script = repo_root / "run_grid_search.py"
    if grid_search_script.exists():
        shutil.copy2(grid_search_script, grid_parent / "run_grid_search.py")
        print(f"✓ Copied {grid_search_script.name} to {grid_parent}")
    else:
        print(f"⚠ Warning: {grid_search_script} not found, skipping copy")

    print(f"\nGrid search parent: {grid_parent}")

    # Count total simulations first
    config_list = list(config_generator(base_params))
    total_simulations = len(config_list)

    print(f"\n{'=' * 60}")
    print(f"GRID SEARCH: {total_simulations} simulations will be run")
    print(f"GPUs: {cuda_devices} ({len(cuda_devices)} workers)")
    print(f"{'=' * 60}\n")

    print("Generating configurations...")

    # Generate all experiment configs
    workspace = Path("workspace/temp_grid_configs")
    workspace.mkdir(parents=True, exist_ok=True)

    experiment_configs = []
    all_runs = []

    for i, (params, description) in enumerate(config_list):
        # Save modified parameters file
        params_file = workspace / f"params_{i:03d}.toml"
        with open(params_file, "w") as f:
            toml.dump(params, f)

        # Create experiment config
        exp_config = deepcopy(experiment_config)
        exp_config["parameters_file"] = str(params_file)
        exp_config["output_dir"] = str(grid_parent / description)

        # Save experiment config
        exp_config_file = workspace / f"experiment_{i:03d}.toml"
        with open(exp_config_file, "w") as f:
            toml.dump(exp_config, f)

        experiment_configs.append(exp_config_file)
        all_runs.append(
            {
                "run_id": i,
                "description": description,
                "output_dir": str(grid_parent / description),
                "parameters": params,
                "success": None,  # Initialize success status as None
            }
        )

    print(f"✓ Generated {len(experiment_configs)} configurations\n")

    # Write initial README and parameters file
    def write_readme_and_params():
        success_count = sum(1 for run in all_runs if run["success"] is True)
        failure_count = sum(1 for run in all_runs if run["success"] is False)
        readme = f"""# Grid Search Run

**Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Base Experiment**: {experiment_config_path}
**Base Parameters**: {experiment_config["parameters_file"]}

## Summary
- **Total Runs**: {len(all_runs)}
- **Successful**: {success_count}
- **Failed**: {failure_count}
- **Pending**: {len(all_runs) - success_count - failure_count}

## Runs

"""
        for run in all_runs:
            status = (
                "✓"
                if run["success"] is True
                else "✗"
                if run["success"] is False
                else "..."
            )
            readme += f"- {status} `{run['description']}/`\n"

        readme += "\n## Details\nSee `grid_parameters.toml` for complete parameter configuration of each run.\n"

        readme_path = grid_parent / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme)

        params_path = grid_parent / "grid_parameters.toml"
        with open(params_path, "w") as f:
            toml.dump({"runs": all_runs}, f)

    write_readme_and_params()  # Write initial state

    # Assign GPUs round-robin and run in parallel
    jobs = [
        (cfg, cuda_devices[i % len(cuda_devices)], all_runs[i]["description"])
        for i, cfg in enumerate(experiment_configs)
    ]

    print("Running experiments...\n")
    with ProcessPoolExecutor(max_workers=len(cuda_devices)) as executor:
        for i, result in enumerate(
            tqdm(
                executor.map(run_on_gpu, jobs),
                total=len(jobs),
                desc="Experiments Progress",
            )
        ):
            all_runs[i]["success"] = result["success"]  # Update success status
            write_readme_and_params()  # Update README and parameters file after each job

    print(
        f"\nComplete: {sum(r['success'] for r in all_runs if r['success'] is not None)}/{len(all_runs)} successful"
    )
    print(f"Results: {grid_parent}\n")

    return all_runs

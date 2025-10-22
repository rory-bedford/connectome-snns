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
            # Pass output_dir and params_file directly to the script
            module.main(output_dir=tracker.output_dir, params_file=tracker.params_file)
        else:
            print(f"ERROR: Script {script_path} has no main() function")
            sys.exit(1)


def run_on_gpu(args):
    """Run experiment on specific GPU."""
    experiment_config_path, gpu_id = args
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

    print(f"\nGrid search parent: {grid_parent}")
    print("Generating configurations...\n")

    # Generate all experiment configs
    workspace = Path("workspace/temp_grid_configs")
    workspace.mkdir(parents=True, exist_ok=True)

    experiment_configs = []
    all_runs = []

    for i, (params, description) in enumerate(config_generator(base_params)):
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
            }
        )

    print(f"✓ Generated {len(experiment_configs)} configurations")
    print(f"  GPUs: {cuda_devices}")
    print(f"  Workers: {len(cuda_devices)}\n")

    # Assign GPUs round-robin and run in parallel
    jobs = [
        (cfg, cuda_devices[i % len(cuda_devices)])
        for i, cfg in enumerate(experiment_configs)
    ]

    print("Running experiments...\n")
    with ProcessPoolExecutor(max_workers=len(cuda_devices)) as executor:
        results = list(executor.map(run_on_gpu, jobs))

    # Print results
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"{status} GPU{r['gpu']}: {r['config']}")

    # Update runs with success status
    for i, run in enumerate(all_runs):
        run["success"] = results[i]["success"]

    # Save complete grid parameters
    grid_params = {"runs": all_runs}
    params_path = grid_parent / "grid_parameters.toml"
    with open(params_path, "w") as f:
        toml.dump(grid_params, f)

    # Create README
    success_count = sum(r["success"] for r in results)
    readme = f"""# Grid Search Run

**Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Base Experiment**: {experiment_config_path}
**Base Parameters**: {experiment_config["parameters_file"]}

## Summary
- **Total Runs**: {len(results)}
- **Successful**: {success_count}
- **Failed**: {len(results) - success_count}

## Runs

"""
    for run in all_runs:
        status = "✓" if run["success"] else "✗"
        readme += f"- {status} `{run['description']}/`\n"

    readme += "\n## Details\nSee `grid_parameters.toml` for complete parameter configuration of each run.\n"

    readme_path = grid_parent / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)

    print(f"\nComplete: {success_count}/{len(results)} successful")
    print(f"Results: {grid_parent}\n")

    # Cleanup temp configs
    shutil.rmtree(workspace)

    return results

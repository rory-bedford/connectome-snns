#!/usr/bin/env python
"""
Resume student network training from a checkpoint.

This script resumes training of a student network from a previously saved
checkpoint. It loads the most recent checkpoint and continues the training
process from where it left off.

NOTE: Since experiment tracking has already been initialized, this script
should be called directly, outside of the usual experiment runner.

Usage:
    python resume_training.py <output_dir> [--no-wandb]

Example:
    python resume_training.py workspace/my_experiment_2025-10-30_10-30-45
    python resume_training.py workspace/my_experiment --no-wandb
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import toml

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.fitting_activity.train_student import main


def resume_training(output_dir, disable_wandb=False):
    """Resume training from checkpoint in output directory.

    Args:
        output_dir (Path): Directory containing checkpoints and parameters
        disable_wandb (bool): Override to disable wandb even if enabled in config
    """
    output_dir = Path(output_dir)

    # Find checkpoint
    checkpoint_path = output_dir / "checkpoints" / "checkpoint_latest.pt"
    if not checkpoint_path.exists():
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        print(f"Available files in {output_dir / 'checkpoints'}:")
        if (output_dir / "checkpoints").exists():
            for f in (output_dir / "checkpoints").iterdir():
                print(f"  - {f.name}")
        sys.exit(1)

    # Find parameters file
    params_file = output_dir / "parameters.toml"
    if not params_file.exists():
        print(f"ERROR: No parameters file found at {params_file}")
        sys.exit(1)

    # Load experiment config from the CHECKPOINT DIRECTORY (not workspace)
    experiment_config_file = output_dir / "experiment.toml"
    wandb_config = None
    if experiment_config_file.exists():
        experiment_config = toml.load(experiment_config_file)
        wandb_config = experiment_config.get("wandb", {})

        # Apply override to disable if requested
        if disable_wandb and wandb_config:
            wandb_config = wandb_config.copy()
            wandb_config["enabled"] = False

    # Create a timestamped subdirectory for resumed training plots
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resumed_output_dir = output_dir / f"resumed_{timestamp}"
    resumed_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resuming training from: {output_dir}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using parameters: {params_file}")
    print(f"Using experiment config: {experiment_config_file}")
    if wandb_config:
        print(f"Wandb enabled: {wandb_config.get('enabled', False)}")
        if wandb_config.get("enabled", False):
            print(f"Wandb project: {wandb_config.get('project', 'N/A')}")
    print(f"Plots will be saved to: {resumed_output_dir}")
    print()

    # Check if inputs directory exists in output_dir
    input_dir = output_dir / "inputs" if (output_dir / "inputs").exists() else None

    # Resume training with new output directory for plots
    main(
        input_dir=input_dir,
        output_dir=output_dir,
        params_file=params_file,
        wandb_config=wandb_config,
        resume_from=checkpoint_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resume homeostatic plasticity training from checkpoint"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory containing checkpoints and parameters",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging (overrides experiment.toml setting)",
    )

    args = parser.parse_args()

    resume_training(
        output_dir=args.output_dir,
        disable_wandb=args.no_wandb,
    )

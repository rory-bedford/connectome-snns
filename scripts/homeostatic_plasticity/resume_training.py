#!/usr/bin/env python
"""
Resume training from a checkpoint.

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

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from homeostatic_plasticity import main


def resume_training(output_dir, use_wandb=True):
    """Resume training from checkpoint in output directory.

    Args:
        output_dir (Path): Directory containing checkpoints and parameters
        use_wandb (bool): Whether to use wandb logging
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

    # Create a timestamped subdirectory for resumed training plots
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resumed_output_dir = output_dir / f"resumed_{timestamp}"
    resumed_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resuming training from: {output_dir}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using parameters: {params_file}")
    print(f"Wandb enabled: {use_wandb}")
    print(f"Plots will be saved to: {resumed_output_dir}")
    print()

    # Resume training with new output directory for plots
    main(
        output_dir=output_dir,
        params_file=params_file,
        resume_from=checkpoint_path,
        use_wandb=use_wandb,
        resumed_output_dir=resumed_output_dir,
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
        help="Disable wandb logging",
    )

    args = parser.parse_args()

    resume_training(
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
    )

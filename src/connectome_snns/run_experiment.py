"""
Universal experiment runner that loads and executes experiments from TOML configs.

Usage:
    python src/run_experiment.py path/to/experiment.toml [--no-commit]
    python src/run_experiment.py path/to/experiment.toml --resume path/to/output_dir [--no-commit]

Use --no-commit to skip git status checks (useful for development/debugging).
Use --resume to continue training from the latest checkpoint in an existing output directory.

Examples:
    python -m connectome_snns.run_experiment path/to/experiment.toml
    python -m connectome_snns.run_experiment path/to/experiment.toml --no-commit
    python -m connectome_snns.run_experiment path/to/experiment.toml --resume /path/to/output
"""

import os
import sys

# Limit PyTorch threads so multiple experiments can share the machine.
# Must be set before importing torch (which experiment scripts will do).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


from connectome_snns.utils.experiment_runners import run_experiment, resume_experiment


if __name__ == "__main__":
    # Parse arguments
    config_path = None
    skip_git_check = False
    resume_dir = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--no-commit":
            skip_git_check = True
        elif args[i] == "--resume":
            i += 1
            if i < len(args):
                resume_dir = args[i]
            else:
                print("ERROR: --resume requires an output directory path")
                sys.exit(1)
        elif not args[i].startswith("--"):
            config_path = args[i]
        i += 1

    if config_path is None:
        print("ERROR: Experiment config file required")
        print(
            "Usage: python src/run_experiment.py path/to/experiment.toml [--resume output_dir]"
        )
        sys.exit(1)

    if resume_dir is not None:
        resume_experiment(config_path, resume_dir, skip_git_check=skip_git_check)
    else:
        run_experiment(config_path, skip_git_check=skip_git_check)

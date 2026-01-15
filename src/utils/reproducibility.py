"""
Reproducibility module for running experiments with tracked parameters and metadata.

Usage in your scripts:
    from utils.reproducibility import ExperimentTracker

    tracker = ExperimentTracker()

    with tracker:
        output_dir, params_file = tracker.output_dir, tracker.params_file

        # Your code here...
        # Errors are automatically caught and logged
"""

import subprocess
import sys
import toml
import json
import shutil
import traceback
from pathlib import Path
from datetime import datetime


def get_repo_root():
    """Get the repository root directory."""
    # Navigate up from this file (src/utils/reproducibility.py) to repo root
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


class ExperimentTracker:
    """
    Tracks experiment reproducibility by checking git status, logging metadata,
    and managing output directories.
    """

    def __init__(self, config_path=None, skip_git_check=False):
        """
        Initialize the experiment tracker.

        Args:
            config_path: Path to the TOML configuration file.
                        Defaults to workspace/experiment.toml in the repository root.
            skip_git_check: If True, skip git status validation (useful for development).
        """
        if config_path is None:
            # Default to workspace/experiment.toml in repository root
            config_path = get_repo_root() / "workspace" / "experiment.toml"
        self.config_path = Path(config_path)
        self.skip_git_check = skip_git_check
        self.config = self._load_config()
        self.start_time = None
        self.git_info = None
        self.calling_script = None
        self.output_dir = None
        self.completed_successfully = False
        self.initial_metadata = None
        self.params_file = None
        self.wandb_config = None
        self.input_dir = None  # Will be populated if data section exists

    def __enter__(self):
        """Context manager entry - start tracking."""
        self._start_tracking()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Context manager exit - handle success or failure."""
        if exc_type is None:
            # No exception - successful completion
            self._end_tracking_success()
        else:
            # Exception occurred - handle failure
            self._end_tracking_failure(exc_type, exc_value, exc_tb)
            # Return False to re-raise the exception
            return False

    def _load_config(self):
        """Load experiment configuration from TOML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        return toml.load(self.config_path)

    def _check_git_status(self):
        """Check if there are uncommitted changes and get current commit hash.

        Allows uncommitted changes to parameters/*.toml and workspace/* files,
        but requires all code files to be committed for reproducibility.
        """
        try:
            # If skipping git checks, still get commit info but don't validate cleanliness
            if self.skip_git_check:
                print("⚠️  Skipping git status validation (--no-commit flag used)")
            else:
                # Check for uncommitted changes
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                if result.stdout.strip():
                    # Parse dirty files
                    dirty_files = []
                    for line in result.stdout.strip().split("\n"):
                        if line:
                            # Format: "XY filename" where X and Y are status codes
                            # XY is exactly 2 chars; find first space/whitespace after that
                            # to handle edge cases more robustly
                            rest = line[2:]  # Skip the 2 status chars
                            filename = rest.lstrip().strip()
                            if filename:
                                dirty_files.append(filename)

                    # Filter out allowed dirty files (parameters and workspace)
                    code_dirty_files = [
                        f
                        for f in dirty_files
                        if not f.startswith("parameters/")
                        and not f.startswith("workspace/")
                    ]

                    if code_dirty_files:
                        print("ERROR: You have uncommitted changes in code files:")
                        for f in code_dirty_files:
                            print(f"  {f}")
                        print(
                            "\nPlease commit code changes before running experiments."
                        )
                        print(
                            "Note: Changes to parameters/*.toml and workspace/* files are allowed."
                        )
                        print("Or use --no-commit flag to skip this check entirely.")
                        sys.exit(1)
                    elif dirty_files:
                        # Only parameter/workspace files are dirty - this is fine
                        print("ℹ️  Uncommitted parameter files detected (allowed):")
                        param_files = [
                            f
                            for f in dirty_files
                            if f.startswith("parameters/") or f.startswith("workspace/")
                        ]
                        for f in param_files:
                            print(f"  {f}")

            # Get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            )
            commit_hash = result.stdout.strip()

            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            branch = result.stdout.strip()

            # Get remote URL
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                check=True,
            )
            remote_url = result.stdout.strip()

            return {
                "commit_hash": commit_hash,
                "branch": branch,
                "remote_url": remote_url,
            }

        except subprocess.CalledProcessError as e:
            print(f"ERROR: Git command failed: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("ERROR: Git not found. Is this a git repository?")
            sys.exit(1)

    def _validate_config(self):
        """Validate that required fields are present in config."""
        required_fields = ["script", "output_dir", "parameters_file", "log_file"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field '{field}' in config")

        # Validate that script exists (resolve relative paths to repo root)
        script_path = resolve_relative_to_repo(self.config["script"])
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Validate that parameters file exists (resolve relative paths to repo root)
        params_path = resolve_relative_to_repo(self.config["parameters_file"])
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_path}")

    def _extract_wandb_config(self):
        """Extract wandb configuration from config if present."""
        if "wandb" in self.config:
            return self.config["wandb"]
        return None

    def _setup_input_data(self, output_dir):
        """
        Setup input data directory based on config.

        Creates output_dir/inputs/ with either symlinks or copies of source data,
        depending on strategy specified in config.

        Args:
            output_dir: Path to experiment output directory

        Returns:
            Path to inputs directory, or None if no data inputs specified
        """
        if "data" not in self.config:
            return None

        data_config = self.config["data"]

        # Get input entries
        input_entries = data_config.get("inputs", [])
        if not input_entries:
            # No inputs specified - this is fine, just return None
            return None

        # Create inputs directory in output folder
        inputs_dir = output_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        print("\nSetting up input data...")

        # Process each input entry
        for i, entry in enumerate(input_entries):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Input entry {i} must be a dict with 'path' and 'strategy' keys"
                )

            source_path_str = entry.get("path")
            if not source_path_str:
                raise ValueError(f"Input entry {i} missing 'path' key")

            strategy = entry.get("strategy")
            if not strategy:
                raise ValueError(
                    f"Input entry {i} missing 'strategy' key. Must be 'symlink' or 'copy'."
                )

            # Validate strategy
            if strategy not in ["symlink", "copy"]:
                raise ValueError(
                    f"Invalid strategy '{strategy}' for {source_path_str}. Must be 'symlink' or 'copy'."
                )

            source_path = resolve_relative_to_repo(source_path_str)

            # Check if source exists
            if not source_path.exists():
                raise FileNotFoundError(f"Input data not found: {source_path}")

            # Determine destination name
            dest_path = inputs_dir / source_path.name

            # Handle files vs directories
            if source_path.is_file():
                if strategy == "symlink":
                    dest_path.symlink_to(source_path.absolute())
                    print(f"  ✓ Linked: {source_path.name}")
                else:  # copy
                    shutil.copy2(source_path, dest_path)
                    print(f"  ✓ Copied: {source_path.name}")
            elif source_path.is_dir():
                if strategy == "symlink":
                    dest_path.symlink_to(
                        source_path.absolute(), target_is_directory=True
                    )
                    print(f"  ✓ Linked: {source_path.name}/")
                else:  # copy
                    shutil.copytree(source_path, dest_path)
                    print(f"  ✓ Copied: {source_path.name}/")
            else:
                raise ValueError(
                    f"Input path is neither file nor directory: {source_path}"
                )

        print(f"✓ Input data ready: {inputs_dir}")
        return inputs_dir

    def _create_initial_metadata(self):
        """Create initial metadata at experiment start."""
        return {
            "script": self.config["script"],
            "description": self.config.get("description", ""),
            "parameters_file": self.config["parameters_file"],
            "output_dir": str(self.config["output_dir"]),
            "git_commit": self.git_info["commit_hash"],
            "git_branch": self.git_info["branch"],
            "git_remote_url": self.git_info["remote_url"],
            "start_time": self.start_time.isoformat(),
            "status": "running",
        }

    def _finalize_metadata(self, end_time, success=True):
        """Finalize metadata with end time and status."""
        metadata = self.initial_metadata.copy()
        metadata["end_time"] = end_time.isoformat()
        metadata["duration_seconds"] = (end_time - self.start_time).total_seconds()
        metadata["status"] = "completed" if success else "failed"
        return metadata

    def _log_metadata(self, metadata, log_file):
        """Append metadata to log file."""
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Append as JSON lines
        with open(log_file, "a") as f:
            f.write(json.dumps(metadata) + "\n")

        print(f"✓ Logged metadata to: {log_file}")

    def _end_tracking_failure(self, exc_type, exc_value, exc_tb):
        """Handle experiment failure by logging error and saving partial results."""
        end_time = datetime.now()

        # Format error trace
        error_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

        # Save error trace to log.err in output directory
        error_file = self.output_dir / "log.err"
        error_file.write_text(error_trace)
        print(f"\n✗ Error trace saved to: {error_file}")

        # Update metadata with failure status
        metadata = self._finalize_metadata(end_time, success=False)
        metadata["error_file"] = str(error_file)

        # Update metadata.json in output directory
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create README in output directory
        readme_content = self._create_readme(metadata)
        readme_file = self.output_dir / "README.md"
        with open(readme_file, "w") as f:
            f.write(readme_content)

        # Log failed experiment to central log
        log_file = resolve_relative_to_repo(self.config["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(metadata) + "\n")

        print(
            f"✗ Experiment failed. Partial results and error log saved to: {self.output_dir}"
        )
        print(f"✗ Failure logged to: {log_file}\n")

    def _create_readme(self, metadata):
        """Generate a README for the experiment output folder."""
        status = metadata.get("status", "unknown")
        status_emoji = "✓" if status == "completed" else "✗"

        # Extract repo name from URL
        repo_url = metadata["git_remote_url"]
        if repo_url.endswith(".git"):
            repo_name = repo_url.split("/")[-1][:-4]
        else:
            repo_name = repo_url.split("/")[-1]

        # Absolute path to the reproducibility copy of experiment.toml in output dir
        # Use metadata["output_dir"] which is written earlier when the file is copied
        try:
            experiment_toml_abs = str(
                (Path(metadata["output_dir"]) / "experiment.toml").resolve()
            )
        except Exception:
            # Fallback to self.output_dir if metadata is unexpectedly formatted
            experiment_toml_abs = str((self.output_dir / "experiment.toml").resolve())

        readme = f"""# Experiment Run - {status_emoji} {status.upper()}

## Description
{metadata.get("description", "No description provided.")}

## Execution Details
- **Script**: `{metadata["script"]}`
- **Start Time**: {metadata["start_time"]}
- **End Time**: {metadata.get("end_time", "N/A")}
- **Duration**: {metadata.get("duration_seconds", 0):.2f} seconds
- **Status**: {status_emoji} **{status.upper()}**
"""

        if status == "failed" and "error_file" in metadata:
            readme += f"""
## Error Information
⚠️ **This experiment failed during execution.**

Error details saved to: `{Path(metadata["error_file"]).name}`

See the error file in this directory for the full traceback.
"""

        readme += f"""
## Git Information
- **Repository**: {metadata["git_remote_url"]}
- **Commit**: {metadata["git_commit"]}
- **Branch**: {metadata["git_branch"]}

## Parameters
Parameters loaded from: `{metadata["parameters_file"]}`

See `parameters.{Path(metadata["parameters_file"]).suffix[1:]}` in this directory for the exact parameters used.

## Reproducibility

To reproduce this experiment exactly:

### 1. Clone the repository
```bash
git clone {metadata["git_remote_url"]}
cd {repo_name}
```

### 2. Check out the exact commit
```bash
git checkout {metadata["git_commit"]}
```

### 3. Set up the Python environment
```bash
# Using uv (recommended)
uv sync --extra cpu  # or --extra gpu

# Or using pip
pip install -e .
```

### 4. Run the experiment
```bash
# Using the executable wrapper (recommended)
./run {experiment_toml_abs}

# Or using uv directly
uv run python run_experiment.py {experiment_toml_abs}

You will most likeley be prompted to enter a new output directory to avoid overwriting existing results.
```

**Note**: The command above uses the absolute path to the `experiment.toml` file in this output directory, which contains the exact parameters and paths for this specific run.

## Files in This Directory
- `parameters.{Path(metadata["parameters_file"]).suffix[1:]}` - Exact parameters used for this run
- `experiment.toml` - Configuration file (with paths updated to this directory)
- `metadata.json` - Machine-readable metadata
- `README.md` - This file
"""
        if status == "failed":
            readme += "- `log.err` - Error traceback from failed run\n"

        return readme

    def _start_tracking(self):
        """Start tracking the experiment (called by context manager)."""
        # Validate configuration
        self._validate_config()

        # Extract wandb config
        self.wandb_config = self._extract_wandb_config()

        title = "Starting Experiment Tracking"
        print(f"\n{'=' * len(title)}")
        print(title)
        print(f"{'=' * len(title)}\n")
        print(f"Script: {self.config['script']}")
        print(f"Parameters: {self.config['parameters_file']}")
        print(f"Output: {self.config['output_dir']}")
        print(f"Description: {self.config.get('description', 'N/A')}")
        if self.wandb_config and self.wandb_config.get("enabled", False):
            print(f"W&B Project: {self.wandb_config.get('project', 'N/A')}")

        # Check git status
        if self.skip_git_check:
            print("\nChecking git status (--no-commit mode)...")
            self.git_info = self._check_git_status()
            print(f"⚠️  Git check skipped (commit: {self.git_info['commit_hash'][:8]})")
        else:
            print("\nChecking git status...")
            self.git_info = self._check_git_status()
            print(f"✓ Git status clean (commit: {self.git_info['commit_hash'][:8]})")

        # Check if output directory exists (resolve relative paths to repo root)
        output_dir = resolve_relative_to_repo(self.config["output_dir"])
        while output_dir.exists():
            print(f"\n⚠️  WARNING: Output directory already exists: {output_dir}")
            print("\nChoose an option:")
            print("  [c] Cancel experiment")
            print("  [o] Overwrite existing folder")
            print("  [n] Specify a new output folder")
            response = input("\nYour choice (c/o/n): ").lower().strip()

            if response == "c":
                print("Experiment cancelled.")
                sys.exit(0)
            elif response == "o":
                print(f"⚠️  Overwriting: {output_dir}")
                # Remove existing directory and its contents
                shutil.rmtree(output_dir)
                break
            elif response == "n":
                new_path = input("Enter new output folder path: ").strip()
                if new_path:
                    output_dir = Path(new_path)
                    # Update config with new path
                    self.config["output_dir"] = str(output_dir)
                    print(f"Updated output directory to: {output_dir}")
                else:
                    print("Invalid path. Please try again.")
            else:
                print("Invalid choice. Please enter 'c', 'o', or 'n'.")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")

        # Copy parameters file to output folder
        params_path = resolve_relative_to_repo(self.config["parameters_file"])
        params_filename = f"parameters{params_path.suffix}"
        dest_params = output_dir / params_filename
        shutil.copy2(params_path, dest_params)
        print(f"✓ Copied parameters file to: {dest_params}")

        # Setup input data if specified in config
        self.input_dir = self._setup_input_data(output_dir)

        # Copy and modify experiment.toml to output folder
        dest_toml = output_dir / "experiment.toml"
        modified_config = self.config.copy()
        # Update paths to point to copied files in output directory
        modified_config["parameters_file"] = str(
            (output_dir / params_filename).resolve()
        )
        modified_config["output_dir"] = str(output_dir.resolve())
        # Keep log_file absolute as-is (resolve relative paths to repo root)
        modified_config["log_file"] = str(
            resolve_relative_to_repo(self.config["log_file"])
        )
        # If data section exists, update inputs to point to local inputs folder
        if "data" in modified_config and self.input_dir:
            modified_config["data"]["inputs"] = [str(self.input_dir.resolve())]
        # Keep script relative to repo root
        with open(dest_toml, "w") as f:
            toml.dump(modified_config, f)
        print(f"✓ Copied experiment config to: {dest_toml}")

        # Store params_file for easy access
        self.params_file = params_path

        # Record start time and output directory
        self.start_time = datetime.now()
        self.output_dir = output_dir

        # Create and save initial metadata
        self.initial_metadata = self._create_initial_metadata()
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.initial_metadata, f, indent=2)
        print(f"✓ Saved initial metadata to: {metadata_file}")

        title = "Experiment tracking initialized. Running your code now."
        print(f"\n{'=' * len(title)}")
        print(title)
        print(f"{'=' * len(title)}\n")

    def _end_tracking_success(self):
        """End tracking for successful experiment completion."""
        end_time = datetime.now()

        title = "Finalizing Experiment Tracking"
        print(f"\n{'=' * len(title)}")
        print(title)
        print(f"{'=' * len(title)}\n")

        # Finalize metadata with success status
        metadata = self._finalize_metadata(end_time, success=True)

        # Update metadata file in output folder
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Updated metadata: {metadata_file}")

        # Create README
        readme_content = self._create_readme(metadata)
        readme_file = self.output_dir / "README.md"
        with open(readme_file, "w") as f:
            f.write(readme_content)
        print(f"✓ Generated README: {readme_file}")

        # Log to central log file
        log_file = resolve_relative_to_repo(self.config["log_file"])
        self._log_metadata(metadata, log_file)

        title = f"Experiment complete! Results saved to: {self.output_dir}"
        print(f"\n{'=' * len(title)}")
        print(title)
        print(f"Duration: {metadata['duration_seconds']:.2f} seconds")
        print(f"{'=' * len(title)}\n")

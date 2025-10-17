"""
Reproducibility module for running experiments with tracked parameters and metadata.

Usage in your scripts:
    from utils.reproducibility import ExperimentTracker

    tracker = ExperimentTracker()

    with tracker:
        output_dir, params_csv = tracker.output_dir, tracker.params_csv

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


class ExperimentTracker:
    """
    Tracks experiment reproducibility by checking git status, logging metadata,
    and managing output directories.
    """

    def __init__(self, config_path=None):
        """
        Initialize the experiment tracker.

        Args:
            config_path: Path to the TOML configuration file.
                        Defaults to experiment.toml in the repository root.
        """
        if config_path is None:
            # Default to experiment.toml in repository root (two levels up from this file)
            repo_root = Path(__file__).resolve().parent.parent.parent
            config_path = repo_root / "experiment.toml"
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.start_time = None
        self.git_info = None
        self.calling_script = None
        self.output_dir = None
        self.completed_successfully = False
        self.initial_metadata = None
        self.params_csv = None

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
        """Check if there are uncommitted changes and get current commit hash."""
        try:
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            )

            if result.stdout.strip():
                # Filter out whitelisted files that can be edited on the fly
                whitelist = [
                    "experiment.toml",  # Root experiment config
                    "parameters/",  # Parameter files directory
                ]

                # Parse git status output and filter
                lines = result.stdout.strip().split("\n")
                non_whitelisted_changes = []

                for line in lines:
                    if not line.strip():
                        continue
                    # Git status format: "XY filename" where XY are status codes
                    # Extract the filename (everything after the first 3 characters)
                    if len(line) > 3:
                        filename = line[3:].strip()
                        # Check if this file is whitelisted
                        is_whitelisted = any(
                            filename == wl or filename.startswith(wl)
                            for wl in whitelist
                        )
                        if not is_whitelisted:
                            non_whitelisted_changes.append(line)

                if non_whitelisted_changes:
                    print("ERROR: You have uncommitted changes:")
                    print("\n".join(non_whitelisted_changes))
                    print("\nPlease commit all changes before running experiments.")
                    print(
                        "(Note: experiment.toml and parameters/ are whitelisted and can be edited)"
                    )
                    sys.exit(1)

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
        required_fields = ["script", "output_dir", "parameters_csv", "log_file"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field '{field}' in config")

        # Validate that script exists
        script_path = Path(self.config["script"])
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Validate that parameters CSV exists
        csv_path = Path(self.config["parameters_csv"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Parameters CSV not found: {csv_path}")

    def _create_initial_metadata(self):
        """Create initial metadata at experiment start."""
        return {
            "script": self.config["script"],
            "description": self.config.get("description", ""),
            "parameters_csv": self.config["parameters_csv"],
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
        log_file = Path(self.config["log_file"])
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
Parameters loaded from: `{metadata["parameters_csv"]}`

See `parameters.csv` in this directory for the exact parameters used.

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
./run_experiment {experiment_toml_abs}

# Or using uv directly
uv run python run_experiment.py {experiment_toml_abs}

You will most likeley be prompted to enter a new output directory to avoid overwriting existing results.
```

**Note**: The command above uses the absolute path to the `experiment.toml` file in this output directory, which contains the exact parameters and paths for this specific run.

## Files in This Directory
- `parameters.csv` - Exact parameters used for this run
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

        title = "Starting Experiment Tracking"
        print(f"\n{'=' * len(title)}")
        print(title)
        print(f"{'=' * len(title)}\n")
        print(f"Script: {self.config['script']}")
        print(f"Parameters: {self.config['parameters_csv']}")
        print(f"Output: {self.config['output_dir']}")
        print(f"Description: {self.config.get('description', 'N/A')}")

        # Check git status
        print("\nChecking git status...")
        self.git_info = self._check_git_status()
        print(f"✓ Git status clean (commit: {self.git_info['commit_hash'][:8]})")

        # Check if output directory exists
        output_dir = Path(self.config["output_dir"])
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

        # Copy parameters CSV to output folder
        csv_path = Path(self.config["parameters_csv"])
        dest_csv = output_dir / "parameters.csv"
        shutil.copy2(csv_path, dest_csv)
        print(f"✓ Copied parameters CSV to: {dest_csv}")

        # Copy and modify experiment.toml to output folder
        dest_toml = output_dir / "experiment.toml"
        modified_config = self.config.copy()
        # Update paths to point to copied files in output directory
        modified_config["parameters_csv"] = str(
            (output_dir / "parameters.csv").resolve()
        )
        modified_config["output_dir"] = str(output_dir.resolve())
        # Keep log_file absolute as-is
        modified_config["log_file"] = str(Path(self.config["log_file"]).resolve())
        # Keep script relative to repo root
        with open(dest_toml, "w") as f:
            toml.dump(modified_config, f)
        print(f"✓ Copied experiment config to: {dest_toml}")

        # Store params_csv for easy access
        self.params_csv = csv_path

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
        log_file = Path(self.config["log_file"])
        self._log_metadata(metadata, log_file)

        title = f"Experiment complete! Results saved to: {self.output_dir}"
        print(f"\n{'=' * len(title)}")
        print(title)
        print(f"Duration: {metadata['duration_seconds']:.2f} seconds")
        print(f"{'=' * len(title)}\n")

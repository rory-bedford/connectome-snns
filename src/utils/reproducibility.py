"""
Reproducibility module for running experiments with tracked parameters and metadata.

Usage in your scripts:
    from utils.reproducibility import ExperimentTracker

    tracker = ExperimentTracker("experiment.toml")
    output_dir, params_csv = tracker.start()

    # Your code here...
    # Use output_dir and params_csv as needed

    tracker.end()
"""

import subprocess
import sys
import toml
import json
import shutil
import inspect
import atexit
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

    def _load_config(self):
        """Load experiment configuration from TOML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        return toml.load(self.config_path)

    def _detect_calling_script(self):
        """Detect the script that called this module."""
        # Walk up the call stack to find the first file that's not this module
        for frame_info in inspect.stack():
            filepath = Path(frame_info.filename).resolve()
            # Skip if it's this reproducibility module
            if filepath == Path(__file__).resolve():
                continue
            # Skip internal Python files
            if str(filepath).startswith(sys.prefix):
                continue
            # Found the calling script
            return filepath
        return None

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
                print("ERROR: You have uncommitted changes:")
                print(result.stdout)
                print("\nPlease commit all changes before running experiments.")
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
        """Validate that required fields are present in config and script matches."""
        required_fields = ["script", "output_dir", "parameters_csv", "log_file"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field '{field}' in config")

        # Detect the calling script
        self.calling_script = self._detect_calling_script()

        # Validate that the script field matches the calling script
        if self.calling_script is not None:
            config_script = Path(self.config["script"]).resolve()
            if config_script != self.calling_script:
                print("ERROR: Script mismatch!")
                print(f"  Config specifies: {config_script}")
                print(f"  Actually running: {self.calling_script}")
                print(f"\nPlease update 'script' field in {self.config_path}")
                sys.exit(1)

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

    def _handle_failure(self):
        """Handle experiment failure by logging error and saving partial results."""
        if self.completed_successfully or self.output_dir is None:
            return

        end_time = datetime.now()

        # Get exception information
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_type is not None:
            error_trace = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )

            # Save error trace to log.err in output directory
            error_file = self.output_dir / "log.err"
            error_file.write_text(error_trace)
            print(f"✗ Error trace saved to: {error_file}")

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
            print(f"✗ Failure logged to: {log_file}")

    def _create_readme(self, metadata):
        """Generate a README for the experiment output folder."""
        status = metadata.get("status", "unknown")
        status_emoji = "✓" if status == "completed" else "✗"

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
To reproduce this experiment:
1. Check out commit: `git checkout {metadata["git_commit"]}`
2. Run the script with the same parameters
"""
        return readme

    def start(self):
        """
        Start tracking the experiment. Call this at the beginning of your script.

        Returns:
            tuple: (output_dir, parameters_csv) as Path objects
        """
        # Validate configuration
        self._validate_config()

        title = "Starting Experiment Tracking"
        print(f"\n{'=' * len(title)}")
        print(title)
        print(f"{'=' * len(title)}\n")
        print(f"Script: {self.calling_script or self.config['script']}")
        print(f"Parameters: {self.config['parameters_csv']}")
        print(f"Output: {self.config['output_dir']}")
        print(f"Description: {self.config.get('description', 'N/A')}")

        # Check git status
        print("\nChecking git status...")
        self.git_info = self._check_git_status()
        print(f"✓ Git status clean (commit: {self.git_info['commit_hash'][:8]})")

        # Check if output directory exists
        output_dir = Path(self.config["output_dir"])
        if output_dir.exists():
            print(f"\n⚠️  WARNING: Output directory already exists: {output_dir}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != "y":
                print("Experiment cancelled.")
                sys.exit(0)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created output directory: {output_dir}")

        # Copy parameters CSV to output folder
        csv_path = Path(self.config["parameters_csv"])
        dest_csv = output_dir / "parameters.csv"
        shutil.copy2(csv_path, dest_csv)
        print(f"✓ Copied parameters CSV to: {dest_csv}")

        # Record start time and output directory
        self.start_time = datetime.now()
        self.output_dir = output_dir

        # Create and save initial metadata
        self.initial_metadata = self._create_initial_metadata()
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.initial_metadata, f, indent=2)
        print(f"✓ Saved initial metadata to: {metadata_file}")

        # Create initial README
        readme_content = self._create_readme(self.initial_metadata)
        readme_file = output_dir / "README.md"
        with open(readme_file, "w") as f:
            f.write(readme_content)
        print(f"✓ Created README: {readme_file}")

        # Register error handler to run on unexpected exit
        atexit.register(self._handle_failure)

        title = "Experiment tracking initialized. Run your code now."
        print(f"\n{'=' * len(title)}")
        print(title)
        print(f"{'=' * len(title)}\n")

        # Return paths for the script to use
        return output_dir, csv_path

    def end(self):
        """
        End tracking the experiment. Call this at the end of your script.
        Saves metadata, creates README, and logs to central log directory.
        """
        if self.start_time is None:
            raise RuntimeError("Must call start() before end()")

        # Mark as successfully completed (prevents error handler from running)
        self.completed_successfully = True

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

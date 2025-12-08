"""
Debug script for visualizing cross-correlation between trials.

This script loads saved spike data from zarr files and generates
cross-correlation scatter plots comparing different patterns/trials.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import toml
import zarr
from visualization.firing_statistics import plot_cross_correlation_scatter


def main():
    """Load spike data and generate cross-correlation scatter plots."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate cross-correlation scatter plots from saved spike data"
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to the output folder containing spike_data.zarr and parameters.toml",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=10.0,
        help="Window size in seconds for binning spikes (default: 10.0)",
    )
    parser.add_argument(
        "--pattern1",
        type=str,
        default="odour_0",
        help="First pattern to compare: 'odour_X' or 'control' (default: odour_0)",
    )
    parser.add_argument(
        "--pattern2",
        type=str,
        default="control",
        help="Second pattern to compare: 'odour_X' or 'control' (default: control)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path for the plot (default: folder/figures/cross_correlation_debug.png)",
    )

    args = parser.parse_args()

    # Validate folder exists
    if not args.folder.exists():
        raise FileNotFoundError(f"Folder not found: {args.folder}")

    zarr_path = args.folder / "spike_data.zarr"
    params_path = args.folder / "parameters.toml"

    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr file not found: {zarr_path}")

    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    print("=" * 60)
    print("CROSS-CORRELATION SCATTER PLOT DEBUG")
    print("=" * 60)
    print(f"Loading data from: {args.folder}")
    print(f"Window size: {args.window_size}s")
    print(f"Comparing: {args.pattern1} vs {args.pattern2}")

    # Load parameters
    with open(params_path, "r") as f:
        params_data = toml.load(f)
    dt = params_data["simulation"]["dt"]
    print(f"Time step (dt): {dt} ms")

    # Load zarr data
    print("\nLoading spike data from zarr...")
    root = zarr.open_group(zarr_path, mode="r")

    # Helper function to parse pattern specification
    def get_pattern_data(pattern_spec):
        if pattern_spec == "control":
            return root["output_spikes_control"][:]
        elif pattern_spec.startswith("odour_"):
            pattern_idx = int(pattern_spec.split("_")[1])
            odour_data = root["output_spikes_odour"][:]
            if pattern_idx >= odour_data.shape[1]:
                raise ValueError(
                    f"Pattern index {pattern_idx} out of range. "
                    f"Available odour patterns: 0-{odour_data.shape[1] - 1}"
                )
            # Extract single pattern and add back the pattern dimension
            return odour_data[:, pattern_idx : pattern_idx + 1, :, :]
        else:
            raise ValueError(
                f"Invalid pattern specification: {pattern_spec}. "
                "Use 'control' or 'odour_X' (e.g., 'odour_0')"
            )

    # Load the two patterns
    spikes_pattern1 = get_pattern_data(args.pattern1)
    spikes_pattern2 = get_pattern_data(args.pattern2)

    print(f"Pattern 1 ({args.pattern1}) shape: {spikes_pattern1.shape}")
    print(f"Pattern 2 ({args.pattern2}) shape: {spikes_pattern2.shape}")

    # Squeeze out the pattern dimension (should be 1 after extraction)
    # Shape: (batch_size, 1, time, neurons) -> (batch_size, time, neurons)
    spikes_pattern1 = np.squeeze(spikes_pattern1, axis=1)
    spikes_pattern2 = np.squeeze(spikes_pattern2, axis=1)

    batch_size, n_steps, n_neurons = spikes_pattern1.shape
    duration_s = n_steps * dt / 1000

    print("\nData summary:")
    print(f"  Batch size: {batch_size}")
    print(f"  Time steps: {n_steps}")
    print(f"  Duration: {duration_s:.2f}s")
    print(f"  Neurons: {n_neurons}")

    # Generate the cross-correlation scatter plot
    print("\nGenerating cross-correlation scatter plot...")
    fig = plot_cross_correlation_scatter(
        spike_trains_trial1=spikes_pattern1,
        spike_trains_trial2=spikes_pattern2,
        window_size=args.window_size,
        dt=dt,
        cell_indices=None,  # Use all cells
        title=f"{args.pattern1} vs {args.pattern2}",
        x_label=f"{args.pattern1.capitalize()} Firing Rate (Hz)",
        y_label=f"{args.pattern2.capitalize()} Firing Rate (Hz)",
    )

    # Determine output path
    if args.output is None:
        figures_dir = args.folder / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        output_path = figures_dir / "cross_correlation_debug.png"
    else:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✓ Saved plot to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Generate plots for odour input patterns from saved simulation data."""

import matplotlib.pyplot as plt
import numpy as np
import toml
import zarr
from pathlib import Path
from analysis.firing_rate import compute_firing_rates_from_zarr
from visualization import (
    plot_input_firing_rate_histogram,
    plot_firing_rate_variance_comparison,
)
from visualization.firing_statistics import (
    plot_firing_rate_distribution,
    plot_cv_histogram,
    plot_isi_histogram,
    plot_fano_factor_vs_window_size,
    plot_psth,
)


def main(output_dir: Path):
    """Generate odour-related plots from saved simulation data.

    Args:
        output_dir (Path): Directory containing the results/ folder with spike_data.zarr
    """
    # Load simulation parameters to get dt
    param_file = output_dir / "parameters.toml"
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")

    with open(param_file, "r") as f:
        params = toml.load(f)
        dt = params["simulation"]["dt"]

    print(f"Using dt = {dt} ms")

    # Load network structure for activity dashboards
    network_structure_path = output_dir / "inputs" / "network_structure.npz"
    if not network_structure_path.exists():
        raise FileNotFoundError(
            f"Network structure file not found: {network_structure_path}"
        )

    network_data = np.load(network_structure_path)
    cell_type_indices = network_data["cell_type_indices"]

    print(f"Loaded network structure: {len(cell_type_indices)} neurons")

    # Compute firing rates using memory-efficient Dask computation
    print("Computing input firing rates from spike data...")
    results_dir = output_dir / "results"

    # Returns shape: (batch_size, n_patterns, n_input_neurons)
    input_firing_rates = compute_firing_rates_from_zarr(
        zarr_path=results_dir / "spike_data.zarr",
        dataset_name="input_spikes",
        dt=dt,
    )

    print(f"✓ Computed input firing rates: {input_firing_rates.shape}")

    # Compute output firing rates for main experiment
    print("Computing output firing rates from spike data...")
    output_firing_rates = compute_firing_rates_from_zarr(
        zarr_path=results_dir / "spike_data.zarr",
        dataset_name="output_spikes",
        dt=dt,
    )
    print(f"✓ Computed output firing rates: {output_firing_rates.shape}")

    # Compute output firing rates for control experiment
    print("Computing control output firing rates...")
    output_firing_rates_control = compute_firing_rates_from_zarr(
        zarr_path=results_dir / "spike_data_control.zarr",
        dataset_name="output_spikes",
        dt=dt,
    )
    print(
        f"✓ Computed control output firing rates: {output_firing_rates_control.shape}"
    )

    # Create figures directory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot input firing rate histogram (pass full array, function will handle averaging)
    fig = plot_input_firing_rate_histogram(
        firing_rates=input_firing_rates,
    )
    fig.savefig(
        figures_dir / "input_firing_rate_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"✓ Saved input firing rate histogram to {figures_dir}")

    # Plot firing rate variance comparison between odour-modulated and control
    fig = plot_firing_rate_variance_comparison(
        output_firing_rates=output_firing_rates,
        output_firing_rates_control=output_firing_rates_control,
        n_spaghetti=50,
    )
    fig.savefig(
        figures_dir / "firing_rate_variance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"✓ Saved firing rate variance comparison to {figures_dir}")

    # Generate spike-based statistics plots for odour and control
    print("\nGenerating spike-based statistics plots...")

    # Load spike data from zarr (first batch, first pattern)
    root = zarr.open(results_dir / "spike_data.zarr", mode="r")
    root_control = zarr.open(results_dir / "spike_data_control.zarr", mode="r")

    # Get cell type names from params
    cell_type_names = params["recurrent"]["cell_types"]["names"]
    n_cell_types = len(cell_type_names)

    # Load input and output spikes: add batch dimension back for compatibility
    input_spikes_odour = np.array(root["input_spikes"][0, 0, :, :])[
        np.newaxis, :, :
    ]  # (1, time, n_input_neurons)
    input_spikes_control = np.array(root_control["input_spikes"][0, 0, :, :])[
        np.newaxis, :, :
    ]

    output_spikes_odour = np.array(root["output_spikes"][0, 0, :, :])[
        np.newaxis, :, :
    ]  # (1, time, neurons)
    output_spikes_control = np.array(root_control["output_spikes"][0, 0, :, :])[
        np.newaxis, :, :
    ]

    print(f"Odour spikes shape: {output_spikes_odour.shape}")
    print(f"Control spikes shape: {output_spikes_control.shape}")

    # Create 2-column figure for each metric (odour on left, control on right)

    # 1. PSTH
    print("Generating PSTH plots...")
    fig_psth, (ax_odour, ax_control) = plt.subplots(1, 2, figsize=(16, 5))
    plot_psth(
        spike_trains=output_spikes_odour,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        window_size=50.0,
        dt=dt,
        ax=ax_odour,
        title="PSTH: Odour-Modulated",
        input_spike_trains=input_spikes_odour,
    )
    plot_psth(
        spike_trains=output_spikes_control,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        window_size=50.0,
        dt=dt,
        ax=ax_control,
        title="PSTH: Control (Baseline)",
        input_spike_trains=input_spikes_control,
    )
    plt.tight_layout()
    fig_psth.savefig(figures_dir / "psth_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig_psth)
    print("✓ Saved PSTH comparison")

    # 2. Firing Rate Distribution
    print("Generating firing rate distribution plots...")
    fig_fr, axes_fr = plt.subplots(
        2, n_cell_types, figsize=(6 * n_cell_types, 10), sharey=True
    )
    if n_cell_types == 1:
        axes_fr = axes_fr.reshape(2, 1)

    plot_firing_rate_distribution(
        output_spikes=output_spikes_odour,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        ax=axes_fr[0, :].tolist(),
    )
    plot_firing_rate_distribution(
        output_spikes=output_spikes_control,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        ax=axes_fr[1, :].tolist(),
    )
    # Add row labels
    axes_fr[0, 0].text(
        -0.3,
        0.5,
        "Odour-Modulated",
        transform=axes_fr[0, 0].transAxes,
        rotation=90,
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    axes_fr[1, 0].text(
        -0.3,
        0.5,
        "Control (Baseline)",
        transform=axes_fr[1, 0].transAxes,
        rotation=90,
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig_fr.savefig(
        figures_dir / "firing_rate_distribution_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_fr)
    print("✓ Saved firing rate distribution comparison")

    # 3. CV Histogram
    print("Generating CV histogram plots...")
    fig_cv, axes_cv = plt.subplots(
        2, n_cell_types, figsize=(6 * n_cell_types, 10), sharey=True
    )
    if n_cell_types == 1:
        axes_cv = axes_cv.reshape(2, 1)

    plot_cv_histogram(
        spike_trains=output_spikes_odour,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        bins=50,
        ax=axes_cv[0, :].tolist(),
    )
    plot_cv_histogram(
        spike_trains=output_spikes_control,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        bins=50,
        ax=axes_cv[1, :].tolist(),
    )
    # Add row labels
    axes_cv[0, 0].text(
        -0.3,
        0.5,
        "Odour-Modulated",
        transform=axes_cv[0, 0].transAxes,
        rotation=90,
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    axes_cv[1, 0].text(
        -0.3,
        0.5,
        "Control (Baseline)",
        transform=axes_cv[1, 0].transAxes,
        rotation=90,
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig_cv.savefig(
        figures_dir / "cv_histogram_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_cv)
    print("✓ Saved CV histogram comparison")

    # 4. ISI Histogram
    print("Generating ISI histogram plots...")
    fig_isi, axes_isi = plt.subplots(
        2, n_cell_types, figsize=(6 * n_cell_types, 10), sharey=True
    )
    if n_cell_types == 1:
        axes_isi = axes_isi.reshape(2, 1)

    plot_isi_histogram(
        spike_trains=output_spikes_odour,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        bins=50,
        ax=axes_isi[0, :].tolist(),
    )
    plot_isi_histogram(
        spike_trains=output_spikes_control,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        bins=50,
        ax=axes_isi[1, :].tolist(),
    )
    # Add row labels
    axes_isi[0, 0].text(
        -0.3,
        0.5,
        "Odour-Modulated",
        transform=axes_isi[0, 0].transAxes,
        rotation=90,
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    axes_isi[1, 0].text(
        -0.3,
        0.5,
        "Control (Baseline)",
        transform=axes_isi[1, 0].transAxes,
        rotation=90,
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig_isi.savefig(
        figures_dir / "isi_histogram_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_isi)
    print("✓ Saved ISI histogram comparison")

    # 5. Fano Factor vs Window Size
    print("Generating Fano factor plots...")
    window_sizes = np.logspace(0, 3, 20).astype(int)  # 1 to 1000 steps
    fig_fano, axes_fano = plt.subplots(
        2, n_cell_types, figsize=(6 * n_cell_types, 10), sharey=True
    )
    if n_cell_types == 1:
        axes_fano = axes_fano.reshape(2, 1)

    plot_fano_factor_vs_window_size(
        spike_trains=output_spikes_odour,
        window_sizes=window_sizes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        ax=axes_fano[0, :].tolist(),
    )
    plot_fano_factor_vs_window_size(
        spike_trains=output_spikes_control,
        window_sizes=window_sizes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        ax=axes_fano[1, :].tolist(),
    )
    # Add row labels
    axes_fano[0, 0].text(
        -0.3,
        0.5,
        "Odour-Modulated",
        transform=axes_fano[0, 0].transAxes,
        rotation=90,
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    axes_fano[1, 0].text(
        -0.3,
        0.5,
        "Control (Baseline)",
        transform=axes_fano[1, 0].transAxes,
        rotation=90,
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig_fano.savefig(
        figures_dir / "fano_factor_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_fano)
    print("✓ Saved Fano factor comparison")

    print("\n✓ All spike-based statistics plots generated successfully!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python odour_plots.py <output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        sys.exit(1)

    main(output_dir)

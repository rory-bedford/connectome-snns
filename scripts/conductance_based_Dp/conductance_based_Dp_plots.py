"""
Plotting functions for Dp network simulation outputs

This script loads simulation outputs and generates comprehensive visualizations
using modular plotting functions from the visualization module.

Overview:
1. Network structure plots: assembly graphs and connectivity matrices
2. Input analysis: mitral cell spike trains and feedforward connectivity
3. Output analysis: spike trains, firing rates, membrane voltages, and currents

Usage:
    python conductance_based_Dp_plots.py <output_directory>
"""

import matplotlib.pyplot as plt
import numpy as np
import toml
from pathlib import Path
import sys
import csv
from visualization.connectivity import (
    plot_assembly_graph,
    plot_weighted_connectivity,
    plot_input_count_histogram,
    plot_synaptic_input_histogram,
    plot_feedforward_connectivity,
)
from visualization.neuronal_dynamics import (
    plot_membrane_voltages,
    plot_synaptic_currents,
    plot_mitral_cell_spikes,
    plot_dp_network_spikes,
    plot_synaptic_conductances,
)
from visualization.firing_statistics import (
    plot_fano_factor_vs_window_size,
    plot_cv_histogram,
    plot_isi_histogram,
    plot_firing_rate_distribution,
)
from analysis.firing_statistics import (
    compute_firing_rate_by_cell_type,
    compute_cv_by_cell_type,
)
from analysis.voltage_statistics import (
    compute_membrane_potential_by_cell_type,
)


def main(output_dir_path):
    """Main plotting function for Dp network simulation outputs.

    Loads parameters and generates all plots.

    Args:
        output_dir_path (str or Path): Path to directory containing simulation outputs
    """
    output_dir = Path(output_dir_path)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"

    # Create figures directory if it doesn't exist
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load parameters from the TOML file
    params_file = output_dir / "parameters.toml"
    if not params_file.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_file}")

    print(f"Loading parameters from {params_file}...")
    with open(params_file, "r") as f:
        params = toml.load(f)

    # Load all data from results/
    print(f"Loading arrays from {results_dir}...")
    connectivity_graph = np.load(results_dir / "connectivity_graph.npy")
    weights = np.load(results_dir / "weights.npy")
    feedforward_weights = np.load(results_dir / "feedforward_weights.npy")
    cell_type_indices = np.load(results_dir / "cell_type_indices.npy")
    input_cell_type_indices = np.load(results_dir / "input_cell_type_indices.npy")
    input_spikes = np.load(results_dir / "input_spikes.npy")
    output_spikes = np.load(results_dir / "output_spikes.npy")
    output_voltages = np.load(results_dir / "output_voltages.npy")
    output_currents = np.load(results_dir / "output_currents.npy")
    output_conductances = np.load(results_dir / "output_conductances.npy")
    input_conductances = np.load(results_dir / "input_conductances.npy")

    # Extract commonly used parameters
    dt = params["simulation"]["dt"]
    duration = params["simulation"]["duration"]
    num_assemblies = int(params["recurrent"]["topology"]["num_assemblies"])
    cell_type_names = params["recurrent"]["cell_types"]["names"]
    input_cell_type_names = params["feedforward"]["cell_types"]["names"]

    print(f"Generating plots and saving to {figures_dir}...")

    # Network structure plots
    fig = plot_assembly_graph(
        connectivity_graph=connectivity_graph,
        cell_type_indices=cell_type_indices,
        num_assemblies=num_assemblies,
    )
    fig.savefig(figures_dir / "01_assembly_graph.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plot_weighted_connectivity(
        weights=weights,
        cell_type_indices=cell_type_indices,
        num_assemblies=num_assemblies,
    )
    fig.savefig(
        figures_dir / "02_weighted_connectivity.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    fig = plot_input_count_histogram(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_cell_type_indices,
        cell_type_names=cell_type_names,
        input_cell_type_names=input_cell_type_names,
    )
    fig.savefig(
        figures_dir / "03_input_count_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Prepare g_bar dictionaries for synaptic input histogram
    recurrent_g_bar_by_type = {}
    for cell_type in cell_type_names:
        g_bar_values = params["recurrent"]["synapses"][cell_type]["g_bar"]
        recurrent_g_bar_by_type[cell_type] = sum(g_bar_values)

    feedforward_g_bar_by_type = {}
    for cell_type in input_cell_type_names:
        g_bar_values = params["feedforward"]["synapses"][cell_type]["g_bar"]
        feedforward_g_bar_by_type[cell_type] = sum(g_bar_values)

    fig = plot_synaptic_input_histogram(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_cell_type_indices,
        cell_type_names=cell_type_names,
        input_cell_type_names=input_cell_type_names,
        recurrent_g_bar_by_type=recurrent_g_bar_by_type,
        feedforward_g_bar_by_type=feedforward_g_bar_by_type,
    )
    fig.savefig(
        figures_dir / "04_synaptic_input_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Input analysis plots
    fig = plot_mitral_cell_spikes(
        input_spikes=input_spikes,
        dt=dt,
        n_neurons_plot=10,
        fraction=1.0,
    )
    fig.savefig(figures_dir / "05_mitral_cell_spikes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plot_feedforward_connectivity(
        feedforward_weights=feedforward_weights,
        input_cell_type_indices=input_cell_type_indices,
    )
    fig.savefig(
        figures_dir / "06_feedforward_connectivity.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Output analysis plots
    fig = plot_dp_network_spikes(
        output_spikes=output_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        n_neurons_plot=20,
        fraction=1.0,
    )
    fig.savefig(figures_dir / "07_dp_network_spikes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plot_firing_rate_distribution(
        output_spikes=output_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
    )
    fig.savefig(
        figures_dir / "08_firing_rate_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Prepare neuron_params for membrane voltage and current plots
    neuron_params = {}
    physiology = params["recurrent"]["physiology"]
    for idx, cell_name in enumerate(cell_type_names):
        cell_params = physiology[cell_name]
        neuron_params[idx] = {
            "threshold": cell_params["theta"],
            "rest": cell_params["U_reset"],
            "name": cell_name,
            "sign": 1 if "excit" in cell_name.lower() else -1,
        }

    # Detailed neuronal dynamics plots
    fig = plot_membrane_voltages(
        voltages=output_voltages,
        spikes=output_spikes,
        neuron_types=cell_type_indices,
        delta_t=dt,
        neuron_params=neuron_params,
        n_neurons_plot=5,
        fraction=1.0,
    )
    fig.savefig(figures_dir / "09_membrane_voltages.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Split currents into excitatory and inhibitory
    # Assuming index 0 = excitatory, index 1 = inhibitory
    I_exc = -output_currents[..., 0] - output_currents[..., 1]
    I_inh = -output_currents[..., 2]

    fig = plot_synaptic_currents(
        I_exc=I_exc,
        I_inh=I_inh,
        delta_t=dt,
        n_neurons_plot=5,
        fraction=1.0,
        show_total=True,
        neuron_types=cell_type_indices,
        neuron_params=neuron_params,
    )
    fig.savefig(figures_dir / "10_synaptic_currents.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Prepare synapse names for conductance plots
    recurrent_synapse_names = {}
    for cell_type in cell_type_names:
        recurrent_synapse_names[cell_type] = params["recurrent"]["synapses"][cell_type][
            "names"
        ]

    feedforward_synapse_names = {}
    for cell_type in input_cell_type_names:
        feedforward_synapse_names[cell_type] = params["feedforward"]["synapses"][
            cell_type
        ]["names"]

    fig = plot_synaptic_conductances(
        output_conductances=output_conductances,
        input_conductances=input_conductances,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        input_cell_type_names=input_cell_type_names,
        recurrent_synapse_names=recurrent_synapse_names,
        feedforward_synapse_names=feedforward_synapse_names,
        dt=dt,
        neuron_id=0,
        fraction=1.0,
    )
    fig.savefig(
        figures_dir / "11_synaptic_conductances.png", dpi=600, bbox_inches="tight"
    )
    plt.close(fig)

    # Firing statistics plots
    import torch

    output_spikes_tensor = torch.from_numpy(output_spikes).float()

    # Fano factor vs window size
    # Window sizes: 0.01s, 0.02s, 0.05s, 0.1s, 0.2s, 0.5s, 1.0s (converted to steps)
    window_sizes = [
        int(0.01 * 1000 / dt),  # 0.01s
        int(0.02 * 1000 / dt),  # 0.02s
        int(0.05 * 1000 / dt),  # 0.05s
        int(0.1 * 1000 / dt),  # 0.1s
        int(0.2 * 1000 / dt),  # 0.2s
        int(0.5 * 1000 / dt),  # 0.5s
        int(1.0 * 1000 / dt),  # 1.0s
    ]
    fig = plot_fano_factor_vs_window_size(
        spike_trains=output_spikes_tensor,
        window_sizes=window_sizes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
    )
    fig.savefig(
        figures_dir / "12_fano_factor_vs_window_size.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # CV histogram
    fig = plot_cv_histogram(
        spike_trains=output_spikes_tensor,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        bins=50,
    )
    fig.savefig(figures_dir / "13_cv_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ISI histogram
    fig = plot_isi_histogram(
        spike_trains=output_spikes_tensor,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        bins=100,
    )
    fig.savefig(figures_dir / "14_isi_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("All plots generated successfully!")

    # ============================================
    # STEP 7: Compute and Save Statistics to CSVs
    # ============================================
    # Create analysis directory if it doesn't exist
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing statistics and saving to {analysis_dir}...")

    # Convert cell_type_indices to torch tensor
    cell_type_indices_tensor = torch.from_numpy(cell_type_indices).long()

    # Convert output_voltages to torch tensor
    output_voltages_tensor = torch.from_numpy(output_voltages).float()

    # Compute firing rate statistics
    firing_rate_stats = compute_firing_rate_by_cell_type(
        spike_trains=output_spikes_tensor,
        cell_type_indices=cell_type_indices_tensor,
        duration=duration,
    )

    # Save firing rate statistics to CSV
    firing_rate_csv_path = analysis_dir / "firing_rate_statistics.csv"
    with open(firing_rate_csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "cell_type",
            "cell_type_name",
            "mean_firing_rate_hz",
            "std_firing_rate_hz",
            "n_silent_cells",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for cell_type_idx, stats in firing_rate_stats.items():
            writer.writerow(
                {
                    "cell_type": cell_type_idx,
                    "cell_type_name": cell_type_names[cell_type_idx],
                    "mean_firing_rate_hz": stats["mean_firing_rate_hz"],
                    "std_firing_rate_hz": stats["std_firing_rate_hz"],
                    "n_silent_cells": stats["n_silent_cells"],
                }
            )
    print(f"  Saved firing rate statistics to {firing_rate_csv_path}")

    # Compute CV statistics
    cv_stats = compute_cv_by_cell_type(
        spike_trains=output_spikes_tensor,
        cell_type_indices=cell_type_indices_tensor,
        dt=dt,
    )

    # Save CV statistics to CSV
    cv_csv_path = analysis_dir / "cv_statistics.csv"
    with open(cv_csv_path, "w", newline="") as csvfile:
        fieldnames = ["cell_type", "cell_type_name", "mean_cv", "std_cv"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for cell_type_idx, stats in cv_stats.items():
            writer.writerow(
                {
                    "cell_type": cell_type_idx,
                    "cell_type_name": cell_type_names[cell_type_idx],
                    "mean_cv": stats["mean_cv"],
                    "std_cv": stats["std_cv"],
                }
            )
    print(f"  Saved CV statistics to {cv_csv_path}")

    # Compute membrane potential statistics
    voltage_stats = compute_membrane_potential_by_cell_type(
        voltages=output_voltages_tensor,
        cell_type_indices=cell_type_indices_tensor,
    )

    # Save membrane potential statistics to CSV
    voltage_csv_path = analysis_dir / "voltage_statistics.csv"
    with open(voltage_csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "cell_type",
            "cell_type_name",
            "mean_of_means",
            "std_of_means",
            "mean_of_stds",
            "std_of_stds",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for cell_type_idx, stats in voltage_stats.items():
            writer.writerow(
                {
                    "cell_type": cell_type_idx,
                    "cell_type_name": cell_type_names[cell_type_idx],
                    "mean_of_means": stats["mean_of_means"],
                    "std_of_means": stats["std_of_means"],
                    "mean_of_stds": stats["mean_of_stds"],
                    "std_of_stds": stats["std_of_stds"],
                }
            )

    print("All statistics saved successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python conductance_based_Dp_plots.py <output_directory>")
        print("Example: python conductance_based_Dp_plots.py /path/to/output/folder")
        sys.exit(1)

    output_directory = sys.argv[1]
    main(output_directory)

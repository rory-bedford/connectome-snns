from visualization.connectivity import (
    plot_assembly_graph,
    plot_weighted_connectivity,
    plot_input_count_histogram,
    plot_synaptic_input_histogram,
    plot_feedforward_connectivity,
)
from visualization.neuronal_dynamics import (
    plot_membrane_voltages,
    plot_mitral_cell_spikes,
    plot_dp_network_spikes,
    plot_synaptic_conductances,
    plot_synaptic_currents,
)
from visualization.firing_statistics import (
    plot_fano_factor_vs_window_size,
    plot_cv_histogram,
    plot_isi_histogram,
    plot_firing_rate_distribution,
)

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for cluster environments
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import csv
from network_simulators.conductance_based.parameter_loader import (
    ConductanceBasedParams,
)
import toml
from analysis.firing_statistics import (
    compute_firing_rate_by_cell_type,
    compute_cv_by_cell_type,
)
from analysis.voltage_statistics import (
    compute_membrane_potential_by_cell_type,
)


def find_neuron_with_feedforward_inputs(
    feedforward_conductances: np.ndarray,
) -> int:
    """Find a neuron that receives feedforward inputs.

    Args:
        feedforward_conductances: Array of shape (batch, time, neurons, synapses)

    Returns:
        int: Index of a neuron that has non-zero feedforward conductances, or 0 if none found
    """
    # Sum over batch and time dimensions to get total conductance per neuron per synapse
    total_conductances = feedforward_conductances.sum(
        axis=(0, 1)
    )  # Shape: (neurons, synapses)

    # Find neurons with any non-zero feedforward conductance
    neurons_with_inputs = np.any(
        total_conductances > 1e-10, axis=1
    )  # Small threshold to avoid numerical issues

    if np.any(neurons_with_inputs):
        # Return the first neuron that has feedforward inputs
        return int(np.nonzero(neurons_with_inputs)[0][0])
    else:
        # Fallback to neuron 0 if no feedforward inputs found
        return 0


def generate_conductance_plots(
    spikes: np.ndarray,
    voltages: np.ndarray,
    conductances: np.ndarray,
    conductances_FF: np.ndarray,
    currents: np.ndarray,
    currents_FF: np.ndarray,
    input_spikes: np.ndarray,
    cell_type_indices: np.ndarray,
    input_cell_type_indices: np.ndarray,
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    weights: np.ndarray,
    feedforward_weights: np.ndarray,
    connectivity_graph: np.ndarray,
    num_assemblies: int,
    dt: float,
    neuron_params: dict,
    recurrent_synapse_names: dict,
    feedforward_synapse_names: dict,
    recurrent_g_bar_by_type: dict,
    feedforward_g_bar_by_type: dict,
) -> dict[str, plt.Figure]:
    """Generate all visualization plots for conductance-based simulation.

    Args:
        Same as homeostatic plots but for conductance-based model without training

    Returns:
        dict: Dictionary of figure objects keyed by plot name
    """
    figures = {}

    # Sum over rise/decay dimension (axis=3) to get total conductance
    # Check if conductances have the rise/decay dimension
    if len(conductances.shape) > 3 and conductances.shape[3] == 2:
        conductances = conductances.sum(axis=3)
    if len(conductances_FF.shape) > 3 and conductances_FF.shape[3] == 2:
        conductances_FF = conductances_FF.sum(axis=3)

    # Network structure plots
    figures["assembly_graph"] = plot_assembly_graph(
        connectivity_graph=connectivity_graph,
        cell_type_indices=cell_type_indices,
        num_assemblies=num_assemblies,
    )

    figures["weighted_connectivity"] = plot_weighted_connectivity(
        weights=weights,
        cell_type_indices=cell_type_indices,
        num_assemblies=num_assemblies,
    )

    figures["input_count_histogram"] = plot_input_count_histogram(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_cell_type_indices,
        cell_type_names=cell_type_names,
        input_cell_type_names=input_cell_type_names,
    )

    figures["synaptic_input_histogram"] = plot_synaptic_input_histogram(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_cell_type_indices,
        cell_type_names=cell_type_names,
        input_cell_type_names=input_cell_type_names,
        recurrent_g_bar_by_type=recurrent_g_bar_by_type,
        feedforward_g_bar_by_type=feedforward_g_bar_by_type,
    )

    figures["feedforward_connectivity"] = plot_feedforward_connectivity(
        feedforward_weights=feedforward_weights,
        input_cell_type_indices=input_cell_type_indices,
    )

    # Input analysis
    figures["mitral_cell_spikes"] = plot_mitral_cell_spikes(
        input_spikes=input_spikes,
        dt=dt,
    )

    # Output analysis
    figures["dp_network_spikes"] = plot_dp_network_spikes(
        output_spikes=spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
    )

    figures["firing_rate_distribution"] = plot_firing_rate_distribution(
        output_spikes=spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
    )

    figures["membrane_voltages"] = plot_membrane_voltages(
        voltages=voltages,
        spikes=spikes,
        neuron_types=cell_type_indices,
        delta_t=dt,
        neuron_params=neuron_params,
        n_neurons_plot=5,
        fraction=1,
    )

    # Synaptic currents - sum currents over synapse types
    total_currents = currents.sum(axis=3)
    total_currents_FF = currents_FF.sum(axis=3)

    # Split into excitatory (positive) and inhibitory (negative) components
    I_exc = total_currents.clip(min=0)
    I_inh = total_currents.clip(max=0)

    # Add feedforward currents
    I_exc += total_currents_FF.clip(min=0)
    I_inh += total_currents_FF.clip(max=0)

    figures["synaptic_currents"] = plot_synaptic_currents(
        I_exc=I_exc,
        I_inh=I_inh,
        delta_t=dt,
        n_neurons_plot=5,
        fraction=1.0,
        show_total=True,
        neuron_types=cell_type_indices,
        neuron_params=neuron_params,
    )

    # Find a neuron that actually receives feedforward inputs
    neuron_with_ff_inputs = find_neuron_with_feedforward_inputs(conductances_FF)

    figures["synaptic_conductances"] = plot_synaptic_conductances(
        output_conductances=conductances,
        input_conductances=conductances_FF,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        input_cell_type_names=input_cell_type_names,
        recurrent_synapse_names=recurrent_synapse_names,
        feedforward_synapse_names=feedforward_synapse_names,
        dt=dt,
        neuron_id=neuron_with_ff_inputs,
        fraction=1.0,
    )

    # Firing statistics
    window_sizes = [
        int(0.01 * 1000 / dt),
        int(0.02 * 1000 / dt),
        int(0.05 * 1000 / dt),
        int(0.1 * 1000 / dt),
        int(0.2 * 1000 / dt),
        int(0.5 * 1000 / dt),
        int(1.0 * 1000 / dt),
    ]
    figures["fano_factor"] = plot_fano_factor_vs_window_size(
        spike_trains=spikes,
        window_sizes=window_sizes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
    )

    figures["cv_histogram"] = plot_cv_histogram(
        spike_trains=spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        bins=50,
    )

    figures["isi_histogram"] = plot_isi_histogram(
        spike_trains=spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        bins=100,
    )

    return figures


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
        data = toml.load(f)
    params = ConductanceBasedParams(**data)

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
    input_currents = np.load(results_dir / "input_currents.npy")
    output_conductances = np.load(results_dir / "output_conductances.npy")
    input_conductances = np.load(results_dir / "input_conductances.npy")

    print(f"Generating plots and saving to {figures_dir}...")

    # Prepare pre-computed static data for plotting functions

    # Neuron parameters for plotting
    neuron_params = {}
    for idx, cell_name in enumerate(params.recurrent.cell_types.names):
        cell_params = params.recurrent.physiology[cell_name]
        neuron_params[idx] = {
            "threshold": cell_params.theta,
            "rest": cell_params.U_reset,
            "name": cell_name,
            "sign": 1 if "excit" in cell_name.lower() else -1,
        }

    # Synapse names for plotting
    recurrent_synapse_names = {}
    for cell_type in params.recurrent.cell_types.names:
        recurrent_synapse_names[cell_type] = params.recurrent.synapses[cell_type].names

    feedforward_synapse_names = {}
    for cell_type in params.feedforward.cell_types.names:
        feedforward_synapse_names[cell_type] = params.feedforward.synapses[
            cell_type
        ].names

    # Compute g_bar values for synaptic input histogram
    recurrent_g_bar_by_type = {}
    for cell_type in params.recurrent.cell_types.names:
        g_bar_values = params.recurrent.synapses[cell_type].g_bar
        recurrent_g_bar_by_type[cell_type] = sum(g_bar_values)

    feedforward_g_bar_by_type = {}
    for cell_type in params.feedforward.cell_types.names:
        g_bar_values = params.feedforward.synapses[cell_type].g_bar
        feedforward_g_bar_by_type[cell_type] = sum(g_bar_values)

    # Generate all plots using the new function-based approach
    figures = generate_conductance_plots(
        spikes=output_spikes,
        voltages=output_voltages,
        conductances=output_conductances,
        conductances_FF=input_conductances,
        currents=output_currents,
        currents_FF=input_currents,
        input_spikes=input_spikes,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_cell_type_indices,
        cell_type_names=params.recurrent.cell_types.names,
        input_cell_type_names=params.feedforward.cell_types.names,
        weights=weights,
        feedforward_weights=feedforward_weights,
        connectivity_graph=connectivity_graph,
        num_assemblies=params.recurrent.topology.num_assemblies,
        dt=params.simulation.dt,
        neuron_params=neuron_params,
        recurrent_synapse_names=recurrent_synapse_names,
        feedforward_synapse_names=feedforward_synapse_names,
        recurrent_g_bar_by_type=recurrent_g_bar_by_type,
        feedforward_g_bar_by_type=feedforward_g_bar_by_type,
    )

    # Save all figures
    plot_names = [
        ("assembly_graph", "01_assembly_graph.png"),
        ("weighted_connectivity", "02_weighted_connectivity.png"),
        ("input_count_histogram", "03_input_count_histogram.png"),
        ("synaptic_input_histogram", "04_synaptic_input_histogram.png"),
        ("mitral_cell_spikes", "05_mitral_cell_spikes.png"),
        ("feedforward_connectivity", "06_feedforward_connectivity.png"),
        ("dp_network_spikes", "07_dp_network_spikes.png"),
        ("firing_rate_distribution", "08_firing_rate_distribution.png"),
        ("membrane_voltages", "09_membrane_voltages.png"),
        ("synaptic_currents", "10_synaptic_currents.png"),
        ("synaptic_conductances", "11_synaptic_conductances.png"),
        ("fano_factor", "12_fano_factor_vs_window_size.png"),
        ("cv_histogram", "13_cv_histogram.png"),
        ("isi_histogram", "14_isi_histogram.png"),
    ]

    for plot_key, filename in plot_names:
        if plot_key in figures:
            fig = figures[plot_key]
            fig.savefig(figures_dir / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

    print("All plots generated successfully!")

    # ============================================
    # Compute and Save Statistics to CSVs
    # ============================================
    # Create analysis directory if it doesn't exist
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing statistics and saving to {analysis_dir}...")

    # Compute firing rate statistics
    firing_rate_stats = compute_firing_rate_by_cell_type(
        spike_trains=output_spikes,
        cell_type_indices=cell_type_indices,
        duration=params.simulation.duration,
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
                    "cell_type_name": params.recurrent.cell_types.names[cell_type_idx],
                    "mean_firing_rate_hz": stats["mean_firing_rate_hz"],
                    "std_firing_rate_hz": stats["std_firing_rate_hz"],
                    "n_silent_cells": stats["n_silent_cells"],
                }
            )
    print(f"  Saved firing rate statistics to {firing_rate_csv_path}")

    # Compute CV statistics
    cv_stats = compute_cv_by_cell_type(
        spike_trains=output_spikes,
        cell_type_indices=cell_type_indices,
        dt=params.simulation.dt,
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
                    "cell_type_name": params.recurrent.cell_types.names[cell_type_idx],
                    "mean_cv": stats["mean_cv"],
                    "std_cv": stats["std_cv"],
                }
            )
    print(f"  Saved CV statistics to {cv_csv_path}")

    # Compute membrane potential statistics
    voltage_stats = compute_membrane_potential_by_cell_type(
        voltages=output_voltages,
        cell_type_indices=cell_type_indices,
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
                    "cell_type_name": params.recurrent.cell_types.names[cell_type_idx],
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

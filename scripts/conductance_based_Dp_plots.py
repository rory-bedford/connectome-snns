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
from visualization.connectivity import (
    plot_assembly_graph,
    plot_weighted_connectivity,
    plot_input_count_histogram,
    plot_synaptic_input_histogram,
    plot_mitral_cell_spikes,
    plot_feedforward_connectivity,
    plot_dp_network_spikes,
    plot_firing_rate_distribution,
    plot_synaptic_conductances,
)
from visualization.neuronal_dynamics import (
    plot_membrane_voltages,
    plot_synaptic_currents,
)


def main(output_dir_path):
    """Main plotting function for Dp network simulation outputs.

    Loads parameters and generates all plots.

    Args:
        output_dir_path (str or Path): Path to directory containing simulation outputs
    """
    output_dir = Path(output_dir_path)

    # Load parameters from the TOML file
    params_file = output_dir / "parameters.toml"
    if not params_file.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_file}")

    print(f"Loading parameters from {params_file}...")
    with open(params_file, "r") as f:
        params = toml.load(f)

    # Load all data once
    connectivity_graph = np.load(output_dir / "connectivity_graph.npy")
    weights = np.load(output_dir / "weights.npy")
    feedforward_weights = np.load(output_dir / "feedforward_weights.npy")
    cell_type_indices = np.load(output_dir / "cell_type_indices.npy")
    input_cell_type_indices = np.load(output_dir / "input_cell_type_indices.npy")
    input_spikes = np.load(output_dir / "input_spikes.npy")
    output_spikes = np.load(output_dir / "output_spikes.npy")
    output_voltages = np.load(output_dir / "output_voltages.npy")
    output_currents = np.load(output_dir / "output_currents.npy")
    output_conductances = np.load(output_dir / "output_conductances.npy")
    input_conductances = np.load(output_dir / "input_conductances.npy")

    # Extract commonly used parameters
    dt = params["simulation"]["dt"]
    duration = params["simulation"]["duration"]
    num_assemblies = int(params["recurrent"]["topology"]["num_assemblies"])
    cell_type_names = params["recurrent"]["cell_types"]["names"]
    input_cell_type_names = params["feedforward"]["cell_types"]["names"]

    print("Generating plots...")

    # Network structure plots
    fig = plot_assembly_graph(
        connectivity_graph=connectivity_graph,
        cell_type_indices=cell_type_indices,
        num_assemblies=num_assemblies,
    )
    fig.savefig(output_dir / "01_assembly_graph.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plot_weighted_connectivity(
        weights=weights,
        cell_type_indices=cell_type_indices,
        num_assemblies=num_assemblies,
    )
    fig.savefig(
        output_dir / "02_weighted_connectivity.png", dpi=300, bbox_inches="tight"
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
        output_dir / "03_input_count_histogram.png", dpi=300, bbox_inches="tight"
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
        output_dir / "04_synaptic_input_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Input analysis plots
    fig = plot_mitral_cell_spikes(
        input_spikes=input_spikes,
        dt=dt,
        duration=duration,
    )
    fig.savefig(output_dir / "05_mitral_cell_spikes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plot_feedforward_connectivity(
        feedforward_weights=feedforward_weights,
        input_cell_type_indices=input_cell_type_indices,
    )
    fig.savefig(
        output_dir / "06_feedforward_connectivity.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Output analysis plots
    fig = plot_dp_network_spikes(
        output_spikes=output_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        duration=duration,
    )
    fig.savefig(output_dir / "07_dp_network_spikes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plot_firing_rate_distribution(
        output_spikes=output_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        duration=duration,
    )
    fig.savefig(
        output_dir / "08_firing_rate_distribution.png", dpi=300, bbox_inches="tight"
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
        duration=duration,
        neuron_params=neuron_params,
        n_neurons_plot=5,
        fraction=1,
    )
    fig.savefig(output_dir / "09_membrane_voltages.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Split currents into excitatory and inhibitory
    # Assuming index 0 = excitatory, index 1 = inhibitory
    I_exc = -output_currents[..., 0] - output_currents[..., 1]
    I_inh = -output_currents[..., 2]

    fig = plot_synaptic_currents(
        I_exc=I_exc,
        I_inh=I_inh,
        delta_t=dt,
        duration=duration,
        n_neurons_plot=5,
        fraction=1.0,
        neuron_types=cell_type_indices,
        neuron_params=neuron_params,
    )
    fig.savefig(output_dir / "10_synaptic_currents.png", dpi=600, bbox_inches="tight")
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
        duration=duration,
        n_neurons_plot=5,
        fraction=1.0,
    )
    fig.savefig(
        output_dir / "11_synaptic_conductances.png", dpi=600, bbox_inches="tight"
    )
    plt.close(fig)

    print("All plots generated successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python conductance_based_Dp_plots.py <output_directory>")
        print("Example: python conductance_based_Dp_plots.py /path/to/output/folder")
        sys.exit(1)

    output_directory = sys.argv[1]
    main(output_directory)

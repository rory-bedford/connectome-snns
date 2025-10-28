"""
Plotting functions for Dp network simulation outputs

This script contains all the plotting and visualization logic for analyzing
the outputs from simulate_Dp.py. It generates comprehensive visualizations
of network structure, dynamics, and activity patterns.

Overview:
1. Network structure plots: assembly graphs and connectivity matrices
2. Input analysis: mitral cell spike trains and feedforward connectivity
3. Output analysis: spike trains, firing rates, membrane voltages, and currents

Usage:
    python simulate_Dp_plots.py <output_directory>
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import toml
from pathlib import Path
import sys
from visualization.neuronal_dynamics import (
    plot_membrane_voltages,
    plot_synaptic_currents,
)


def plot_assembly_graph(params: dict, output_dir: Path) -> None:
    """Plot the assembly graph structure.

    Args:
        params (dict): Parameters dictionary from TOML file
        output_dir (Path): Directory containing simulation outputs
    """
    # Load data
    connectivity_graph = np.load(output_dir / "connectivity_graph.npy")
    cell_type_indices = np.load(output_dir / "cell_type_indices.npy")
    num_assemblies = int(params["recurrent"]["topology"]["num_assemblies"])

    plot_num_assemblies = 2  # Number of assemblies to display
    num_neurons = connectivity_graph.shape[0]
    neurons_per_assembly = num_neurons // num_assemblies
    plot_size_neurons = neurons_per_assembly * plot_num_assemblies

    # Apply signs based on source cell type: 0=excitatory (+1), 1=inhibitory (-1)
    cell_type_signs = np.where(cell_type_indices == 0, 1, -1)
    signed_connectivity = connectivity_graph * cell_type_signs[:, np.newaxis]

    # Fixed size in inches for the heatmap
    heatmap_inches = 8
    fig, ax = plt.subplots(figsize=(heatmap_inches * 1.3, heatmap_inches))

    im = ax.imshow(
        signed_connectivity[:plot_size_neurons, :plot_size_neurons],
        cmap="bwr",
        vmin=-1,
        vmax=1,
        aspect="equal",
    )

    # Force the axes to be square first
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.height, pos.height])

    # Add colorbar after positioning
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["Inhibitory (-1)", "No connection (0)", "Excitatory (+1)"])

    ax.set_title(
        f"Assembly Graph Structure (showing {plot_num_assemblies}/{num_assemblies} assemblies)"
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(output_dir / "01_assembly_graph.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_weighted_connectivity(params: dict, output_dir: Path) -> None:
    """Plot the weighted connectivity matrix.

    Args:
        params (dict): Parameters dictionary from TOML file
        output_dir (Path): Directory containing simulation outputs
    """
    # Load data
    weights = np.load(output_dir / "weights.npy")
    cell_type_indices = np.load(output_dir / "cell_type_indices.npy")
    num_assemblies = int(params["recurrent"]["topology"]["num_assemblies"])

    plot_num_assemblies = 2  # Number of assemblies to display
    num_neurons = weights.shape[0]
    neurons_per_assembly = num_neurons // num_assemblies
    plot_size_neurons = neurons_per_assembly * plot_num_assemblies

    # Apply signs based on source cell type: 0=excitatory (+1), 1=inhibitory (-1)
    cell_type_signs = np.where(cell_type_indices == 0, 1, -1)
    signed_weights = weights * cell_type_signs[:, np.newaxis]

    # Fixed size in inches for the heatmap
    heatmap_inches = 8
    fig, ax = plt.subplots(figsize=(heatmap_inches * 1.3, heatmap_inches))

    im = ax.imshow(
        signed_weights[:plot_size_neurons, :plot_size_neurons],
        cmap="bwr",
        vmin=-1,
        vmax=1,
        aspect="equal",
    )

    # Force the axes to be square first
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.height, pos.height])

    # Add colorbar after positioning
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.ax.set_yticklabels(["-1", "-0.5", "0", "+0.5", "+1"])

    ax.set_title(
        f"Weighted Connectivity Matrix (showing {plot_num_assemblies}/{num_assemblies} assemblies)"
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(
        output_dir / "02_weighted_connectivity.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_synaptic_input_histogram(params: dict, output_dir: Path) -> None:
    """Plot histogram of total synaptic input to each neuron, separated by presynaptic cell type.

    Shows the distribution of input conductances from each presynaptic cell type
    (both recurrent and feedforward) across all postsynaptic neurons in the network.

    Args:
        params (dict): Parameters dictionary from TOML file
        output_dir (Path): Directory containing simulation outputs
    """
    # Load data
    weights = np.load(output_dir / "weights.npy")
    feedforward_weights = np.load(output_dir / "feedforward_weights.npy")
    cell_type_indices = np.load(output_dir / "cell_type_indices.npy")
    input_cell_type_indices = np.load(output_dir / "input_cell_type_indices.npy")

    # Extract cell type names
    cell_type_names = params["recurrent"]["cell_types"]["names"]
    input_cell_type_names = params["feedforward"]["cell_types"]["names"]

    # Prepare data for histogram
    unique_recurrent_types = np.unique(cell_type_indices)
    unique_feedforward_types = np.unique(input_cell_type_indices)

    scaled_conductances_by_type = []
    subplot_titles = []

    # Recurrent connections
    for cell_type_idx in unique_recurrent_types:
        mask = cell_type_indices == cell_type_idx
        cell_type_name = cell_type_names[cell_type_idx]

        # Get g_bar values for this cell type and sum them
        g_bar_values = params["recurrent"]["synapses"][cell_type_name]["g_bar"]
        total_g_bar = sum(g_bar_values)

        # Sum incoming weights and multiply by total g_bar
        conductances = weights[mask, :].sum(axis=0) * total_g_bar
        scaled_conductances_by_type.append(conductances)
        subplot_titles.append(f"Recurrent: {cell_type_name}")

    # Feedforward connections
    for cell_type_idx in unique_feedforward_types:
        mask = input_cell_type_indices == cell_type_idx
        cell_type_name = input_cell_type_names[cell_type_idx]

        # Get g_bar values for this cell type and sum them
        g_bar_values = params["feedforward"]["synapses"][cell_type_name]["g_bar"]
        total_g_bar = sum(g_bar_values)

        # Sum incoming weights and multiply by total g_bar
        conductances = feedforward_weights[mask, :].sum(axis=0) * total_g_bar
        scaled_conductances_by_type.append(conductances)
        subplot_titles.append(f"Feedforward: {cell_type_name}")

    # Plot
    n_types = len(scaled_conductances_by_type)

    # Calculate global min/max for shared axes (excluding zeros)
    all_positive = np.concatenate([c[c > 0] for c in scaled_conductances_by_type])
    global_x_min = all_positive.min() if len(all_positive) > 0 else 1e-3
    global_x_max = all_positive.max() if len(all_positive) > 0 else 1.0

    # Create logarithmically-spaced bins
    log_bins = np.logspace(np.log10(global_x_min), np.log10(global_x_max), 21)

    # Create subplots (one per cell type)
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 4), sharey=True)
    if n_types == 1:
        axes = [axes]

    # Calculate global y-max across all histograms
    max_count = 0
    for conductances in scaled_conductances_by_type:
        counts, _ = np.histogram(conductances, bins=log_bins)
        max_count = max(max_count, counts.max())

    # Plot each cell type
    for i, (conductances, title) in enumerate(
        zip(scaled_conductances_by_type, subplot_titles)
    ):
        ax = axes[i]
        mean_conductance = conductances.mean()

        ax.hist(
            conductances, bins=log_bins, color="#0000FF", edgecolor="black", alpha=0.6
        )
        ax.axvline(
            mean_conductance,
            color="#FF0000",
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label=f"Mean = {mean_conductance:.2f} nS",
        )
        ax.set_title(title)
        ax.set_xlabel("Conductance (nS)")
        ax.set_xscale("log")
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(0, max_count * 1.1)
        ax.legend()

    axes[0].set_ylabel("Number of Postsynaptic Neurons")
    fig.suptitle("Input Conductances by Presynaptic Cell Type", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(
        output_dir / "03_synaptic_input_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_mitral_cell_spikes(params: dict, output_dir: Path) -> None:
    """Plot sample mitral cell spike trains.

    Args:
        params (dict): Parameters dictionary from TOML file
        output_dir (Path): Directory containing simulation outputs
    """
    # Load data
    input_spikes = np.load(output_dir / "input_spikes.npy")
    dt = params["simulation"]["dt"]
    duration = params["simulation"]["duration"]

    n_neurons_plot = 10
    fraction = 1.0  # fraction of duration to plot
    fig, ax = plt.subplots(figsize=(12, 4))
    spike_times, neuron_ids = np.where(input_spikes[0, :, :n_neurons_plot])
    ax.scatter(spike_times * dt * 1e-3, neuron_ids, s=1, color="black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Sample Mitral Cell Spike Trains")
    ax.set_ylim(-0.5, n_neurons_plot - 0.5)
    ax.set_yticks(range(n_neurons_plot))
    ax.set_xlim(0, duration * 1e-3 * fraction)
    plt.tight_layout()
    plt.savefig(output_dir / "04_mitral_cell_spikes.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feedforward_connectivity(params: dict, output_dir: Path) -> None:
    """Plot feedforward connectivity matrix.

    Args:
        params (dict): Parameters dictionary from TOML file
        output_dir (Path): Directory containing simulation outputs
    """
    # Load data
    feedforward_weights = np.load(output_dir / "feedforward_weights.npy")
    input_cell_type_indices = np.load(output_dir / "input_cell_type_indices.npy")

    plot_fraction = 0.1  # Fraction of neurons to display
    n_input, n_output = feedforward_weights.shape
    n_input_plot = int(n_input * plot_fraction)
    n_output_plot = int(n_output * plot_fraction)

    # Apply signs based on source cell type: 0=excitatory (+1), 1=inhibitory (-1)
    input_cell_type_signs = np.where(input_cell_type_indices == 0, 1, -1)
    signed_feedforward_weights = (
        feedforward_weights * input_cell_type_signs[:, np.newaxis]
    )

    # Make plot bigger - use fixed large size
    plot_width = 14
    plot_height = plot_width * n_input_plot / n_output_plot

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    im = ax.imshow(
        signed_feedforward_weights[:n_input_plot, :n_output_plot],
        cmap="bwr",
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.ax.set_yticklabels(["-1", "-0.5", "0", "+0.5", "+1"])
    ax.set_title(
        f"Feedforward Connectivity Matrix ({n_input_plot}/{n_input} mitral cells × {n_output_plot}/{n_output} Dp neurons)"
    )
    ax.set_xlabel("Target Dp Neurons")
    ax.set_ylabel("Source Mitral Cells")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(
        output_dir / "05_feedforward_connectivity.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_dp_network_spikes(params: dict, output_dir: Path) -> None:
    """Plot sample Dp network spike trains colored by cell type.

    Args:
        params (dict): Parameters dictionary from TOML file.
        output_dir (Path): Directory containing simulation outputs.
    """
    # Load data
    output_spikes = np.load(output_dir / "output_spikes.npy")
    cell_type_indices = np.load(output_dir / "cell_type_indices.npy")
    dt = params["simulation"]["dt"]
    duration = params["simulation"]["duration"]

    # Get cell type names from parameters
    cell_type_names = params["recurrent"]["cell_types"]["names"]
    n_cell_types = len(cell_type_names)

    # Define colors: red, blue, then tab10 colormap for additional types
    base_colors = ["#FF0000", "#0000FF"]
    if n_cell_types <= 2:
        colors_map = base_colors[:n_cell_types]
    else:
        cmap = plt.cm.get_cmap("tab10")
        additional_colors = [cmap(i) for i in range(n_cell_types - 2)]
        colors_map = base_colors + additional_colors

    n_neurons_plot = 20
    fraction = 1.0  # fraction of duration to plot
    fig, ax = plt.subplots(figsize=(12, 6))
    spike_times, neuron_ids = np.where(output_spikes[0, :, :n_neurons_plot])

    # Color spikes by cell type
    spike_colors = [colors_map[cell_type_indices[nid]] for nid in neuron_ids]

    ax.scatter(spike_times * dt * 1e-3, neuron_ids, s=1, c=spike_colors)

    # Create legend with cell type names
    legend_elements = [
        Patch(facecolor=colors_map[i], label=cell_type_names[i])
        for i in range(n_cell_types)
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Sample Dp Network Spike Trains (colored by cell type)")
    ax.set_ylim(-0.5, n_neurons_plot - 0.5)
    ax.set_yticks(range(n_neurons_plot))
    ax.set_xlim(0, duration * 1e-3 * fraction)
    plt.tight_layout()
    plt.savefig(output_dir / "06_dp_network_spikes.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_firing_rate_distribution(params: dict, output_dir: Path) -> None:
    """Plot distribution of firing rates in the Dp network by cell type.

    Args:
        params (dict): Parameters dictionary from TOML file.
        output_dir (Path): Directory containing simulation outputs.
    """
    # Load data
    output_spikes = np.load(output_dir / "output_spikes.npy")
    cell_type_indices = np.load(output_dir / "cell_type_indices.npy")
    duration = params["simulation"]["duration"]

    # Get cell type names from parameters
    cell_type_names = params["recurrent"]["cell_types"]["names"]
    n_cell_types = len(cell_type_names)

    # Define colors: red, blue, then tab10 colormap for additional types
    base_colors = ["#FF0000", "#0000FF"]
    if n_cell_types <= 2:
        colors_map = base_colors[:n_cell_types]
    else:
        cmap = plt.cm.get_cmap("tab10")
        additional_colors = [cmap(i) for i in range(n_cell_types - 2)]
        colors_map = base_colors + additional_colors

    # Calculate firing rates (spikes per second)
    spike_counts = output_spikes[0].sum(axis=0)  # Total spikes per neuron
    firing_rates = spike_counts / (duration * 1e-3)  # Convert duration from ms to s

    # Filter out zero firing rates for log scale
    firing_rates_nonzero = firing_rates[firing_rates > 0]

    # Create subplots
    fig, axes = plt.subplots(
        1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
    )
    if n_cell_types == 1:
        axes = [axes]

    # Calculate global x-axis limits
    if len(firing_rates_nonzero) > 0:
        global_x_min = firing_rates_nonzero.min()
        global_x_max = firing_rates_nonzero.max()
    else:
        global_x_min = 0.1
        global_x_max = 10.0

    # Create logarithmically-spaced bins
    log_bins = np.logspace(np.log10(global_x_min), np.log10(global_x_max), 30)

    # Plot each cell type
    for i in range(n_cell_types):
        ax = axes[i]
        cell_type_rates = firing_rates[cell_type_indices == i]
        cell_type_rates_nonzero = cell_type_rates[cell_type_rates > 0]

        ax.hist(
            cell_type_rates_nonzero,
            bins=log_bins,
            alpha=0.6,
            color=colors_map[i],
            label=f"n={len(cell_type_rates)}",
            edgecolor="black",
        )

        # Add mean line
        if len(cell_type_rates_nonzero) > 0:
            mean_rate = cell_type_rates_nonzero.mean()
            ax.axvline(
                mean_rate,
                alpha=0.6,
                color=colors_map[i],
                linestyle="--",
                linewidth=2,
                label=f"Mean = {mean_rate:.2f} Hz",
            )

        ax.set_xlabel("Firing Rate (Hz)")
        ax.set_title(cell_type_names[i])
        ax.set_xscale("log")
        ax.set_xlim(global_x_min, global_x_max)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Number of Neurons")
    fig.suptitle("Firing Rate Distribution", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(
        output_dir / "07_firing_rate_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_membrane_voltages_wrapper(params: dict, output_dir: Path) -> None:
    """Wrapper for neuronal_dynamics.plot_membrane_voltages that loads data and converts params.

    Args:
        params (dict): Parameters dictionary from TOML file
        output_dir (Path): Directory containing simulation outputs
    """
    # Load data
    output_voltages = np.load(output_dir / "output_voltages.npy")
    output_spikes = np.load(output_dir / "output_spikes.npy")
    cell_type_indices = np.load(output_dir / "cell_type_indices.npy")

    # Extract simulation parameters
    dt = params["simulation"]["dt"]
    duration = params["simulation"]["duration"]

    # Get cell type names and build neuron_params dict
    cell_type_names = params["recurrent"]["cell_types"]["names"]
    physiology = params["recurrent"]["physiology"]

    neuron_params = {}
    for idx, cell_name in enumerate(cell_type_names):
        cell_params = physiology[cell_name]
        neuron_params[idx] = {
            "threshold": cell_params["theta"],
            "rest": cell_params["U_reset"],
            "name": cell_name,
            "sign": 1 if "excit" in cell_name.lower() else -1,
        }

    # Call the neuronal_dynamics function
    plot_membrane_voltages(
        voltages=output_voltages,
        spikes=output_spikes,
        neuron_types=cell_type_indices,
        delta_t=dt,
        duration=duration,
        neuron_params=neuron_params,
        n_neurons_plot=5,
        fraction=1,
        save_path=str(output_dir / "08_membrane_voltages.png"),
    )


def plot_synaptic_currents_wrapper(params: dict, output_dir: Path) -> None:
    """Wrapper for neuronal_dynamics.plot_synaptic_currents that loads data and converts params.

    Args:
        params (dict): Parameters dictionary from TOML file
        output_dir (Path): Directory containing simulation outputs
    """
    # Load data
    output_currents = np.load(output_dir / "output_currents.npy")
    cell_type_indices = np.load(output_dir / "cell_type_indices.npy")

    # Extract simulation parameters
    dt = params["simulation"]["dt"]
    duration = params["simulation"]["duration"]

    # Get cell type names and build neuron_params dict
    cell_type_names = params["recurrent"]["cell_types"]["names"]
    physiology = params["recurrent"]["physiology"]

    neuron_params = {}
    for idx, cell_name in enumerate(cell_type_names):
        cell_params = physiology[cell_name]
        neuron_params[idx] = {
            "threshold": cell_params["theta"],
            "rest": cell_params["U_reset"],
            "name": cell_name,
            "sign": 1 if "excit" in cell_name.lower() else -1,
        }

    # Split into excitatory and inhibitory currents
    # Assuming index 0 = excitatory, index 1 = inhibitory
    I_exc = output_currents[..., 0]
    I_inh = output_currents[..., 1]

    # Call the neuronal_dynamics function
    plot_synaptic_currents(
        I_exc=I_exc,
        I_inh=I_inh,
        delta_t=dt,
        duration=duration,
        n_neurons_plot=5,
        fraction=1.0,
        neuron_types=cell_type_indices,
        neuron_params=neuron_params,
        save_path=str(output_dir / "09_synaptic_currents.png"),
    )


def plot_synaptic_conductances(params: dict, output_dir: Path) -> None:
    """Plot synaptic conductances for sample neurons, separated by synapse type.

    Shows both recurrent and feedforward conductances in separate subplots for each
    neuron, with all synapse types displayed in different colors.

    Args:
        params (dict): Parameters dictionary from TOML file
        output_dir (Path): Directory containing simulation outputs
    """
    # Load data
    output_conductances = np.load(output_dir / "output_conductances.npy")
    input_conductances = np.load(output_dir / "input_conductances.npy")
    cell_type_indices = np.load(output_dir / "cell_type_indices.npy")

    # Extract simulation parameters
    dt = params["simulation"]["dt"]
    duration = params["simulation"]["duration"]

    # Get synapse type information
    cell_type_names = params["recurrent"]["cell_types"]["names"]
    input_cell_type_names = params["feedforward"]["cell_types"]["names"]

    # Build list of all synapse types (recurrent + feedforward)
    all_synapse_names = []
    all_synapse_labels = []

    # Recurrent synapse types
    for cell_type in cell_type_names:
        synapse_names = params["recurrent"]["synapses"][cell_type]["names"]
        for syn_name in synapse_names:
            all_synapse_names.append(f"{cell_type}_{syn_name}")
            all_synapse_labels.append(f"{cell_type} {syn_name}")

    # Feedforward synapse types
    ff_start_idx = len(all_synapse_names)
    for cell_type in input_cell_type_names:
        synapse_names = params["feedforward"]["synapses"][cell_type]["names"]
        for syn_name in synapse_names:
            all_synapse_names.append(f"{cell_type}_{syn_name}")
            all_synapse_labels.append(f"{cell_type} {syn_name}")

    # Plotting parameters
    n_neurons_plot = 5
    fraction = 1.0

    # Create time array
    n_steps = output_conductances.shape[1]
    n_steps_plot = int(n_steps * fraction)
    time_axis = np.arange(n_steps_plot) * dt * 1e-3  # Convert to seconds

    # Color palette for different synapse types
    cmap = plt.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(len(all_synapse_names))]

    # Create figure with subplots for each neuron
    fig, axes = plt.subplots(
        n_neurons_plot, 1, figsize=(14, 2.5 * n_neurons_plot), sharex=True
    )
    if n_neurons_plot == 1:
        axes = [axes]

    for neuron_id in range(n_neurons_plot):
        ax = axes[neuron_id]

        # Get cell type for this neuron
        cell_type_idx = cell_type_indices[neuron_id]
        cell_name = cell_type_names[cell_type_idx]

        # Plot recurrent conductances
        for syn_idx in range(output_conductances.shape[3]):
            g_trace = output_conductances[0, :n_steps_plot, neuron_id, syn_idx]
            ax.plot(
                time_axis,
                g_trace,
                linewidth=1.0,
                color=colors[syn_idx],
                label=all_synapse_labels[syn_idx],
                alpha=0.8,
            )

        # Plot feedforward conductances
        for syn_idx in range(input_conductances.shape[3]):
            g_trace = input_conductances[0, :n_steps_plot, neuron_id, syn_idx]
            ax.plot(
                time_axis,
                g_trace,
                linewidth=1.0,
                color=colors[ff_start_idx + syn_idx],
                label=all_synapse_labels[ff_start_idx + syn_idx],
                alpha=0.8,
            )

        # Formatting
        ax.set_ylabel(f"Neuron {neuron_id} ({cell_name})\nConductance (nS)", fontsize=9)
        ax.set_xlim(0, duration * 1e-3 * fraction)
        ax.set_ylim(0, 5)  # Fixed y-axis from 0 to 5 nS
        ax.grid(True, alpha=0.3)

        # Add legend to first subplot only
        if neuron_id == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Synaptic Conductances (First {n_neurons_plot} Neurons)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        output_dir / "10_synaptic_conductances.png", dpi=600, bbox_inches="tight"
    )
    plt.close()


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

    print("Generating plots...")

    # Network structure plots
    plot_assembly_graph(params, output_dir)
    plot_weighted_connectivity(params, output_dir)
    plot_synaptic_input_histogram(params, output_dir)

    # Input analysis plots
    plot_mitral_cell_spikes(params, output_dir)
    plot_feedforward_connectivity(params, output_dir)

    # Output analysis plots
    plot_dp_network_spikes(params, output_dir)
    plot_firing_rate_distribution(params, output_dir)

    # Detailed neuronal dynamics plots using visualization module functions
    plot_membrane_voltages_wrapper(params, output_dir)
    plot_synaptic_currents_wrapper(params, output_dir)
    plot_synaptic_conductances(params, output_dir)

    print("All plots generated successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simulate_Dp_plots.py <output_directory>")
        print("Example: python simulate_Dp_plots.py /path/to/output/folder")
        sys.exit(1)

    output_directory = sys.argv[1]
    main(output_directory)

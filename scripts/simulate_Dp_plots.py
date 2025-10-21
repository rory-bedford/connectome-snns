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
import numpy as np
import toml
from pathlib import Path
import sys
from visualization import plot_membrane_voltages, plot_synaptic_currents


def plot_assembly_graph(connectivity_graph, num_assemblies, save_path):
    """Plot the assembly graph structure.

    Args:
        connectivity_graph (np.ndarray): Binary connectivity matrix
        num_assemblies (int): Number of assemblies in the network
        save_path (Path): Path to save the plot
    """
    plot_num_assemblies = 2  # Number of assemblies to display
    num_neurons = connectivity_graph.shape[0]
    neurons_per_assembly = num_neurons // num_assemblies
    plot_size_neurons = neurons_per_assembly * plot_num_assemblies

    # Fixed size in inches for the heatmap
    heatmap_inches = 8  # Bigger fixed size
    fig, ax = plt.subplots(
        figsize=(heatmap_inches * 1.3, heatmap_inches)
    )  # Extra width for colorbar

    im = ax.imshow(
        connectivity_graph[:plot_size_neurons, :plot_size_neurons],
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

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_weighted_connectivity(weights, num_assemblies, save_path):
    """Plot the weighted connectivity matrix.

    Args:
        weights (np.ndarray): Weighted connectivity matrix
        num_assemblies (int): Number of assemblies in the network
        save_path (Path): Path to save the plot
    """
    plot_num_assemblies = 2  # Number of assemblies to display
    num_neurons = weights.shape[0]
    neurons_per_assembly = num_neurons // num_assemblies
    plot_size_neurons = neurons_per_assembly * plot_num_assemblies

    # Fixed size in inches for the heatmap (same as unweighted)
    heatmap_inches = 8  # Bigger fixed size
    fig, ax = plt.subplots(
        figsize=(heatmap_inches * 1.3, heatmap_inches)
    )  # Extra width for colorbar

    im = ax.imshow(
        weights[:plot_size_neurons, :plot_size_neurons],
        cmap="bwr",
        vmin=-10,
        vmax=10,
        aspect="equal",
    )

    # Force the axes to be square first
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.height, pos.height])

    # Add colorbar after positioning
    cbar = plt.colorbar(im, ax=ax, ticks=[-10, -5, 0, 5, 10])
    cbar.ax.set_yticklabels(["-10", "-5", "0", "+5", "+10"])

    ax.set_title(
        f"Weighted Connectivity Matrix (showing {plot_num_assemblies}/{num_assemblies} assemblies)"
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_synaptic_input_histogram(weights, save_path):
    """Plot histogram of total synaptic input to each neuron.

    Args:
        weights (np.ndarray): Weighted connectivity matrix
        save_path (Path): Path to save the plot
    """
    synaptic_inputs = weights.sum(axis=0)
    mean_input = synaptic_inputs.mean()

    fig, ax = plt.subplots()
    ax.hist(
        synaptic_inputs, bins=20, color="#0000FF", edgecolor="black", alpha=0.6
    )  # Blue from bwr colormap
    ax.axvline(
        mean_input,
        color="#FF0000",
        linestyle="--",
        linewidth=2,
        alpha=0.6,
        label=f"Mean = {mean_input:.2f}",
    )  # Red from bwr
    ax.set_title("Histogram of Total Synaptic Input to Each Neuron")
    ax.set_xlabel("Total Synaptic Input")
    ax.set_ylabel("Number of Neurons")
    ax.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_mitral_cell_spikes(input_spikes, delta_t, duration, save_path):
    """Plot sample mitral cell spike trains.

    Args:
        input_spikes (np.ndarray): Input spike array
        delta_t (float): Time step in ms
        duration (float): Simulation duration in ms
        save_path (Path): Path to save the plot
    """
    n_neurons_plot = 10
    fraction = 1.0  # fraction of duration to plot
    fig, ax = plt.subplots(figsize=(12, 4))
    spike_times, neuron_ids = np.where(input_spikes[0, :, :n_neurons_plot])
    ax.scatter(spike_times * delta_t * 1e-3, neuron_ids, s=1, color="black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Sample Mitral Cell Spike Trains")
    ax.set_ylim(-0.5, n_neurons_plot - 0.5)
    ax.set_yticks(range(n_neurons_plot))
    ax.set_xlim(0, duration * 1e-3 * fraction)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feedforward_connectivity(feedforward_weights, save_path):
    """Plot feedforward connectivity matrix.

    Args:
        feedforward_weights (np.ndarray): Feedforward connectivity weights
        save_path (Path): Path to save the plot
    """
    plot_fraction = 0.1  # Fraction of neurons to display
    n_input, n_output = feedforward_weights.shape
    n_input_plot = int(n_input * plot_fraction)
    n_output_plot = int(n_output * plot_fraction)

    # Make plot bigger - use fixed large size
    plot_width = 14
    plot_height = plot_width * n_input_plot / n_output_plot

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    im = ax.imshow(
        feedforward_weights[:n_input_plot, :n_output_plot],
        cmap="bwr",
        vmin=-2,
        vmax=2,
        aspect="auto",
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, ticks=[-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels(["-2", "-1", "0", "+1", "+2"])
    ax.set_title(
        f"Feedforward Connectivity Matrix ({n_input_plot}/{n_input} mitral cells × {n_output_plot}/{n_output} Dp neurons)"
    )
    ax.set_xlabel("Target Dp Neurons")
    ax.set_ylabel("Source Mitral Cells")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dp_network_spikes(output_spikes, delta_t, duration, save_path):
    """Plot sample Dp network spike trains.

    Args:
        output_spikes (np.ndarray): Output spike array
        delta_t (float): Time step in ms
        duration (float): Simulation duration in ms
        save_path (Path): Path to save the plot
    """
    n_neurons_plot = 10
    fraction = 1.0  # fraction of duration to plot
    fig, ax = plt.subplots(figsize=(12, 4))
    spike_times, neuron_ids = np.where(output_spikes[0, :, :n_neurons_plot])
    ax.scatter(spike_times * delta_t * 1e-3, neuron_ids, s=1, color="black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Sample Dp Network Spike Trains")
    ax.set_ylim(-0.5, n_neurons_plot - 0.5)
    ax.set_yticks(range(n_neurons_plot))
    ax.set_xlim(0, duration * 1e-3 * fraction)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_firing_rate_distribution(output_spikes, neuron_types, duration, save_path):
    """Plot distribution of firing rates in the Dp network.

    Args:
        output_spikes (np.ndarray): Output spike array
        neuron_types (np.ndarray): Neuron type assignments (+1/-1)
        duration (float): Simulation duration in ms
        save_path (Path): Path to save the plot
    """
    # Calculate firing rates (spikes per second)
    spike_counts = output_spikes[0].sum(axis=0)  # Total spikes per neuron
    firing_rates = spike_counts / (duration * 1e-3)  # Convert duration from ms to s

    # Separate firing rates by neuron type
    excitatory_rates = firing_rates[neuron_types == 1]
    inhibitory_rates = firing_rates[neuron_types == -1]

    # Create histogram with bwr colormap colors
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, max(firing_rates.max(), 1), 30)

    ax.hist(
        excitatory_rates,
        bins=bins,
        alpha=0.6,
        color="#0000FF",
        label=f"Excitatory (n={len(excitatory_rates)})",
        edgecolor="black",
    )  # Blue from bwr
    ax.hist(
        inhibitory_rates,
        bins=bins,
        alpha=0.6,
        color="#FF0000",
        label=f"Inhibitory (n={len(inhibitory_rates)})",
        edgecolor="black",
    )  # Red from bwr

    # Add mean lines
    ax.axvline(
        excitatory_rates.mean(),
        alpha=0.6,
        color="#0000FF",
        linestyle="--",
        linewidth=2,
        label=f"E mean = {excitatory_rates.mean():.2f} Hz",
    )
    ax.axvline(
        inhibitory_rates.mean(),
        alpha=0.6,
        color="#FF0000",
        linestyle="--",
        linewidth=2,
        label=f"I mean = {inhibitory_rates.mean():.2f} Hz",
    )

    ax.set_xlabel("Firing Rate (Hz)")
    ax.set_ylabel("Number of Neurons")
    ax.set_title("Distribution of Firing Rates in Dp Network")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(output_dir_path):
    """Main plotting function for Dp network simulation outputs.

    Loads all necessary data from the output directory and generates plots.

    Args:
        output_dir_path (str or Path): Path to directory containing simulation outputs
    """
    output_dir = Path(output_dir_path)

    print(f"Loading data from {output_dir}...")

    # Load all saved arrays
    connectivity_graph = np.load(output_dir / "connectivity_graph.npy")
    weights = np.load(output_dir / "weights.npy")
    feedforward_weights = np.load(output_dir / "feedforward_weights.npy")
    input_spikes = np.load(output_dir / "input_spikes.npy")
    output_spikes = np.load(output_dir / "output_spikes.npy")
    output_voltages = np.load(output_dir / "output_voltages.npy")
    output_I_exc = np.load(output_dir / "output_I_exc.npy")
    output_I_inh = np.load(output_dir / "output_I_inh.npy")
    neuron_types = np.load(output_dir / "neuron_types.npy")

    # Try to load cell type indices if available (for newer saves)
    try:
        cell_type_indices = np.load(output_dir / "cell_type_indices.npy")
    except FileNotFoundError:
        # Fall back to deriving from neuron_types and parameters
        cell_type_indices = None

    # Load parameters from the workspace to get simulation parameters
    # Look for parameter files in common locations
    params_file = None
    possible_params = [
        output_dir / "params.toml",  # If copied to output dir
        output_dir.parent.parent
        / "parameters"
        / "Dp_default.toml",  # Standard location
        output_dir.parent.parent
        / "workspace"
        / "Dp_default.toml",  # Alternative location
        Path("parameters/Dp_default.toml"),  # Relative to current directory
        Path("workspace/Dp_default.toml"),  # Alternative relative path
    ]

    for params_path in possible_params:
        if params_path.exists():
            params_file = params_path
            break

    if params_file is None:
        raise FileNotFoundError(
            "Could not find parameter file. Expected one of: "
            + ", ".join(str(p) for p in possible_params)
        )

    print(f"Loading parameters from {params_file}...")
    with open(params_file, "r") as f:
        params = toml.load(f)

    # Extract needed parameters
    delta_t = params["simulation"]["delta_t"]
    duration = params["simulation"]["duration"]
    num_assemblies = int(params["connectome"]["topology"]["num_assemblies"])

    # Extract neuron parameters for plotting
    cell_type_names = params["connectome"]["cell_types"]["names"]
    cell_type_signs = params["connectome"]["cell_types"]["signs"]

    # Extract LIF parameters for each cell type
    neuron_params = {}
    for i, name in enumerate(cell_type_names):
        sign = cell_type_signs[i]
        suffix = "E" if sign == 1 else "I"
        neuron_params[i] = {
            "threshold": params["model"]["neuron"][f"theta_{suffix}"],
            "rest": params["model"]["neuron"][f"U_rest_{suffix}"],
            "name": name,
            "sign": sign,
        }

    print("Generating plots...")

    # Network structure plots
    plot_assembly_graph(
        connectivity_graph, num_assemblies, output_dir / "01_assembly_graph.png"
    )
    plot_weighted_connectivity(
        weights, num_assemblies, output_dir / "02_weighted_connectivity.png"
    )
    plot_synaptic_input_histogram(
        weights, output_dir / "03_synaptic_input_histogram.png"
    )

    # Input analysis plots
    plot_mitral_cell_spikes(
        input_spikes, delta_t, duration, output_dir / "04_mitral_cell_spikes.png"
    )
    plot_feedforward_connectivity(
        feedforward_weights, output_dir / "05_feedforward_connectivity.png"
    )

    # Output analysis plots
    plot_dp_network_spikes(
        output_spikes, delta_t, duration, output_dir / "06_dp_network_spikes.png"
    )
    plot_firing_rate_distribution(
        output_spikes,
        neuron_types,
        duration,
        output_dir / "07_firing_rate_distribution.png",
    )

    # Use cell_type_indices if available, otherwise fall back to neuron_types
    neuron_type_indices = (
        cell_type_indices if cell_type_indices is not None else neuron_types
    )

    # Detailed neuronal dynamics plots using updated visualization functions
    plot_membrane_voltages(
        voltages=output_voltages,
        spikes=output_spikes,
        neuron_types=neuron_type_indices,
        neuron_params=neuron_params,
        delta_t=delta_t,
        duration=duration,
        n_neurons_plot=10,
        fraction=1,
        y_min=-100,
        y_max=0,
        y_tick_step=50,
        save_path=output_dir / "08_membrane_voltages.png",
    )

    plot_synaptic_currents(
        I_exc=output_I_exc,
        I_inh=output_I_inh,
        delta_t=delta_t,
        duration=duration,
        n_neurons_plot=10,
        fraction=1,
        save_path=output_dir / "09_synaptic_currents.png",
    )

    print("All plots generated successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simulate_Dp_plots.py <output_directory>")
        print("Example: python simulate_Dp_plots.py /path/to/output/folder")
        sys.exit(1)

    output_directory = sys.argv[1]
    main(output_directory)

"""
Generating spike trains from a synthetic connectome

This script generates spiketrains from a recurrent E-I network with current-based
LIF neurons, and assembly connectivity inspired by zebrafish Dp.

Overview:
1. First we generate a biologically plausible recurrent weight matrix with a
   Dp-inspired assembly structure.
2. Next we generate excitatory mitral cell inputs from the OB with Poisson
   statistics and sparse projections.
3. Then we initialise our network with parameters adapted from
   Meissner-Bernard et al. (2025) https://doi.org/10.1016/j.celrep.2025.115330
4. Finally we run our network and examine the output dynamics.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from synthetic_connectome import topology_generators, weight_assigners
from network_simulators.current_lif_network import CurrentLIFNetwork
from visualization import plot_membrane_voltages, plot_synaptic_currents
import torch


def main(output_dir, params_csv):
    """Main execution function for Dp network simulation.

    Args:
        output_dir (Path): Directory where output files will be saved
        params_csv (Path): Path to the CSV file containing network parameters
    """
    # Select device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all network parameters
    params_df = pd.read_csv(params_csv, comment="#")

    # Extract parameters from dataframe
    params = params_df.set_index("symbol")

    delta_t = params.loc["delta_t", "value"]
    duration = params.loc["duration", "value"]
    num_neurons = int(params.loc["num_neurons", "value"])
    num_assemblies = int(params.loc["num_assemblies", "value"])
    p_within = params.loc["p_within", "value"]
    p_between = params.loc["p_between", "value"]
    p_E = params.loc["p_E", "value"]
    w_mu_E = params.loc["w_mu_E", "value"]
    w_sigma_E = params.loc["w_sigma_E", "value"]
    w_mu_I = params.loc["w_mu_I", "value"]
    w_sigma_I = params.loc["w_sigma_I", "value"]
    num_mitral = int(params.loc["num_mitral", "value"])
    r_mitral = params.loc["r_mitral", "value"]
    p_feedforward = params.loc["p_feedforward", "value"]
    w_mu_FF = params.loc["w_mu_FF", "value"]
    w_sigma_FF = params.loc["w_sigma_FF", "value"]
    scaling_factor_E = params.loc["scaling_factor_E", "value"]
    scaling_factor_I = params.loc["scaling_factor_I", "value"]
    scaling_factor_FF = params.loc["scaling_factor_FF", "value"]

    # Generate Assembly-Based Topology
    # Generate assembly-based connectivity graph and neuron types
    connectivity_graph, neuron_types = topology_generators.assembly_generator(
        num_neurons=num_neurons,
        num_assemblies=num_assemblies,
        p_within=p_within,
        p_between=p_between,
        p_E=p_E,
    )

    # Visualize assembly graph structure
    plot_num_assemblies = 2  # Number of assemblies to display
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

    plt.savefig(output_dir / "01_assembly_graph.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Assign Synaptic Weights
    # Assign log-normal weights to connectivity graph
    weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=connectivity_graph,
        neuron_types=neuron_types,
        w_mu_E=w_mu_E,
        w_sigma_E=w_sigma_E,
        w_mu_I=w_mu_I,
        w_sigma_I=w_sigma_I,
    )

    # Visualize weighted connectivity matrix
    plot_num_assemblies = 2  # Number of assemblies to display
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

    plt.savefig(
        output_dir / "02_weighted_connectivity.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot histogram of total synaptic input to each neuron
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
    plt.savefig(
        output_dir / "03_synaptic_input_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Create network inputs
    # Generate Poisson spike trains for mitral cells
    n_steps = int(duration / delta_t)
    shape = (1, n_steps, num_mitral)
    p_spike = r_mitral * delta_t * 1e-3  # rate in Hz, delta_t in ms
    input_spikes = np.random.rand(*shape) < p_spike

    # Visualize sample mitral cell spike trains
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
    plt.savefig(output_dir / "04_mitral_cell_spikes.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Generate feedforward connectivity graph and mask inhibitory targets
    feedforward_connectivity_graph, _ = topology_generators.sparse_graph_generator(
        num_neurons=(num_mitral, num_neurons),
        p=p_feedforward,
        p_E=1.0,
    )
    # Set connections to inhibitory cells to zero (only connect to excitatory cells)
    inhibitory_cells = np.where(neuron_types == -1)[0]
    feedforward_connectivity_graph[:, inhibitory_cells] = 0

    # Assign log-normal weights to feedforward connectivity and scale
    feedforward_weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=feedforward_connectivity_graph,
        neuron_types=np.ones(num_mitral),
        w_mu_E=w_mu_FF,
        w_sigma_E=w_sigma_FF,
        w_mu_I=w_mu_I,
        w_sigma_I=w_sigma_I,
    )

    # Visualize feedforward connectivity matrix
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
    plt.savefig(
        output_dir / "05_feedforward_connectivity.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Initialize LIF network model and run simulation
    model = CurrentLIFNetwork(
        csv_path=params_csv,
        neuron_types=neuron_types,
        recurrent_weights=weights,
        feedforward_weights=feedforward_weights * scaling_factor_FF,
    )

    model.initialise_parameters(
        E_weight=scaling_factor_E,
        I_weight=scaling_factor_I,
    )

    # Move model to device for GPU acceleration
    model.to(device)

    # Run inference
    with model.inference_mode():
        output_spikes, output_voltages, output_I_exc, output_I_inh = model.forward(
            n_steps=n_steps,
            delta_t=delta_t,
            inputs=input_spikes,
        )

    # Visualize Dp network spike trains
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
    plt.savefig(output_dir / "06_dp_network_spikes.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Compute and plot firing rates for all neurons
    # Calculate firing rates (spikes per second)
    spike_counts = output_spikes[0].sum(axis=0).cpu().numpy()  # Total spikes per neuron
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
    plt.savefig(
        output_dir / "07_firing_rate_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Visualize membrane voltages for the first 10 neurons
    plot_membrane_voltages(
        voltages=output_voltages,
        spikes=output_spikes,
        neuron_types=neuron_types,
        model=model,
        delta_t=delta_t,
        duration=duration,
        n_neurons_plot=10,
        fraction=1,
        y_min=-100,
        y_max=0,
        y_tick_step=50,
        save_path=output_dir / "08_membrane_voltages.png",
    )

    # Visualize synaptic currents for the first 10 neurons
    plot_synaptic_currents(
        I_exc=output_I_exc,
        I_inh=output_I_inh,
        delta_t=delta_t,
        duration=duration,
        n_neurons_plot=10,
        fraction=1,
        save_path=output_dir / "09_synaptic_currents.png",
    )

    # Save output arrays
    np.save(output_dir / "output_spikes.npy", output_spikes.cpu().numpy())
    np.save(output_dir / "output_voltages.npy", output_voltages.cpu().numpy())
    np.save(output_dir / "output_I_exc.npy", output_I_exc.cpu().numpy())
    np.save(output_dir / "output_I_inh.npy", output_I_inh.cpu().numpy())
    np.save(output_dir / "input_spikes.npy", input_spikes)
    np.save(output_dir / "neuron_types.npy", neuron_types)
    np.save(output_dir / "connectivity_graph.npy", connectivity_graph)
    np.save(output_dir / "weights.npy", weights)
    np.save(output_dir / "feedforward_weights.npy", feedforward_weights)

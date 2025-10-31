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


def compute_network_statistics(
    spikes: np.ndarray,
    cell_type_indices: np.ndarray,
    dt: float,
) -> dict[str, float]:
    """Compute summary statistics from network activity.

    Args:
        spikes (np.ndarray): Spike tensor of shape (batch, time, neurons)
        cell_type_indices (np.ndarray): Cell type index for each neuron
        dt (float): Time step in milliseconds

    Returns:
        dict: Dictionary containing mean firing rates and CVs
    """
    # Compute firing rates per neuron (Hz)
    spike_counts = spikes.sum(axis=1).squeeze()  # Sum over time
    duration_s = spikes.shape[1] * dt / 1000.0  # Convert ms to s
    firing_rates = spike_counts / duration_s

    # Compute ISIs and CVs per neuron
    cvs = []
    for neuron_idx in range(spikes.shape[2]):
        spike_times = np.nonzero(spikes[0, :, neuron_idx])[0]
        if len(spike_times) > 1:
            isis = np.diff(spike_times.astype(float)) * dt
            if len(isis) > 0:
                cv = np.std(isis) / np.mean(isis)
                cvs.append(cv)

    # Compute statistics by cell type
    stats = {}
    for cell_type in np.unique(cell_type_indices):
        mask = cell_type_indices == cell_type
        stats[f"mean_fr_type_{int(cell_type)}"] = float(firing_rates[mask].mean())
        stats[f"std_fr_type_{int(cell_type)}"] = float(firing_rates[mask].std())

    stats["mean_firing_rate"] = float(firing_rates.mean())
    stats["std_firing_rate"] = float(firing_rates.std())
    stats["mean_cv"] = float(np.mean(cvs)) if cvs else 0.0
    stats["std_cv"] = float(np.std(cvs)) if cvs else 0.0
    stats["fraction_active"] = float((firing_rates > 0).mean())

    return stats


def generate_training_plots(
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
    params: dict,
    dt: float,
) -> dict[str, plt.Figure]:
    """Generate all visualization plots for current training state.

    Args:
        output_dir (Path): Directory where plots will be saved
        epoch (int): Current epoch number
        spikes (np.ndarray): Network spike trains
        voltages (np.ndarray): Membrane voltages
        conductances (np.ndarray): Recurrent synaptic conductances
        conductances_FF (np.ndarray): Feedforward synaptic conductances
        currents (np.ndarray): Recurrent synaptic currents
        currents_FF (np.ndarray): Feedforward synaptic currents
        input_spikes (np.ndarray): Input spike trains
        cell_type_indices (np.ndarray): Cell type assignments
        input_cell_type_indices (np.ndarray): Input cell type assignments
        cell_type_names (list[str]): Names of cell types
        input_cell_type_names (list[str]): Names of input cell types
        weights (np.ndarray): Recurrent weights
        feedforward_weights (np.ndarray): Feedforward weights
        connectivity_graph (np.ndarray): Connectivity graph
        num_assemblies (int): Number of assemblies
        params (dict): All parameters from config file
        dt (float): Time step in ms

    Returns:
        dict: Dictionary of figure objects keyed by plot name
    """
    figures = {}
    duration = spikes.shape[1] * dt

    # Sum over rise/decay dimension (axis=3) to get total conductance
    conductances = conductances.sum(axis=3)
    conductances_FF = conductances_FF.sum(axis=3)

    # Note: Static network structure plots (assembly_graph, weighted_connectivity,
    # input_count_histogram, feedforward_connectivity, synaptic_input_histogram) are
    # generated once at initialization and not included in checkpoint plots to avoid redundancy

    # Input analysis
    figures["mitral_cell_spikes"] = plot_mitral_cell_spikes(
        input_spikes=input_spikes,
        dt=dt,
        duration=duration,
    )

    # Output analysis
    figures["dp_network_spikes"] = plot_dp_network_spikes(
        output_spikes=spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        duration=duration,
    )

    figures["firing_rate_distribution"] = plot_firing_rate_distribution(
        output_spikes=spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        duration=duration,
    )

    # Prepare neuron_params for detailed plots
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

    figures["membrane_voltages"] = plot_membrane_voltages(
        voltages=voltages,
        spikes=spikes,
        neuron_types=cell_type_indices,
        delta_t=dt,
        duration=duration,
        neuron_params=neuron_params,
        n_neurons_plot=5,
        fraction=1,
    )

    # Synaptic currents
    # Sum currents over synapse types to get total excitatory and inhibitory
    # Currents shape is (batch, time, neurons, n_synapse_types)
    # Sum over the last axis to get total current per neuron
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
        duration=duration,
        n_neurons_plot=5,
        fraction=1.0,
        show_total=True,
        neuron_types=cell_type_indices,
        neuron_params=neuron_params,
    )

    # Synaptic conductances
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

    figures["synaptic_conductances"] = plot_synaptic_conductances(
        output_conductances=conductances,
        input_conductances=conductances_FF,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        input_cell_type_names=input_cell_type_names,
        recurrent_synapse_names=recurrent_synapse_names,
        feedforward_synapse_names=feedforward_synapse_names,
        dt=dt,
        duration=duration,
        neuron_id=0,
        fraction=1.0,
    )

    # Firing statistics (these functions expect numpy arrays)
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

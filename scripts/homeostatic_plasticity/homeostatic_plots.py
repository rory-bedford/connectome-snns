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
import torch


def compute_network_statistics(
    spikes: torch.Tensor,
    cell_type_indices: torch.Tensor,
    dt: float,
) -> dict[str, float]:
    """Compute summary statistics from network activity.

    Args:
        spikes (torch.Tensor): Spike tensor of shape (batch, time, neurons)
        cell_type_indices (torch.Tensor): Cell type index for each neuron
        dt (float): Time step in milliseconds

    Returns:
        dict: Dictionary containing mean firing rates and CVs
    """
    # Compute firing rates per neuron (Hz)
    spike_counts = spikes.sum(dim=1).squeeze()  # Sum over time
    duration_s = spikes.shape[1] * dt / 1000.0  # Convert ms to s
    firing_rates = spike_counts / duration_s

    # Compute ISIs and CVs per neuron
    cvs = []
    for neuron_idx in range(spikes.shape[2]):
        spike_times = torch.nonzero(spikes[0, :, neuron_idx]).squeeze()
        if spike_times.ndim > 0 and len(spike_times) > 1:
            isis = torch.diff(spike_times.float()) * dt
            if len(isis) > 0:
                cv = torch.std(isis) / torch.mean(isis)
                cvs.append(cv.item())

    # Compute statistics by cell type
    stats = {}
    for cell_type in torch.unique(cell_type_indices):
        mask = cell_type_indices == cell_type
        stats[f"mean_fr_type_{cell_type.item()}"] = firing_rates[mask].mean().item()
        stats[f"std_fr_type_{cell_type.item()}"] = firing_rates[mask].std().item()

    stats["mean_firing_rate"] = firing_rates.mean().item()
    stats["std_firing_rate"] = firing_rates.std().item()
    stats["mean_cv"] = np.mean(cvs) if cvs else 0.0
    stats["std_cv"] = np.std(cvs) if cvs else 0.0
    stats["fraction_active"] = (firing_rates > 0).float().mean().item()

    return stats


def generate_training_plots(
    spikes: torch.Tensor,
    voltages: torch.Tensor,
    conductances: torch.Tensor,
    conductances_FF: torch.Tensor,
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
        spikes (torch.Tensor): Network spike trains
        voltages (torch.Tensor): Membrane voltages
        conductances (torch.Tensor): Recurrent synaptic conductances
        conductances_FF (torch.Tensor): Feedforward synaptic conductances
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

    # Prepare g_bar dictionaries
    recurrent_g_bar_by_type = {}
    for cell_type in cell_type_names:
        g_bar_values = params["recurrent"]["synapses"][cell_type]["g_bar"]
        recurrent_g_bar_by_type[cell_type] = sum(g_bar_values)

    feedforward_g_bar_by_type = {}
    for cell_type in input_cell_type_names:
        g_bar_values = params["feedforward"]["synapses"][cell_type]["g_bar"]
        feedforward_g_bar_by_type[cell_type] = sum(g_bar_values)

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

    # Input analysis
    figures["mitral_cell_spikes"] = plot_mitral_cell_spikes(
        input_spikes=input_spikes,
        dt=dt,
        duration=duration,
    )

    figures["feedforward_connectivity"] = plot_feedforward_connectivity(
        feedforward_weights=feedforward_weights,
        input_cell_type_indices=input_cell_type_indices,
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

    # Synaptic currents (need to compute from conductances)
    # Split currents into excitatory and inhibitory based on conductances
    # Note: This assumes conductances shape is (batch, time, neurons, n_synapse_types)
    # For conductance-based model, we compute I = g * (V - E_syn)
    # This is an approximation - ideally pass actual currents if available
    # For now, we'll skip detailed current plots or use a simplified version
    # TODO: Add synaptic current computation if needed

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
        fraction=0.1,
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

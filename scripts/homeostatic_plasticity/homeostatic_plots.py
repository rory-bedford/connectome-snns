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
    total_conductances = feedforward_conductances.sum(axis=(0, 1))  # Shape: (neurons, synapses)
    
    # Find neurons with any non-zero feedforward conductance
    neurons_with_inputs = np.any(total_conductances > 1e-10, axis=1)  # Small threshold to avoid numerical issues
    
    if np.any(neurons_with_inputs):
        # Return the first neuron that has feedforward inputs
        return int(np.nonzero(neurons_with_inputs)[0][0])
    else:
        # Fallback to neuron 0 if no feedforward inputs found
        return 0


def compute_network_statistics(
    spikes: np.ndarray,
    cell_type_indices: np.ndarray,
    cell_type_names: list[str],
    dt: float,
) -> dict[str, float]:
    """Compute summary statistics from network activity.

    Args:
        spikes (np.ndarray): Spike tensor of shape (batch, time, neurons)
        cell_type_indices (np.ndarray): Cell type index for each neuron
        cell_type_names (list[str]): Names of cell types
        dt (float): Time step in milliseconds

    Returns:
        dict: Dictionary containing mean firing rates and CVs by cell type
    """
    # Compute firing rates per neuron (Hz), averaged over batch
    spike_counts = spikes.sum(axis=1)  # Sum over time: (batch, neurons)
    spike_counts_avg = spike_counts.mean(axis=0)  # Average over batch: (neurons,)
    duration_s = spikes.shape[1] * dt / 1000.0  # Convert ms to s
    firing_rates = spike_counts_avg / duration_s

    # Compute ISIs and CVs per neuron, averaged over batch
    cvs = np.full(spikes.shape[2], np.nan)  # Initialize with NaN for silent neurons
    for neuron_idx in range(spikes.shape[2]):
        neuron_cvs = []
        for batch_idx in range(spikes.shape[0]):
            spike_times = np.nonzero(spikes[batch_idx, :, neuron_idx])[0]
            if len(spike_times) > 1:
                isis = np.diff(spike_times.astype(float)) * dt
                if len(isis) > 0:
                    neuron_cvs.append(np.std(isis) / np.mean(isis))
        
        # Average CV across batch (excluding NaN values)
        if neuron_cvs:
            cvs[neuron_idx] = np.mean(neuron_cvs)

    # Compute statistics by cell type (use actual cell type names)
    stats = {}
    for cell_type in np.unique(cell_type_indices):
        mask = cell_type_indices == cell_type
        cell_type_name = cell_type_names[int(cell_type)]

        # Firing rate statistics
        stats[f"firing_rate/{cell_type_name}/mean"] = float(firing_rates[mask].mean())
        stats[f"firing_rate/{cell_type_name}/std"] = float(firing_rates[mask].std())

        # CV statistics (only for neurons with valid CVs)
        cell_cvs = cvs[mask]
        valid_cvs = cell_cvs[~np.isnan(cell_cvs)]
        stats[f"cv/{cell_type_name}/mean"] = (
            float(np.mean(valid_cvs)) if len(valid_cvs) > 0 else 0.0
        )
        stats[f"cv/{cell_type_name}/std"] = (
            float(np.std(valid_cvs)) if len(valid_cvs) > 0 else 0.0
        )

        # Fraction active
        stats[f"fraction_active/{cell_type_name}"] = float(
            (firing_rates[mask] > 0).mean()
        )

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
    dt: float,
    neuron_params: dict,
    recurrent_synapse_names: dict,
    feedforward_synapse_names: dict,
    recurrent_g_bar_by_type: dict,
    feedforward_g_bar_by_type: dict,
) -> dict[str, plt.Figure]:
    """Generate all visualization plots for current training state.

    Args:
        output_dir (Path): Directory where plots will be saved
        epoch (int): Current epoch number
        spikes (np.ndarray): Network spike trains (batch, time, neurons)
        voltages (np.ndarray): Membrane voltages (batch, time, neurons)
        conductances (np.ndarray): Recurrent synaptic conductances (batch, time, neurons, synapses, rise/decay)
        conductances_FF (np.ndarray): Feedforward synaptic conductances (batch, time, neurons, synapses, rise/decay)
        currents (np.ndarray): Recurrent synaptic currents (batch, time, neurons, synapses)
        currents_FF (np.ndarray): Feedforward synaptic currents (batch, time, neurons, synapses)
        input_spikes (np.ndarray): Input spike trains (batch, time, neurons)
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

    # Prepare g_bar dictionaries for synaptic input histogram (use pre-computed values)

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

    # Use pre-computed neuron_params (passed as parameter)

    figures["membrane_voltages"] = plot_membrane_voltages(
        voltages=voltages,
        spikes=spikes,
        neuron_types=cell_type_indices,
        delta_t=dt,
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
        n_neurons_plot=5,
        fraction=1.0,
        show_total=True,
        neuron_types=cell_type_indices,
        neuron_params=neuron_params,
    )

    # Use pre-computed synapse names (passed as parameters)
    
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

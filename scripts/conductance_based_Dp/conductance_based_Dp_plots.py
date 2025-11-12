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
    plot_psth,
)

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for cluster environments
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from network_simulators.conductance_based.parameter_loader import (
    ConductanceBasedParams,
)
import toml


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

    # Vectorized CV computation using the analysis module
    from analysis.firing_statistics import compute_spike_train_cv

    cv_values = compute_spike_train_cv(spikes, dt=dt)  # Shape: (batch, neurons)
    cv_per_neuron = np.nanmean(cv_values, axis=0)  # Average over batches

    # Compute statistics by cell type (use actual cell type names)
    stats = {}
    for cell_type in np.unique(cell_type_indices):
        mask = cell_type_indices == cell_type
        cell_type_name = cell_type_names[int(cell_type)]

        # Firing rate statistics
        stats[f"firing_rate/{cell_type_name}/mean"] = float(firing_rates[mask].mean())
        stats[f"firing_rate/{cell_type_name}/std"] = float(firing_rates[mask].std())

        # CV statistics (only for neurons with valid CVs)
        cell_cvs = cv_per_neuron[mask]
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
        dt (float): Time step in ms
        neuron_params (dict): Neuron parameters for plotting
        recurrent_synapse_names (dict): Names of recurrent synapses by cell type
        feedforward_synapse_names (dict): Names of feedforward synapses by cell type
        recurrent_g_bar_by_type (dict): Maximum conductance values for recurrent synapses
        feedforward_g_bar_by_type (dict): Maximum conductance values for feedforward synapses

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

    figures["psth"] = plot_psth(
        spike_trains=spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        window_size=50.0,  # 50 ms window size - can be made configurable
        dt=dt,
    )

    # Assembly PSTH - slice to first assembly (assemblies are equal-sized and contiguous)
    n_neurons = spikes.shape[2]
    neurons_per_assembly = n_neurons // num_assemblies
    assembly_start = 0  # First assembly
    assembly_end = neurons_per_assembly

    figures["assembly_psth"] = plot_psth(
        spike_trains=spikes[
            :, :, assembly_start:assembly_end
        ],  # Slice to first assembly
        cell_type_indices=cell_type_indices[assembly_start:assembly_end],
        cell_type_names=cell_type_names,
        window_size=50.0,  # 50 ms window size - can be made configurable
        dt=dt,
        title="Assembly 1 PSTH (window = 50.0 ms)",
    )

    return figures


if __name__ == "__main__":
    """Standalone script interface - loads data from disk and generates plots."""
    if len(sys.argv) != 2:
        print("Usage: python conductance_based_Dp_plots.py <output_directory>")
        print("Example: python conductance_based_Dp_plots.py /path/to/output/folder")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
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

    # Prepare parameters for plotting
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

    print(f"Generating plots and saving to {figures_dir}...")

    # Generate all plots using the uniform function
    figures = generate_training_plots(
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
        ("psth", "15_psth.png"),
        ("assembly_psth", "16_assembly_psth.png"),
    ]

    for plot_key, filename in plot_names:
        if plot_key in figures:
            fig = figures[plot_key]
            fig.savefig(figures_dir / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

    print(f"✓ Saved all plots to {figures_dir}")
    print("All plots generated successfully!")

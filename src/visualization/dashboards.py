"""Unified dashboard module for comprehensive network visualization.

This module creates dashboards displaying connectivity and activity plots
as subplots in unified figures.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.parameter_loaders import (
    EXCITATORY_SYNAPSE_TYPES,
    INHIBITORY_SYNAPSE_TYPES,
)
from visualization.connectivity import (
    plot_weighted_connectivity,
    plot_input_count_pie_chart,
    plot_weight_distribution_by_input_type,
    plot_weight_statistics_matrix,
    plot_feedforward_connectivity,
)
from visualization.firing_statistics import (
    plot_psth,
    plot_firing_rate_distribution,
    plot_cv_histogram,
    plot_isi_histogram,
    plot_fano_factor_vs_window_size,
)
from visualization.neuronal_dynamics import (
    plot_spike_trains,
    plot_membrane_voltages,
    plot_synaptic_currents,
    plot_synaptic_conductances,
)


# ===================== CONNECTIVITY DASHBOARD =====================


def create_connectivity_dashboard(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    connectome_mask: NDArray[np.bool_],
    feedforward_mask: NDArray[np.bool_],
    num_assemblies: int | None = None,
    heatmap_inches: float = 6.0,
    plot_fraction_feedforward: float = 0.1,
    plot_fraction_recurrent: float = 0.1,
    scaling_factors: NDArray[np.float32] | None = None,
    scaling_factors_FF: NDArray[np.float32] | None = None,
) -> plt.Figure:
    """Create a connectivity dashboard showing connectomes and weight distributions.

    Args:
        weights (NDArray[np.float32]): Weight matrix (N x N).
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for recurrent neurons.
        input_cell_type_indices (NDArray[np.int32]): Array of cell type indices for input neurons.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.
        connectome_mask (NDArray[np.bool_]): Binary mask for valid recurrent connections.
        feedforward_mask (NDArray[np.bool_]): Binary mask for valid feedforward connections.
        num_assemblies (int | None): Number of assemblies in the network. If provided, uses this
            to determine plotting size instead of plot_fraction_recurrent. Defaults to None.
        heatmap_inches (float): Size of heatmaps in inches. Defaults to 6.0.
        plot_fraction_feedforward (float): Fraction of neurons to show in feedforward connectivity. Defaults to 0.1.
        plot_fraction_recurrent (float): Fraction of neurons to show in recurrent connectivity (only used if num_assemblies is None). Defaults to 0.1.
        scaling_factors (NDArray[np.float32] | None): Recurrent scaling factors (n_source_types, n_target_types). If provided, weights will be scaled by cell type.
        scaling_factors_FF (NDArray[np.float32] | None): Feedforward scaling factors (n_source_types, n_target_types). If provided, feedforward weights will be scaled by cell type.

    Returns:
        plt.Figure: Single comprehensive dashboard figure with connectivity plots arranged.
    """
    # Apply scaling factors if provided
    if scaling_factors is not None:
        weights = weights.copy()
        for source_idx in range(len(cell_type_names)):
            source_mask = cell_type_indices == source_idx
            for target_idx in range(len(cell_type_names)):
                target_mask = cell_type_indices == target_idx
                scale = scaling_factors[source_idx, target_idx]
                weights[np.ix_(target_mask, source_mask)] *= scale

    if scaling_factors_FF is not None:
        feedforward_weights = feedforward_weights.copy()
        for source_idx in range(len(input_cell_type_names)):
            source_mask = input_cell_type_indices == source_idx
            for target_idx in range(len(cell_type_names)):
                target_mask = cell_type_indices == target_idx
                scale = scaling_factors_FF[source_idx, target_idx]
                feedforward_weights[np.ix_(source_mask, target_mask)] *= scale

    # Create main dashboard figure
    fig = plt.figure(figsize=(24, 16))

    # Create layout:
    # - Left column (2 rows): connectivity heatmaps (feedforward above weighted)
    # - Right column (2 rows of 2 boxes): pie chart top left, violin plot top right, bottom two empty
    gs_main = fig.add_gridspec(
        2,
        2,
        hspace=0.35,
        wspace=0.15,
        height_ratios=[1, 2.4],
        width_ratios=[1, 1],
        left=0.05,
        right=0.98,
        top=0.94,
        bottom=0.05,
    )

    # Top left: Feedforward connectivity
    ax1 = fig.add_subplot(gs_main[0, 0])

    # Middle left: Weighted connectivity (1.5x the size of feedforward)
    ax2 = fig.add_subplot(gs_main[1, 0])

    # Plot both connectivity matrices
    plot_feedforward_connectivity(
        feedforward_weights=feedforward_weights,
        input_cell_type_indices=input_cell_type_indices,
        plot_fraction=plot_fraction_feedforward,
        ax=ax1,
        show_legend=True,
        section_title="Weighted Connectomes",
        section_title_axes=[ax1, ax2],
    )

    plot_weighted_connectivity(
        weights=weights,
        cell_type_indices=cell_type_indices,
        num_assemblies=num_assemblies,
        plot_fraction=plot_fraction_recurrent,
        heatmap_inches=heatmap_inches,
        ax=ax2,
    )

    # Right side: 2x2 grid (pie chart top left, violin plot top right, bottom two empty)
    gs_right = gs_main[0:2, 1].subgridspec(
        2, 2, hspace=0.3, wspace=0.15, height_ratios=[1, 1]
    )

    # Top left: Pie chart (shift left by adjusting position)
    ax_pie = fig.add_subplot(gs_right[0, 0])
    # Shift the pie chart left by 20 pixels (~0.08 figure coordinates at width=24)
    pos = ax_pie.get_position()
    ax_pie.set_position([pos.x0 - 0.02, pos.y0, pos.width, pos.height])
    plot_input_count_pie_chart(
        weights,
        feedforward_weights,
        cell_type_indices,
        input_cell_type_indices,
        cell_type_names,
        input_cell_type_names,
        ax_pie,
    )

    # Top right: Violin plot
    ax_violin = fig.add_subplot(gs_right[0, 1])
    plot_weight_distribution_by_input_type(
        weights,
        feedforward_weights,
        cell_type_indices,
        input_cell_type_indices,
        cell_type_names,
        input_cell_type_names,
        connectome_mask,
        feedforward_mask,
        ax_violin,
    )

    # Bottom left: Mean weight matrix
    ax_mean = fig.add_subplot(gs_right[1, 0])
    # Bottom right: Std weight matrix
    ax_std = fig.add_subplot(gs_right[1, 1])
    plot_weight_statistics_matrix(
        weights,
        feedforward_weights,
        cell_type_indices,
        input_cell_type_indices,
        cell_type_names,
        input_cell_type_names,
        ax_mean,
        ax_std,
    )

    return fig


# ===================== ACTIVITY DASHBOARD =====================


def create_activity_dashboard(
    output_spikes: NDArray[np.int32],
    input_spikes: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float,
    voltages: NDArray[np.float32],
    neuron_types: NDArray[np.int32],
    neuron_params: dict,
    recurrent_currents: NDArray[np.float32],
    feedforward_currents: NDArray[np.float32],
    leak_currents: NDArray[np.float32],
    recurrent_conductances: NDArray[np.float32],
    feedforward_conductances: NDArray[np.float32],
    input_cell_type_names: list[str],
    recurrent_synapse_names: dict[str, list[str]],
    feedforward_synapse_names: dict[str, list[str]],
    window_size: float = 50.0,
    n_neurons_plot: int = 20,
    fraction: float = 1.0,
    random_seed: int = 42,
    assembly_ids: NDArray[np.int32] | None = None,
) -> plt.Figure:
    """Create a comprehensive activity dashboard with two columns: Activity and Spiking Statistics.

    Args:
        output_spikes (NDArray[np.int32]): Output spike array with shape (batch, time, neurons).
        input_spikes (NDArray[np.int32]): Input spike array with shape (batch, time, neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step in milliseconds.
        voltages (NDArray[np.float32]): Membrane voltages with shape (batch, time, neurons).
        neuron_types (NDArray[np.int32]): Neuron type indices.
        neuron_params (dict): Dictionary mapping cell type indices to parameters.
        recurrent_currents (NDArray[np.float32]): Recurrent currents with shape (batch, time, neurons, synapses).
        feedforward_currents (NDArray[np.float32]): Feedforward currents with shape (batch, time, neurons, synapses).
        leak_currents (NDArray[np.float32]): Leak currents with shape (batch, time, neurons).
        recurrent_conductances (NDArray[np.float32]): Recurrent conductances with shape (batch, time, neurons, synapses).
        feedforward_conductances (NDArray[np.float32]): Feedforward conductances with shape (batch, time, neurons, synapses).
        input_cell_type_names (list[str]): Names of input cell types.
        recurrent_synapse_names (dict[str, list[str]]): Synapse names for each recurrent cell type.
        feedforward_synapse_names (dict[str, list[str]]): Synapse names for each feedforward cell type.
        window_size (float): Window size for PSTH in milliseconds. Defaults to 50.0.
        n_neurons_plot (int): Number of neurons to plot in raster. Defaults to 20.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        random_seed (int): Random seed for shuffling neurons. Defaults to 42.
        assembly_ids (NDArray[np.int32] | None): Optional array of assembly IDs for each neuron.
            If provided, neuron selection for detailed plots is limited to the first assembly (index 0).

    Returns:
        plt.Figure: Matplotlib figure object containing the activity dashboard.
    """
    n_cell_types = len(cell_type_names)

    # Split currents into excitatory and inhibitory
    # Get synapse indices for exc/inh classification

    # Build synapse indices for recurrent currents
    exc_indices_recurrent = []
    inh_indices_recurrent = []
    synapse_idx = 0
    for cell_name in cell_type_names:
        synapse_names_list = recurrent_synapse_names[cell_name]
        for syn_name in synapse_names_list:
            if syn_name in EXCITATORY_SYNAPSE_TYPES:
                exc_indices_recurrent.append(synapse_idx)
            elif syn_name in INHIBITORY_SYNAPSE_TYPES:
                inh_indices_recurrent.append(synapse_idx)
            synapse_idx += 1

    # Build synapse indices for feedforward currents
    exc_indices_ff = []
    inh_indices_ff = []
    synapse_idx = 0
    for cell_name in input_cell_type_names:
        synapse_names_list = feedforward_synapse_names[cell_name]
        for syn_name in synapse_names_list:
            if syn_name in EXCITATORY_SYNAPSE_TYPES:
                exc_indices_ff.append(synapse_idx)
            elif syn_name in INHIBITORY_SYNAPSE_TYPES:
                inh_indices_ff.append(synapse_idx)
            synapse_idx += 1

    # Sum currents across excitatory and inhibitory synapses
    # Negate currents so that depolarizing (excitatory) appears positive and hyperpolarizing (inhibitory) appears negative
    I_exc_recurrent = recurrent_currents[:, :, :, exc_indices_recurrent].sum(axis=-1)
    I_exc_ff = feedforward_currents[:, :, :, exc_indices_ff].sum(axis=-1)
    I_exc = -(I_exc_recurrent + I_exc_ff)  # Negate to flip sign for plotting

    I_inh_recurrent = recurrent_currents[:, :, :, inh_indices_recurrent].sum(axis=-1)
    I_inh_ff = feedforward_currents[:, :, :, inh_indices_ff].sum(axis=-1)
    # Combine inhibitory and leak currents (both hyperpolarizing), then negate for plotting
    I_inh = -(
        I_inh_recurrent + I_inh_ff + leak_currents
    )  # Negate to flip sign for plotting

    # Create figure with 2 columns: Activity (left) and Spiking Statistics (right)
    # Same size as connectivity dashboard (24x16), first column 2x width of second
    fig = plt.figure(figsize=(24, 16))
    gs_main = fig.add_gridspec(
        1,
        2,
        hspace=0.3,
        wspace=0.25,
        width_ratios=[2, 1],
        left=0.05,
        right=0.98,
        top=0.94,
        bottom=0.05,
    )

    # LEFT COLUMN: Activity (spike trains, PSTH, membrane voltages, conductances, currents)
    # Now 5 rows: combined spikes, PSTH, voltages, conductances (1.5x height), currents
    gs_activity = gs_main[0, 0].subgridspec(
        5, 1, hspace=0.4, height_ratios=[1, 1, 1, 2.0, 1]
    )

    # Row 0: Combined input and output spike trains
    # Sample neurons maintaining network ratios
    n_input_neurons = input_spikes.shape[2]
    n_output_neurons = output_spikes.shape[2]
    n_total = n_input_neurons + n_output_neurons

    # Calculate how many of each type to plot (maintain ratio)
    n_input_plot = max(1, int(n_neurons_plot * n_input_neurons / n_total))
    n_output_plot = n_neurons_plot - n_input_plot

    # Randomly sample neurons
    np.random.seed(random_seed)
    input_indices = np.random.choice(n_input_neurons, n_input_plot, replace=False)
    output_indices = np.random.choice(n_output_neurons, n_output_plot, replace=False)

    # Concatenate spikes: (batch, time, neurons)
    combined_spikes = np.concatenate(
        [input_spikes[:, :, input_indices], output_spikes[:, :, output_indices]], axis=2
    )

    # Create combined cell type indices and names
    # Input neurons get a special "feedforward" index
    combined_cell_type_indices = np.concatenate(
        [
            np.full(n_input_plot, -1),  # -1 for feedforward
            cell_type_indices[output_indices],
        ]
    )
    combined_cell_type_names = ["Feedforward"] + cell_type_names

    ax_combined_spikes = fig.add_subplot(gs_activity[0, 0])
    plot_spike_trains(
        spikes=combined_spikes,
        dt=dt,
        cell_type_indices=combined_cell_type_indices,
        cell_type_names=combined_cell_type_names,
        n_neurons_plot=n_neurons_plot,
        fraction=1.0,
        random_seed=None,  # Already sampled
        title="Network Spike Trains (Feedforward + Recurrent)",
        ylabel="",
        ax=ax_combined_spikes,
    )

    # Row 1: PSTH
    ax_psth = fig.add_subplot(gs_activity[1, 0])
    plot_psth(
        spike_trains=output_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        window_size=window_size,
        dt=dt,
        ax=ax_psth,
        input_spike_trains=input_spikes,
    )

    # Row 2: Membrane voltages (1 excitatory neuron only)
    # Find a suitable excitatory neuron that:
    # 1. Is excitatory
    # 2. Has fired in the time window
    # 3. Has received excitatory recurrent input (conductance > 0)
    # 4. Has received inhibitory input (conductance > 0)

    n_steps = output_spikes.shape[1]
    n_steps_plot = n_steps  # Use full duration for neuron selection

    # Sum over rise/decay components for neuron selection
    recurrent_g_total_for_selection = recurrent_conductances.sum(
        axis=3
    )  # (batch, time, neurons, n_synapses)

    # Find excitatory neurons
    exc_neuron_indices = []
    for i, cell_type_idx in enumerate(cell_type_indices):
        cell_name = cell_type_names[cell_type_idx].lower()
        if "excit" in cell_name:
            exc_neuron_indices.append(i)

    # If assembly_ids provided, filter to first assembly (index 0)
    if assembly_ids is not None:
        first_assembly_mask = assembly_ids == 0
        exc_neuron_indices = [
            idx for idx in exc_neuron_indices if first_assembly_mask[idx]
        ]

    selected_neuron = None

    if len(exc_neuron_indices) > 0:
        # Build synapse indices across ALL cell types (same as in plotting function)
        exc_syn_indices = []
        inh_syn_indices = []
        synapse_idx = 0
        for cell_name in cell_type_names:
            synapse_names_list = recurrent_synapse_names[cell_name]
            for syn_name in synapse_names_list:
                if syn_name in EXCITATORY_SYNAPSE_TYPES:
                    exc_syn_indices.append(synapse_idx)
                elif syn_name in INHIBITORY_SYNAPSE_TYPES:
                    inh_syn_indices.append(synapse_idx)
                synapse_idx += 1

        candidates = []
        for neuron_idx in exc_neuron_indices:
            # Check if neuron has spiked in the time window
            has_spiked = output_spikes[0, -n_steps_plot:, neuron_idx].sum() > 0

            # Check if neuron has received excitatory recurrent input
            has_exc_input = False
            if exc_syn_indices:
                exc_conductance_sum = recurrent_g_total_for_selection[
                    0, -n_steps_plot:, neuron_idx, exc_syn_indices
                ].sum()
                has_exc_input = exc_conductance_sum > 0

            # Check if neuron has received inhibitory input
            has_inh_input = False
            if inh_syn_indices:
                inh_conductance_sum = recurrent_g_total_for_selection[
                    0, -n_steps_plot:, neuron_idx, inh_syn_indices
                ].sum()
                has_inh_input = inh_conductance_sum > 0

            if has_spiked and has_exc_input and has_inh_input:
                candidates.append(neuron_idx)

        # Rank candidates by firing rate and choose median
        if len(candidates) > 0:
            # Compute firing rates for candidates
            candidate_rates = []
            for neuron_idx in candidates:
                spike_count = output_spikes[0, :, neuron_idx].sum()
                duration_s = output_spikes.shape[1] * dt * 1e-3  # Convert ms to s
                firing_rate = spike_count / duration_s
                candidate_rates.append(firing_rate)

            # Sort candidates by firing rate and select median
            sorted_indices = np.argsort(candidate_rates)
            median_idx = sorted_indices[len(sorted_indices) // 2]
            selected_neuron = candidates[median_idx]
        else:
            selected_neuron = exc_neuron_indices[0]
    else:
        # No excitatory neurons found, use first neuron
        selected_neuron = 0

    # Create single axis for 1 neuron
    ax_voltage = fig.add_subplot(gs_activity[2, 0])

    # Subset voltages and spikes for selected neuron
    voltages_subset = voltages[:, :, [selected_neuron]]
    spikes_subset = output_spikes[:, :, [selected_neuron]]
    neuron_types_subset = neuron_types[[selected_neuron]]

    plot_membrane_voltages(
        voltages=voltages_subset,
        spikes=spikes_subset,
        neuron_types=neuron_types_subset,
        delta_t=dt,
        neuron_params=neuron_params,
        n_neurons_plot=1,
        fraction=1.0,
        ax=[ax_voltage],
    )
    # Add title for membrane potential
    ax_voltage.set_title("Membrane Potential", fontsize=11, loc="center", pad=10)

    # Row 3: Conductances (combined: E, I, Feedforward)
    # Use the same selected neuron as for membrane voltage
    # Show only 10% of the time range for detailed view
    # Sum over rise/decay components (axis 3) to get total conductances
    recurrent_g_total = recurrent_conductances.sum(
        axis=3
    )  # (batch, time, neurons, n_synapses)
    feedforward_g_total = feedforward_conductances.sum(
        axis=3
    )  # (batch, time, neurons, n_synapses)

    gs_conductances = gs_activity[3, 0].subgridspec(3, 1, hspace=0.2)
    conductance_axes = [fig.add_subplot(gs_conductances[i, 0]) for i in range(3)]
    plot_synaptic_conductances(
        recurrent_conductances=recurrent_g_total,
        feedforward_conductances=feedforward_g_total,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        input_cell_type_names=input_cell_type_names,
        recurrent_synapse_names=recurrent_synapse_names,
        feedforward_synapse_names=feedforward_synapse_names,
        dt=dt,
        neuron_id=selected_neuron,
        fraction=0.1,
        ax=conductance_axes,
    )
    # Add title for conductances
    conductance_axes[0].set_title("Conductance", fontsize=11, loc="center", pad=10)

    # Row 4: Currents (same selected neuron)
    # Subset currents for the selected neuron
    # Show only 10% of the time range for detailed view
    I_exc_subset = I_exc[:, :, [selected_neuron]]
    I_inh_subset = I_inh[:, :, [selected_neuron]]  # Already includes leak current
    I_tot = I_exc_subset + I_inh_subset
    neuron_types_subset_current = neuron_types[[selected_neuron]]

    ax_currents = fig.add_subplot(gs_activity[4, 0])
    plot_synaptic_currents(
        I_exc=I_exc_subset,
        I_inh=I_inh_subset,
        I_tot=I_tot,
        delta_t=dt,
        n_neurons_plot=1,
        fraction=0.1,
        show_total=False,
        neuron_types=neuron_types_subset_current,
        neuron_params=neuron_params,
        ax=[ax_currents],
    )
    # Add title for currents
    ax_currents.set_title("Input Current", fontsize=11, loc="center", pad=10)

    # RIGHT COLUMN: Spiking Statistics (CV, firing rate, ISI, Fano factor)
    gs_statistics = gs_main[0, 1].subgridspec(4, 1, hspace=0.6)

    # Row 0: Firing rate distribution
    gs_firing_rate = gs_statistics[0, 0].subgridspec(1, n_cell_types, wspace=0.3)
    firing_rate_axes = [
        fig.add_subplot(gs_firing_rate[0, i]) for i in range(n_cell_types)
    ]
    plot_firing_rate_distribution(
        output_spikes=output_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        ax=firing_rate_axes,
    )
    # Add row title centered over the histograms
    bbox = firing_rate_axes[0].get_position()
    bbox_right = firing_rate_axes[-1].get_position()
    x_center = (bbox.x0 + bbox_right.x1) / 2
    y_top = bbox.y1 + 0.01
    fig.text(
        x_center,
        y_top,
        "Firing Rate Distribution",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    # Row 1: CV histogram
    gs_cv = gs_statistics[1, 0].subgridspec(1, n_cell_types, wspace=0.3)
    cv_axes = [fig.add_subplot(gs_cv[0, i]) for i in range(n_cell_types)]
    plot_cv_histogram(
        spike_trains=output_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        bins=50,
        ax=cv_axes,
    )
    # Add row title centered over the histograms
    bbox = cv_axes[0].get_position()
    bbox_right = cv_axes[-1].get_position()
    x_center = (bbox.x0 + bbox_right.x1) / 2
    y_top = bbox.y1 + 0.01
    fig.text(
        x_center,
        y_top,
        "Coefficient of Variation",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    # Row 2: ISI histogram
    gs_isi = gs_statistics[2, 0].subgridspec(1, n_cell_types, wspace=0.3)
    isi_axes = [fig.add_subplot(gs_isi[0, i]) for i in range(n_cell_types)]
    plot_isi_histogram(
        spike_trains=output_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        bins=50,
        ax=isi_axes,
    )
    # Add row title centered over the histograms
    bbox = isi_axes[0].get_position()
    bbox_right = isi_axes[-1].get_position()
    x_center = (bbox.x0 + bbox_right.x1) / 2
    y_top = bbox.y1 + 0.01
    fig.text(
        x_center, y_top, "Inter-Spike Interval", ha="center", va="bottom", fontsize=11
    )

    # Row 3: Fano factor vs window size
    gs_fano = gs_statistics[3, 0].subgridspec(1, n_cell_types, wspace=0.3)
    fano_axes = [fig.add_subplot(gs_fano[0, i]) for i in range(n_cell_types)]
    # Define window sizes for Fano factor analysis
    window_sizes = np.logspace(0, 3, 20).astype(int)  # 1 to 1000 steps
    plot_fano_factor_vs_window_size(
        spike_trains=output_spikes,
        window_sizes=window_sizes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        ax=fano_axes,
    )
    # Add row title centered over the histograms
    bbox = fano_axes[0].get_position()
    bbox_right = fano_axes[-1].get_position()
    x_center = (bbox.x0 + bbox_right.x1) / 2
    y_top = bbox.y1 + 0.01
    fig.text(x_center, y_top, "Fano Factor", ha="center", va="bottom", fontsize=11)

    return fig

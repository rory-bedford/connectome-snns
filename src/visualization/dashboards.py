"""Unified dashboard module for comprehensive network visualization.

This module creates dashboards displaying connectivity and activity plots
as subplots in unified figures.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from visualization.connectivity import (
    plot_weighted_connectivity,
    plot_input_count_histogram,
    plot_synaptic_input_histogram,
    plot_feedforward_connectivity,
)
from visualization.firing_statistics import (
    plot_psth,
    plot_firing_rate_distribution,
    plot_cv_histogram,
)
from visualization.neuronal_dynamics import (
    plot_spike_trains,
    plot_membrane_voltages,
)


# ===================== CONNECTIVITY DASHBOARD =====================


def create_connectivity_dashboard(
    connectivity_graph: NDArray[np.float32],
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    num_assemblies: int,
    recurrent_g_bar_by_type: dict[str, float],
    feedforward_g_bar_by_type: dict[str, float],
    plot_num_assemblies: int = 2,
    heatmap_inches: float = 6.0,
    plot_fraction: float = 0.1,
) -> plt.Figure:
    """Create a comprehensive connectivity dashboard with all connectivity plots.

    Args:
        connectivity_graph (NDArray[np.float32]): Binary connectivity matrix (N x N).
        weights (NDArray[np.float32]): Weight matrix (N x N).
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for recurrent neurons.
        input_cell_type_indices (NDArray[np.int32]): Array of cell type indices for input neurons.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.
        num_assemblies (int): Total number of assemblies in the network.
        recurrent_g_bar_by_type (dict[str, float]): Total g_bar for each recurrent cell type.
        feedforward_g_bar_by_type (dict[str, float]): Total g_bar for each feedforward cell type.
        plot_num_assemblies (int): Number of assemblies to display. Defaults to 2.
        heatmap_inches (float): Size of heatmaps in inches. Defaults to 6.0.
        plot_fraction (float): Fraction of neurons for feedforward connectivity. Defaults to 0.1.

    Returns:
        plt.Figure: Single comprehensive dashboard figure with all connectivity plots arranged.
    """
    # Create main dashboard figure
    fig = plt.figure(figsize=(24, 16))

    # Create layout:
    # - Left column (2 rows): connectivity heatmaps (feedforward above weighted)
    # - Right column (2 rows): input count histograms (3x2 grid)
    # - Bottom row (full width): conductance histograms (1x3 grid)
    gs_main = fig.add_gridspec(
        3,
        2,
        hspace=0.35,
        wspace=0.25,
        height_ratios=[1, 2.4, 1],
        width_ratios=[1, 1.5],
        left=0.05,
        right=0.98,
        top=0.94,
        bottom=0.05,
    )

    # Top left: Feedforward connectivity (without legend in dashboard)
    ax1 = fig.add_subplot(gs_main[0, 0])

    # Middle left: Weighted connectivity (1.5x the size of feedforward)
    ax2 = fig.add_subplot(gs_main[1, 0])

    # Plot both connectivity matrices
    plot_feedforward_connectivity(
        feedforward_weights=feedforward_weights,
        input_cell_type_indices=input_cell_type_indices,
        plot_fraction=plot_fraction,
        ax=ax1,
        show_legend=False,
        section_title="Weighted Connectomes",
        section_title_axes=[ax1, ax2],
    )
    # Add 20% padding on the right side of feedforward plot
    pos = ax1.get_position()
    ax1.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])

    plot_weighted_connectivity(
        weights=weights,
        cell_type_indices=cell_type_indices,
        num_assemblies=num_assemblies,
        plot_num_assemblies=plot_num_assemblies,
        heatmap_inches=heatmap_inches,
        ax=ax2,
    )

    # Right side: Input count histograms
    # This creates a grid of (n_input_sources x n_output_types)
    # With 2 recurrent + 1 feedforward = 3 input sources and 2 output types = 3x2 grid
    gs_histograms = gs_main[0:2, 1].subgridspec(3, 2, hspace=0.35, wspace=0.25)
    # Create axes in row-major order (what the function expects)
    histogram_axes = [
        fig.add_subplot(gs_histograms[i, j]) for i in range(3) for j in range(2)
    ]

    plot_input_count_histogram(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_cell_type_indices,
        cell_type_names=cell_type_names,
        input_cell_type_names=input_cell_type_names,
        ax=histogram_axes,
    )

    # Bottom row: Create 1x3 subplot grid for synaptic conductance histograms
    gs_conductance = gs_main[2, :].subgridspec(1, 3, wspace=0.25)
    conductance_axes = [fig.add_subplot(gs_conductance[0, j]) for j in range(3)]

    plot_synaptic_input_histogram(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_cell_type_indices,
        cell_type_names=cell_type_names,
        input_cell_type_names=input_cell_type_names,
        recurrent_g_bar_by_type=recurrent_g_bar_by_type,
        feedforward_g_bar_by_type=feedforward_g_bar_by_type,
        ax=conductance_axes,
    )

    return fig


# ===================== ACTIVITY DASHBOARD =====================


def create_activity_dashboard(
    output_spikes: NDArray[np.int32],
    input_spikes: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float,
    voltages: NDArray[np.float32] | None = None,
    neuron_types: NDArray[np.int32] | None = None,
    neuron_params: dict | None = None,
    window_size: float = 50.0,
    n_neurons_plot: int = 20,
    fraction: float = 1.0,
    random_seed: int = 42,
) -> plt.Figure:
    """Create a comprehensive activity dashboard with key firing plots.

    Args:
        output_spikes (NDArray[np.int32]): Output spike array with shape (batch, time, neurons).
        input_spikes (NDArray[np.int32]): Input spike array with shape (batch, time, neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step in milliseconds.
        voltages (NDArray[np.float32] | None): Membrane voltages. Defaults to None.
        neuron_types (NDArray[np.int32] | None): Neuron type indices. Defaults to None.
        neuron_params (dict | None): Neuron parameters. Defaults to None.
        window_size (float): Window size for PSTH in milliseconds. Defaults to 50.0.
        n_neurons_plot (int): Number of neurons to plot in raster. Defaults to 20.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        random_seed (int): Random seed for shuffling neurons. Defaults to 42.

    Returns:
        plt.Figure: Matplotlib figure object containing the activity dashboard.
    """
    # Create figure with 2x2 grid layout (or 2x3 if voltages provided)
    if voltages is not None and neuron_types is not None and neuron_params is not None:
        fig = plt.figure(figsize=(30, 16))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top left: Input spikes (mitral cells)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_spike_trains(
        spikes=input_spikes,
        dt=dt,
        cell_type="mitral",
        n_neurons_plot=min(10, input_spikes.shape[2]),
        fraction=fraction,
        title="Input Spike Trains (Mitral Cells)",
        ax=ax1,
    )

    # Top middle: Output network spikes (colored by cell type)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_spike_trains(
        spikes=output_spikes,
        dt=dt,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        n_neurons_plot=n_neurons_plot,
        fraction=fraction,
        random_seed=random_seed,
        title="Output Network Spike Trains (by cell type)",
        ax=ax2,
    )

    # Top right: Membrane voltages (if provided)
    if voltages is not None and neuron_types is not None and neuron_params is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        plot_membrane_voltages(
            voltages=voltages,
            spikes=output_spikes,
            neuron_types=neuron_types,
            delta_t=dt,
            neuron_params=neuron_params,
            n_neurons_plot=5,
            fraction=fraction,
            ax=ax3,
        )

    # Bottom left: PSTH
    ax4 = fig.add_subplot(gs[1, 0])
    plot_psth(
        spike_trains=output_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        window_size=window_size,
        dt=dt,
        ax=ax4,
    )

    # Bottom middle: Firing rate distribution
    ax5 = fig.add_subplot(gs[1, 1])
    plot_firing_rate_distribution(
        output_spikes=output_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        ax=ax5,
    )

    # Bottom right: CV histogram (if voltages provided, otherwise placeholder)
    if voltages is not None and neuron_types is not None and neuron_params is not None:
        ax6 = fig.add_subplot(gs[1, 2])
        plot_cv_histogram(
            spike_trains=output_spikes,
            cell_type_indices=cell_type_indices,
            cell_type_names=cell_type_names,
            dt=dt,
            bins=50,
            ax=ax6,
        )

    fig.suptitle("Activity Dashboard", fontsize=20, y=0.98)

    return fig

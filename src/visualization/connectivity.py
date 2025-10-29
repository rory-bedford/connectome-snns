"""Network structure and activity visualization functions.

This module contains plotting functions for analyzing network connectivity,
structure, and population-level activity patterns in spiking neural networks.
All functions return matplotlib figure objects for flexible handling.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_assembly_graph(
    connectivity_graph: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    num_assemblies: int,
    plot_num_assemblies: int = 2,
    heatmap_inches: float = 8.0,
) -> plt.Figure:
    """Plot the assembly graph structure.

    Args:
        connectivity_graph (NDArray[np.float32]): Binary connectivity matrix (N x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        num_assemblies (int): Total number of assemblies in the network.
        plot_num_assemblies (int): Number of assemblies to display. Defaults to 2.
        heatmap_inches (float): Size of the heatmap in inches. Defaults to 8.0.

    Returns:
        plt.Figure: Matplotlib figure object containing the assembly graph plot.
    """
    num_neurons = connectivity_graph.shape[0]
    neurons_per_assembly = num_neurons // num_assemblies
    plot_size_neurons = neurons_per_assembly * plot_num_assemblies

    # Apply signs based on source cell type: 0=excitatory (+1), 1=inhibitory (-1)
    cell_type_signs = np.where(cell_type_indices == 0, 1, -1)
    signed_connectivity = connectivity_graph * cell_type_signs[:, np.newaxis]

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

    return fig


def plot_weighted_connectivity(
    weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    num_assemblies: int,
    plot_num_assemblies: int = 2,
    heatmap_inches: float = 8.0,
) -> plt.Figure:
    """Plot the weighted connectivity matrix.

    Args:
        weights (NDArray[np.float32]): Weight matrix (N x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        num_assemblies (int): Total number of assemblies in the network.
        plot_num_assemblies (int): Number of assemblies to display. Defaults to 2.
        heatmap_inches (float): Size of the heatmap in inches. Defaults to 8.0.

    Returns:
        plt.Figure: Matplotlib figure object containing the weighted connectivity plot.
    """
    num_neurons = weights.shape[0]
    neurons_per_assembly = num_neurons // num_assemblies
    plot_size_neurons = neurons_per_assembly * plot_num_assemblies

    # Apply signs based on source cell type: 0=excitatory (+1), 1=inhibitory (-1)
    cell_type_signs = np.where(cell_type_indices == 0, 1, -1)
    signed_weights = weights * cell_type_signs[:, np.newaxis]

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

    return fig


def plot_input_count_histogram(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
) -> plt.Figure:
    """Plot histogram of input connection counts split by input and output cell types.

    Creates a 2D grid of histograms where:
    - Y-axis (rows): Input (presynaptic) cell types
    - X-axis (columns): Output (postsynaptic) cell types
    Each subplot shows the distribution of connection counts from a specific
    input type to a specific output type.

    Args:
        weights (NDArray[np.float32]): Recurrent weight matrix (N x N).
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for recurrent neurons.
        input_cell_type_indices (NDArray[np.int32]): Array of cell type indices for input neurons.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.

    Returns:
        plt.Figure: Matplotlib figure object containing the input count histogram.
    """
    # Get unique types
    unique_recurrent_types = np.unique(cell_type_indices)
    unique_feedforward_types = np.unique(input_cell_type_indices)
    unique_output_types = np.unique(cell_type_indices)

    # Combine all input types (recurrent + feedforward)
    all_input_types = []
    all_input_names = []

    for cell_type_idx in unique_recurrent_types:
        all_input_types.append(("recurrent", cell_type_idx))
        all_input_names.append(cell_type_names[cell_type_idx])

    for cell_type_idx in unique_feedforward_types:
        all_input_types.append(("feedforward", cell_type_idx))
        all_input_names.append(input_cell_type_names[cell_type_idx])

    n_input_types = len(all_input_types)
    n_output_types = len(unique_output_types)

    # Compute all input counts for each (input_type, output_type) pair
    input_counts_grid = {}
    for i, (input_source, input_idx) in enumerate(all_input_types):
        for j, output_idx in enumerate(unique_output_types):
            output_mask = cell_type_indices == output_idx

            if input_source == "recurrent":
                input_mask = cell_type_indices == input_idx
                counts = (weights[input_mask, :][:, output_mask] != 0).sum(axis=0)
            else:  # feedforward
                input_mask = input_cell_type_indices == input_idx
                counts = (feedforward_weights[input_mask, :][:, output_mask] != 0).sum(
                    axis=0
                )

            input_counts_grid[(i, j)] = counts

    # Calculate global max for shared x-axis
    global_x_max = max(
        [counts.max() for counts in input_counts_grid.values() if len(counts) > 0]
    )

    # Create bins of size 1
    bins = (
        np.arange(0, global_x_max + 2) - 0.5
    )  # Offset by 0.5 to center bins on integers

    # Calculate global y-max across all histograms
    max_count = 0
    for counts in input_counts_grid.values():
        if len(counts) > 0:
            hist_counts, _ = np.histogram(counts, bins=bins)
            max_count = max(max_count, hist_counts.max())

    # Create 2D grid of subplots
    fig, axes = plt.subplots(
        n_input_types,
        n_output_types,
        figsize=(4 * n_output_types, 3 * n_input_types),
        sharex=True,
        sharey=True,
    )

    # Ensure axes is always 2D
    if n_input_types == 1 and n_output_types == 1:
        axes = np.array([[axes]])
    elif n_input_types == 1:
        axes = axes.reshape(1, -1)
    elif n_output_types == 1:
        axes = axes.reshape(-1, 1)

    # Plot each (input, output) pair
    for i in range(n_input_types):
        for j in range(n_output_types):
            ax = axes[i, j]
            counts = input_counts_grid[(i, j)]

            if len(counts) > 0:
                mean_count = counts.mean()

                # Color based on input type
                input_source, input_idx = all_input_types[i]
                if input_source == "recurrent":
                    cell_name = cell_type_names[input_idx]
                    color = "#FF0000" if "excit" in cell_name.lower() else "#0000FF"
                else:
                    color = "#FF0000"  # Feedforward assumed excitatory

                ax.hist(
                    counts,
                    bins=bins,
                    color=color,
                    edgecolor="black",
                    alpha=0.6,
                )
                ax.axvline(
                    mean_count,
                    color="black",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.6,
                )

                # Add mean as text annotation
                ax.text(
                    0.95,
                    0.95,
                    f"μ={mean_count:.1f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )

            ax.set_xlim(0, global_x_max)
            ax.set_ylim(0, max_count * 1.1)
            ax.grid(True, alpha=0.3)

            # Add titles only to top row and left column
            if i == 0:
                ax.set_title(
                    cell_type_names[unique_output_types[j]], fontsize=10, pad=10
                )
            if j == 0:
                ax.set_ylabel(all_input_names[i], fontsize=10)

    # Add shared axis labels with proper spacing
    fig.text(0.5, 0.04, "Number of Inputs", ha="center", fontsize=12)
    # Calculate center position of the actual plot area (between left=0.14 and right=0.98)
    plot_center_x = 0.14 + (0.98 - 0.14) / 2
    fig.text(
        plot_center_x, 0.92, "Output (Postsynaptic) Cell Type", ha="center", fontsize=12
    )

    # Add row label
    fig.text(
        0.01,
        0.5,
        "Input (Presynaptic) Cell Type",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    fig.suptitle(
        "Input Connection Counts by Cell Type", fontsize=14, x=plot_center_x, y=0.97
    )

    plt.subplots_adjust(
        left=0.14, right=0.98, top=0.88, bottom=0.10, hspace=0.3, wspace=0.3
    )

    return fig


def plot_synaptic_input_histogram(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_g_bar_by_type: dict[str, float],
    feedforward_g_bar_by_type: dict[str, float],
) -> plt.Figure:
    """Plot histogram of total synaptic input to each neuron, separated by presynaptic cell type.

    Shows the distribution of input conductances from each presynaptic cell type
    (both recurrent and feedforward) across all postsynaptic neurons in the network.

    Args:
        weights (NDArray[np.float32]): Recurrent weight matrix (N x N).
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for recurrent neurons.
        input_cell_type_indices (NDArray[np.int32]): Array of cell type indices for input neurons.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.
        recurrent_g_bar_by_type (dict[str, float]): Total g_bar for each recurrent cell type.
        feedforward_g_bar_by_type (dict[str, float]): Total g_bar for each feedforward cell type.

    Returns:
        plt.Figure: Matplotlib figure object containing the synaptic input histogram.
    """
    # Prepare data for histogram
    unique_recurrent_types = np.unique(cell_type_indices)
    unique_feedforward_types = np.unique(input_cell_type_indices)

    scaled_conductances_by_type = []
    subplot_titles = []

    # Recurrent connections
    for cell_type_idx in unique_recurrent_types:
        mask = cell_type_indices == cell_type_idx
        cell_type_name = cell_type_names[cell_type_idx]
        total_g_bar = recurrent_g_bar_by_type[cell_type_name]

        # Sum incoming weights and multiply by total g_bar
        conductances = weights[mask, :].sum(axis=0) * total_g_bar
        scaled_conductances_by_type.append(conductances)
        subplot_titles.append(f"Recurrent: {cell_type_name}")

    # Feedforward connections
    for cell_type_idx in unique_feedforward_types:
        mask = input_cell_type_indices == cell_type_idx
        cell_type_name = input_cell_type_names[cell_type_idx]
        total_g_bar = feedforward_g_bar_by_type[cell_type_name]

        # Sum incoming weights and multiply by total g_bar
        # Only include inputs to excitatory postsynaptic neurons (exclude inhibitory at index 1)
        excitatory_mask = cell_type_indices == 0
        conductances = (
            feedforward_weights[mask, :][:, excitatory_mask].sum(axis=0) * total_g_bar
        )
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

    return fig


def plot_feedforward_connectivity(
    feedforward_weights: NDArray[np.float32],
    input_cell_type_indices: NDArray[np.int32],
    plot_fraction: float = 0.1,
) -> plt.Figure:
    """Plot feedforward connectivity matrix.

    Args:
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        input_cell_type_indices (NDArray[np.int32]): Array of cell type indices for input neurons.
        plot_fraction (float): Fraction of neurons to display. Defaults to 0.1.

    Returns:
        plt.Figure: Matplotlib figure object containing the feedforward connectivity plot.
    """
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

    return fig

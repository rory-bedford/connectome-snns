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
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot the assembly graph structure.

    Args:
        connectivity_graph (NDArray[np.float32]): Binary connectivity matrix (N x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        num_assemblies (int): Total number of assemblies in the network.
        plot_num_assemblies (int): Number of assemblies to display. Defaults to 2.
        heatmap_inches (float): Size of the heatmap in inches. Defaults to 8.0.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    num_neurons = connectivity_graph.shape[0]
    neurons_per_assembly = num_neurons // num_assemblies
    plot_size_neurons = neurons_per_assembly * plot_num_assemblies

    # Apply signs based on source cell type: 0=excitatory (+1), 1=inhibitory (-1)
    cell_type_signs = np.where(cell_type_indices == 0, 1, -1)
    signed_connectivity = connectivity_graph * cell_type_signs[:, np.newaxis]

    if ax is None:
        fig, ax = plt.subplots(figsize=(heatmap_inches * 1.3, heatmap_inches))
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

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

    return fig if return_fig else None


def plot_weighted_connectivity(
    weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    num_assemblies: int,
    plot_num_assemblies: int = 2,
    heatmap_inches: float = 8.0,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot the weighted connectivity matrix.

    Args:
        weights (NDArray[np.float32]): Weight matrix (N x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        num_assemblies (int): Total number of assemblies in the network.
        plot_num_assemblies (int): Number of assemblies to display. Defaults to 2.
        heatmap_inches (float): Size of the heatmap in inches. Defaults to 8.0.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    num_neurons = weights.shape[0]
    neurons_per_assembly = num_neurons // num_assemblies
    plot_size_neurons = neurons_per_assembly * plot_num_assemblies

    # Apply signs based on source cell type: 0=excitatory (+1), 1=inhibitory (-1)
    cell_type_signs = np.where(cell_type_indices == 0, 1, -1)
    signed_weights = weights * cell_type_signs[:, np.newaxis]

    if ax is None:
        fig, ax = plt.subplots(figsize=(heatmap_inches * 1.3, heatmap_inches))
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

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
    ax.set_xlabel("Postsynaptic Dp Cells")
    ax.set_ylabel("Presynaptic Dp Cells")
    ax.set_xticks([])
    ax.set_yticks([])

    return fig if return_fig else None


def plot_input_count_histogram(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
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
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
            Note: This function creates a multi-subplot figure internally, so ax parameter
            is accepted but ignored to maintain API consistency.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
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

    # Create 2D grid of subplots or use provided axes
    if ax is None:
        fig, axes = plt.subplots(
            n_input_types,
            n_output_types,
            figsize=(4 * n_output_types, 3 * n_input_types),
            sharex=True,
            sharey=True,
        )
        return_fig = True
        # Ensure axes is always 2D
        if n_input_types == 1 and n_output_types == 1:
            axes = np.array([[axes]])
        elif n_input_types == 1:
            axes = axes.reshape(1, -1)
        elif n_output_types == 1:
            axes = axes.reshape(-1, 1)
    elif isinstance(ax, list):
        # If a list of axes is provided, use them
        if len(ax) != n_input_types * n_output_types:
            raise ValueError(
                f"Expected {n_input_types * n_output_types} axes, got {len(ax)}"
            )
        # Reshape flat list into 2D array
        axes = np.array(ax).reshape(n_input_types, n_output_types)
        fig = axes[0, 0].get_figure()
        return_fig = False
    else:
        # Single axis provided - create our own figure
        fig, axes = plt.subplots(
            n_input_types,
            n_output_types,
            figsize=(4 * n_output_types, 3 * n_input_types),
            sharex=True,
            sharey=True,
        )
        return_fig = False
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
            ax.set_xlabel("Number of Inputs", fontsize=8)
            # Add Postsynaptic Cell Count to right-column axes
            if j > 0:
                ax.set_ylabel("Postsynaptic Cell Count", fontsize=8)

            # Add input/output labels as axis labels
            input_name = all_input_names[i]
            output_name = cell_type_names[unique_output_types[j]]

            # Make labels slightly bold only for mitral → inhibitory subplot
            is_mitral_to_inhibitory = (
                "mitral" in input_name.lower() and "inhibit" in output_name.lower()
            )
            weight = "semibold" if is_mitral_to_inhibitory else "normal"

            # Add title showing output type at the top of each column
            if i == 0:
                ax.set_title(
                    f"Output: {output_name}", fontsize=10, pad=8, fontweight=weight
                )

            # Add ylabel showing input type and cell count on the left of each row
            if j == 0:
                ax.set_ylabel(
                    f"Input: {input_name}\nPostsynaptic Cell Count", fontsize=8
                )
                if is_mitral_to_inhibitory:
                    ax.yaxis.label.set_weight("semibold")
                # Make the input type part of the label larger by setting it separately
                ax.yaxis.label.set_fontsize(10)

    # Add overall title
    if return_fig:
        fig.suptitle(
            "Input Connection Counts Split by Pre/Post Cell Type", fontsize=14, y=0.98
        )
        plt.tight_layout()
    else:
        # When in a dashboard, add a title above the subplot group
        bbox_first = axes[0, 0].get_position()
        bbox_last = axes[-1, -1].get_position()
        center_x = (bbox_first.x0 + bbox_last.x1) / 2

        # Main title at the top (above the top row - use bbox_first for top position)
        fig.text(
            center_x,
            bbox_first.y1 + 0.035,
            "Input Connection Counts by Cell Type",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    return fig if return_fig else None


def plot_synaptic_input_histogram(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_g_bar_by_type: dict[str, float],
    feedforward_g_bar_by_type: dict[str, float],
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
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
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
            Note: This function creates a multi-subplot figure internally, so ax parameter
            is accepted but ignored to maintain API consistency.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
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
        subplot_titles.append(f"recurrent / {cell_type_name}")

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
        subplot_titles.append(f"feedforward / {cell_type_name}")

    # Plot
    n_types = len(scaled_conductances_by_type)

    # Calculate global min/max for shared axes (excluding zeros)
    all_positive = np.concatenate([c[c > 0] for c in scaled_conductances_by_type])
    global_x_min = all_positive.min() if len(all_positive) > 0 else 1e-3
    global_x_max = all_positive.max() if len(all_positive) > 0 else 1.0

    # Create logarithmically-spaced bins
    log_bins = np.logspace(np.log10(global_x_min), np.log10(global_x_max), 21)

    # Create subplots (one per cell type) or use provided axes
    if ax is None:
        fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 4), sharey=True)
        return_fig = True
        if n_types == 1:
            axes = [axes]
    elif isinstance(ax, list):
        # If a list of axes is provided, use them
        if len(ax) != n_types:
            raise ValueError(f"Expected {n_types} axes, got {len(ax)}")
        axes = ax
        fig = axes[0].get_figure()
        return_fig = False
    else:
        # Single axis provided - create our own figure
        fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 4), sharey=True)
        return_fig = False
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

        # Determine if this is excitatory (recurrent excitatory or feedforward mitral)
        # Use red for excitatory/mitral, blue for inhibitory
        if "excit" in title.lower() or "mitral" in title.lower():
            hist_color = "#FF0000"
        else:
            hist_color = "#0000FF"

        ax.hist(
            conductances, bins=log_bins, color=hist_color, edgecolor="black", alpha=0.6
        )
        ax.axvline(
            mean_conductance,
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Mean = {mean_conductance:.2f} nS",
        )
        ax.set_title(title)
        ax.set_xlabel("Conductance (nS)")
        ax.set_xscale("log")
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(0, max_count * 1.1)
        ax.set_ylabel("Postsynaptic Cell Count")
        ax.legend()

    # Add title (only if we created our own figure)
    if return_fig:
        fig.suptitle(
            "Total Synaptic Input Conductance Split by Presynaptic Cell Types",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()
    else:
        # When in a dashboard, add a title above the subplot group
        bbox_first = axes[0].get_position()
        bbox_last = axes[-1].get_position()
        center_x = (bbox_first.x0 + bbox_last.x1) / 2
        top_y = bbox_first.y1 + 0.02
        fig.text(
            center_x,
            top_y,
            "Total Synaptic Input Conductance Split by Presynaptic Cell Types",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    return fig if return_fig else None


def plot_feedforward_connectivity(
    feedforward_weights: NDArray[np.float32],
    input_cell_type_indices: NDArray[np.int32],
    plot_fraction: float = 0.1,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
    section_title: str | None = None,
    section_title_axes: list[plt.Axes] | None = None,
) -> plt.Figure | None:
    """Plot feedforward connectivity matrix.

    Args:
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        input_cell_type_indices (NDArray[np.int32]): Array of cell type indices for input neurons.
        plot_fraction (float): Fraction of neurons to display. Defaults to 0.1.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
        show_legend (bool): Whether to show the colorbar legend. Defaults to True.
        section_title (str | None): Optional section title to add above subplot group in dashboard.
        section_title_axes (list[plt.Axes] | None): List of axes to span title across. Required if section_title provided.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False
    im = ax.imshow(
        signed_feedforward_weights[:n_input_plot, :n_output_plot],
        cmap="bwr",
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    if show_legend:
        cbar = plt.colorbar(im, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1])
        cbar.ax.set_yticklabels(["-1", "-0.5", "0", "+0.5", "+1"])
    ax.set_title(
        f"Feedforward Connectivity Matrix ({n_input_plot}/{n_input} mitral cells × {n_output_plot}/{n_output} Dp cells)"
    )
    ax.set_xlabel("Postsynaptic Dp Cells")
    ax.set_ylabel("Presynaptic Mitral Cells")
    ax.set_xticks([])
    ax.set_yticks([])

    # Add section title if requested (for dashboard)
    if section_title and section_title_axes:
        bbox_first = section_title_axes[0].get_position()
        center_x = (bbox_first.x0 + bbox_first.x1) / 2
        fig.text(
            center_x,
            bbox_first.y1 + 0.035,
            section_title,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None

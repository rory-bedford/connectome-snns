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
    num_assemblies: int | None = None,
    plot_num_assemblies: int = 2,
    heatmap_inches: float = 8.0,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot the assembly graph structure.

    Args:
        connectivity_graph (NDArray[np.float32]): Binary connectivity matrix (N x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        num_assemblies (int | None): Total number of assemblies in the network.
            If None, shows 10% of the matrix. Defaults to None.
        plot_num_assemblies (int): Number of assemblies to display (ignored if num_assemblies is None). Defaults to 2.
        heatmap_inches (float): Size of the heatmap in inches. Defaults to 8.0.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    num_neurons = connectivity_graph.shape[0]

    if num_assemblies is None:
        # Show 10% of the matrix
        plot_size_neurons = int(num_neurons * 0.1)
    else:
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

    # Create title based on whether we're showing assemblies or a percentage
    if num_assemblies is None:
        title = f"Assembly Graph Structure (showing {plot_size_neurons}/{num_neurons} cells)"
    else:
        title = f"Assembly Graph Structure (showing {plot_num_assemblies}/{num_assemblies} assemblies)"

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig if return_fig else None


def plot_weighted_connectivity(
    weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    num_assemblies: int | None = None,
    plot_num_assemblies: int = 2,
    plot_fraction: float = 0.1,
    heatmap_inches: float = 8.0,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot the weighted connectivity matrix.

    Args:
        weights (NDArray[np.float32]): Weight matrix (N x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        num_assemblies (int | None): Total number of assemblies in the network.
            If None, uses plot_fraction to determine matrix size. Defaults to None.
        plot_num_assemblies (int): Number of assemblies to display (only used if num_assemblies is not None). Defaults to 2.
        plot_fraction (float): Fraction of neurons to display (only used if num_assemblies is None). Defaults to 0.1.
        heatmap_inches (float): Size of the heatmap in inches. Defaults to 8.0.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    num_neurons = weights.shape[0]

    if num_assemblies is None:
        # Show specified fraction of the matrix
        plot_size_neurons = int(num_neurons * plot_fraction)
    else:
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
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1], shrink=0.5)
    cbar.ax.set_yticklabels(["-1", "-0.5", "0", "+0.5", "+1"])
    cbar.set_label("Weight", rotation=270, labelpad=15)

    # Create title based on whether we're showing assemblies or a percentage
    if num_assemblies is None:
        title = f"Weighted Connectivity Matrix (showing {plot_size_neurons}/{num_neurons} cells)"
    else:
        title = f"Weighted Connectivity Matrix (showing {plot_num_assemblies}/{num_assemblies} assemblies)"

    ax.set_title(title)
    ax.set_xlabel("Postsynaptic Dp Cells")
    ax.set_ylabel("Presynaptic Dp Cells")
    ax.set_xticks([])
    ax.set_yticks([])

    return fig if return_fig else None


def plot_input_count_pie_chart(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    ax: plt.Axes,
) -> None:
    """Plot pie chart showing mean input counts from each source.

    Args:
        weights: Recurrent weight matrix (N x N).
        feedforward_weights: Feedforward weight matrix (M x N).
        cell_type_indices: Array of cell type indices for recurrent neurons.
        input_cell_type_indices: Array of cell type indices for input neurons.
        cell_type_names: Names of recurrent cell types.
        input_cell_type_names: Names of input cell types.
        ax: Matplotlib axes to plot on.
    """
    # Prepare data: count inputs from each source
    input_counts_list = []
    input_labels = []
    colors = []

    # Feedforward inputs by input type
    for i, input_type_name in enumerate(input_cell_type_names):
        input_mask = input_cell_type_indices == i
        ff_counts = (feedforward_weights[input_mask, :] != 0).sum(axis=0)
        input_counts_list.append(ff_counts)
        input_labels.append(input_type_name)
        colors.append("#00AA00")  # Green for feedforward/mitral

    # Recurrent inputs by cell type
    for i, cell_type_name in enumerate(cell_type_names):
        cell_mask = cell_type_indices == i
        rec_counts = (weights[cell_mask, :] != 0).sum(axis=0)
        input_counts_list.append(rec_counts)
        input_labels.append(cell_type_name)

        # Assign colors: excitatory=red, inhibitory=blue
        if "excitatory" in cell_type_name.lower() or "exc" in cell_type_name.lower():
            colors.append("#FF0000")
        elif "inhibitory" in cell_type_name.lower() or "inh" in cell_type_name.lower():
            colors.append("#0000FF")
        else:
            colors.append("#888888")  # Gray for unknown types

    # Calculate means for pie chart
    means = np.array([counts.mean() for counts in input_counts_list])

    # Create pie chart with counts instead of percentages
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f"{val:.0f}"

        return my_autopct

    wedges, texts, autotexts = ax.pie(
        means,
        labels=input_labels,
        colors=[
            c + "99" if len(c) == 7 else c for c in colors
        ],  # Add alpha to hex colors
        autopct=make_autopct(means),
        startangle=90,
        textprops={"size": 9, "weight": "bold"},
    )

    # Make count text white and bold
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)

    ax.set_title("Mean Synapse Input Counts", fontsize=11, pad=10, fontweight="bold")


def plot_input_count_distribution(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    ax: plt.Axes,
) -> None:
    """Plot box and whisker plots showing distribution of input counts.

    Args:
        weights: Recurrent weight matrix (N x N).
        feedforward_weights: Feedforward weight matrix (M x N).
        cell_type_indices: Array of cell type indices for recurrent neurons.
        input_cell_type_indices: Array of cell type indices for input neurons.
        cell_type_names: Names of recurrent cell types.
        input_cell_type_names: Names of input cell types.
        ax: Matplotlib axes to plot on.
    """
    # Prepare data: count inputs from each source
    input_counts_list = []
    input_labels = []
    colors = []

    # Feedforward inputs by input type
    for i, input_type_name in enumerate(input_cell_type_names):
        input_mask = input_cell_type_indices == i
        ff_counts = (feedforward_weights[input_mask, :] != 0).sum(axis=0)
        input_counts_list.append(ff_counts)
        input_labels.append(input_type_name)
        colors.append("#00AA00")  # Green for feedforward/mitral

    # Recurrent inputs by cell type
    for i, cell_type_name in enumerate(cell_type_names):
        cell_mask = cell_type_indices == i
        rec_counts = (weights[cell_mask, :] != 0).sum(axis=0)
        input_counts_list.append(rec_counts)
        input_labels.append(cell_type_name)

        # Assign colors: excitatory=red, inhibitory=blue
        if "excitatory" in cell_type_name.lower() or "exc" in cell_type_name.lower():
            colors.append("#FF0000")
        elif "inhibitory" in cell_type_name.lower() or "inh" in cell_type_name.lower():
            colors.append("#0000FF")
        else:
            colors.append("#888888")  # Gray for unknown types

    # Create box and whisker plots (vertical, no outliers)
    positions = np.arange(len(input_labels))
    bp = ax.boxplot(
        input_counts_list,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        showfliers=False,  # Don't show outliers
        meanprops=dict(
            marker="D", markerfacecolor="black", markeredgecolor="black", markersize=6
        ),
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    # Color boxes to match pie chart
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")

    # Configure box plot axes
    ax.set_xticks(positions)
    ax.set_xticklabels(input_labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Input Count", fontsize=10)
    ax.set_title(
        "Synapse Input Count Distributions", fontsize=11, pad=10, fontweight="bold"
    )
    ax.set_ylim(bottom=0)  # Start y-axis at zero
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_charge_transfer_matrix(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_g_bar_by_type: dict[str, float],
    feedforward_g_bar_by_type: dict[str, float],
    recurrent_tau_rise_by_type: dict[str, float],
    recurrent_tau_decay_by_type: dict[str, float],
    feedforward_tau_rise_by_type: dict[str, float],
    feedforward_tau_decay_by_type: dict[str, float],
    recurrent_E_syn_by_type: dict[str, float],
    feedforward_E_syn_by_type: dict[str, float],
    recurrent_V_mem_by_type: dict[str, float],
    ax: plt.Axes,
) -> None:
    """Plot confusion matrix of mean charge transfer per spike for each input-output pair.

    Args:
        weights: Recurrent weight matrix (N x N).
        feedforward_weights: Feedforward weight matrix (M x N).
        cell_type_indices: Array of cell type indices for recurrent neurons.
        input_cell_type_indices: Array of cell type indices for input neurons.
        cell_type_names: Names of recurrent cell types (output/target neurons).
        input_cell_type_names: Names of input cell types (source neurons).
        recurrent_g_bar_by_type: Total g_bar for each recurrent cell type.
        feedforward_g_bar_by_type: Total g_bar for each feedforward cell type.
        recurrent_tau_rise_by_type: Rise time constant for each recurrent cell type.
        recurrent_tau_decay_by_type: Decay time constant for each recurrent cell type.
        feedforward_tau_rise_by_type: Rise time constant for each feedforward cell type.
        feedforward_tau_decay_by_type: Decay time constant for each feedforward cell type.
        recurrent_E_syn_by_type: Reversal potential for each recurrent source cell type.
        feedforward_E_syn_by_type: Reversal potential for each feedforward source cell type.
        recurrent_V_mem_by_type: Mean membrane potential for each recurrent target cell type.
        ax: Matplotlib axes to plot on.
    """
    # Build matrix: rows = output cell types, cols = input sources
    n_outputs = len(cell_type_names)
    n_inputs = len(input_cell_type_names) + len(cell_type_names)  # FF + recurrent

    matrix = np.zeros((n_outputs, n_inputs))
    input_labels = list(input_cell_type_names) + list(cell_type_names)

    # For each output (target) cell type
    for out_idx, output_cell_type in enumerate(cell_type_names):
        # Get target neurons of this type
        target_mask = cell_type_indices == out_idx
        n_targets = target_mask.sum()

        if n_targets == 0:
            continue

        # Get mean membrane potential for this output type
        V_mem = recurrent_V_mem_by_type.get(output_cell_type, -55.0)

        col_idx = 0

        # Process feedforward inputs
        for ff_idx, input_type_name in enumerate(input_cell_type_names):
            input_mask = input_cell_type_indices == ff_idx

            # Get weights from this input type to these targets
            ff_weights_to_targets = feedforward_weights[input_mask, :][:, target_mask]

            if ff_weights_to_targets.size == 0:
                col_idx += 1
                continue

            # Get parameters for this source type
            g_bar = feedforward_g_bar_by_type.get(input_type_name, 1.0)
            tau_rise = feedforward_tau_rise_by_type.get(input_type_name, 1.0)
            tau_decay = feedforward_tau_decay_by_type.get(input_type_name, 5.0)
            E_syn = feedforward_E_syn_by_type.get(input_type_name, 0.0)

            # Calculate synaptic drive per connection
            r = tau_rise / tau_decay
            peak_norm = r ** (r / (1 - r)) - r ** (1 / (1 - r))
            area = tau_decay - tau_rise
            driving_force = V_mem - E_syn

            # Mean charge transfer per spike across all I/O pairs
            charge_per_spike = (
                ff_weights_to_targets * g_bar * area / peak_norm
            ) * driving_force
            matrix[out_idx, col_idx] = charge_per_spike.mean()

            col_idx += 1

        # Process recurrent inputs
        for rec_idx, input_cell_type in enumerate(cell_type_names):
            source_mask = cell_type_indices == rec_idx

            # Get weights from this source type to these targets
            rec_weights_to_targets = weights[source_mask, :][:, target_mask]

            if rec_weights_to_targets.size == 0:
                col_idx += 1
                continue

            # Get parameters for this source type
            g_bar = recurrent_g_bar_by_type.get(input_cell_type, 1.0)
            tau_rise = recurrent_tau_rise_by_type.get(input_cell_type, 1.0)
            tau_decay = recurrent_tau_decay_by_type.get(input_cell_type, 5.0)
            E_syn = recurrent_E_syn_by_type.get(input_cell_type, 0.0)

            # Calculate synaptic drive per connection
            r = tau_rise / tau_decay
            peak_norm = r ** (r / (1 - r)) - r ** (1 / (1 - r))
            area = tau_decay - tau_rise
            driving_force = V_mem - E_syn

            # Mean charge transfer per spike across all I/O pairs
            charge_per_spike = (
                rec_weights_to_targets * g_bar * area / peak_norm
            ) * driving_force
            matrix[out_idx, col_idx] = charge_per_spike.mean()

            col_idx += 1

    # Plot heatmap
    im = ax.imshow(
        matrix,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-np.abs(matrix).max(),
        vmax=np.abs(matrix).max(),
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Charge Transfer (fC/spike)", fontsize=9)

    # Set ticks and labels
    ax.set_xticks(np.arange(n_inputs))
    ax.set_yticks(np.arange(n_outputs))
    ax.set_xticklabels(input_labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(cell_type_names, fontsize=9)

    # Add text annotations
    for i in range(n_outputs):
        for j in range(n_inputs):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    ax.set_xlabel("Input Source", fontsize=10)
    ax.set_ylabel("Output Target", fontsize=10)
    ax.set_title(
        "Charge Transfer per Spike (I/O Pairs)", fontsize=11, pad=10, fontweight="bold"
    )


def plot_synaptic_drive_pie_chart(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_g_bar_by_type: dict[str, float],
    feedforward_g_bar_by_type: dict[str, float],
    recurrent_tau_rise_by_type: dict[str, float],
    recurrent_tau_decay_by_type: dict[str, float],
    feedforward_tau_rise_by_type: dict[str, float],
    feedforward_tau_decay_by_type: dict[str, float],
    recurrent_E_syn_by_type: dict[str, float],
    feedforward_E_syn_by_type: dict[str, float],
    mean_membrane_potential: float,
    ax: plt.Axes,
) -> None:
    """Plot pie chart showing mean synaptic drive from each source.

    Args:
        weights: Recurrent weight matrix (N x N).
        feedforward_weights: Feedforward weight matrix (M x N).
        cell_type_indices: Array of cell type indices for recurrent neurons.
        input_cell_type_indices: Array of cell type indices for input neurons.
        cell_type_names: Names of recurrent cell types.
        input_cell_type_names: Names of input cell types.
        recurrent_g_bar_by_type: Total g_bar for each recurrent cell type.
        feedforward_g_bar_by_type: Total g_bar for each feedforward cell type.
        recurrent_tau_rise_by_type: Rise time constant for each recurrent cell type.
        recurrent_tau_decay_by_type: Decay time constant for each recurrent cell type.
        feedforward_tau_rise_by_type: Rise time constant for each feedforward cell type.
        feedforward_tau_decay_by_type: Decay time constant for each feedforward cell type.
        recurrent_E_syn_by_type: Reversal potential for each recurrent cell type.
        feedforward_E_syn_by_type: Reversal potential for each feedforward cell type.
        mean_membrane_potential: Mean membrane potential in mV (for driving force calculation).
        ax: Matplotlib axes to plot on.
    """
    # Prepare data: sum weights and calculate synaptic drive for each source
    drive_sums_list = []
    input_labels = []
    colors = []

    # Feedforward synaptic drive by input type
    for i, input_type_name in enumerate(input_cell_type_names):
        input_mask = input_cell_type_indices == i
        ff_weight_sums = feedforward_weights[input_mask, :].sum(axis=0)
        if (
            feedforward_g_bar_by_type is None
            or input_type_name not in feedforward_g_bar_by_type
        ):
            raise ValueError(
                f"g_bar value for feedforward input type '{input_type_name}' is missing. "
                "This is a critical error - all cell types must have g_bar values."
            )
        if (
            input_type_name not in feedforward_tau_rise_by_type
            or input_type_name not in feedforward_tau_decay_by_type
        ):
            raise ValueError(
                f"Tau values for feedforward input type '{input_type_name}' are missing. "
                "This is a critical error - all cell types must have tau_rise and tau_decay values."
            )
        g_bar = feedforward_g_bar_by_type[input_type_name]
        tau_rise = feedforward_tau_rise_by_type[input_type_name]
        tau_decay = feedforward_tau_decay_by_type[input_type_name]
        if input_type_name not in feedforward_E_syn_by_type:
            raise ValueError(
                f"E_syn value for feedforward input type '{input_type_name}' is missing. "
                "This is a critical error - all cell types must have E_syn values."
            )
        E_syn = feedforward_E_syn_by_type[input_type_name]

        # Calculate synaptic drive in charge units with driving force
        r = tau_rise / tau_decay
        peak_norm = r ** (r / (1 - r)) - r ** (1 / (1 - r))
        area = tau_decay - tau_rise
        driving_force = mean_membrane_potential - E_syn
        ff_drive = (ff_weight_sums * g_bar * area / peak_norm) * driving_force
        drive_sums_list.append(ff_drive)
        input_labels.append(input_type_name)
        colors.append("#00AA00")  # Green for feedforward/mitral

    # Recurrent synaptic drive by cell type
    for i, cell_type_name in enumerate(cell_type_names):
        cell_mask = cell_type_indices == i
        rec_weight_sums = weights[cell_mask, :].sum(axis=0)
        if (
            recurrent_g_bar_by_type is None
            or cell_type_name not in recurrent_g_bar_by_type
        ):
            raise ValueError(
                f"g_bar value for recurrent cell type '{cell_type_name}' is missing. "
                "This is a critical error - all cell types must have g_bar values."
            )
        if (
            cell_type_name not in recurrent_tau_rise_by_type
            or cell_type_name not in recurrent_tau_decay_by_type
        ):
            raise ValueError(
                f"Tau values for recurrent cell type '{cell_type_name}' are missing. "
                "This is a critical error - all cell types must have tau_rise and tau_decay values."
            )
        g_bar = recurrent_g_bar_by_type[cell_type_name]
        tau_rise = recurrent_tau_rise_by_type[cell_type_name]
        tau_decay = recurrent_tau_decay_by_type[cell_type_name]
        if cell_type_name not in recurrent_E_syn_by_type:
            raise ValueError(
                f"E_syn value for recurrent cell type '{cell_type_name}' is missing. "
                "This is a critical error - all cell types must have E_syn values."
            )
        E_syn = recurrent_E_syn_by_type[cell_type_name]

        # Calculate synaptic drive in charge units with driving force
        r = tau_rise / tau_decay
        peak_norm = r ** (r / (1 - r)) - r ** (1 / (1 - r))
        area = tau_decay - tau_rise
        driving_force = mean_membrane_potential - E_syn
        rec_drive = (rec_weight_sums * g_bar * area / peak_norm) * driving_force
        drive_sums_list.append(rec_drive)
        input_labels.append(cell_type_name)

        # Assign colors: excitatory=red, inhibitory=blue
        if "excitatory" in cell_type_name.lower() or "exc" in cell_type_name.lower():
            colors.append("#FF0000")
        elif "inhibitory" in cell_type_name.lower() or "inh" in cell_type_name.lower():
            colors.append("#0000FF")
        else:
            colors.append("#888888")  # Gray for unknown types

    # Calculate means for pie chart (use absolute values since pie chart can't show negative)
    means = np.array([np.abs(drive).mean() for drive in drive_sums_list])

    # Create pie chart with values instead of percentages
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = pct * total / 100.0
            return f"{val:.1f}"

        return my_autopct

    wedges, texts, autotexts = ax.pie(
        means,
        labels=input_labels,
        colors=[
            c + "99" if len(c) == 7 else c for c in colors
        ],  # Add alpha to hex colors
        autopct=make_autopct(means),
        startangle=90,
        textprops={"size": 9, "weight": "bold"},
    )

    # Make count text white and bold
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)

    ax.set_title("Mean Synaptic Drive (Charge)", fontsize=11, pad=10, fontweight="bold")


def plot_synaptic_drive_distribution(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_g_bar_by_type: dict[str, float],
    feedforward_g_bar_by_type: dict[str, float],
    recurrent_tau_rise_by_type: dict[str, float],
    recurrent_tau_decay_by_type: dict[str, float],
    feedforward_tau_rise_by_type: dict[str, float],
    feedforward_tau_decay_by_type: dict[str, float],
    recurrent_E_syn_by_type: dict[str, float],
    feedforward_E_syn_by_type: dict[str, float],
    mean_membrane_potential: float,
    ax: plt.Axes,
) -> None:
    """Plot violin plots showing distribution of synaptic drive.

    Args:
        weights: Recurrent weight matrix (N x N).
        feedforward_weights: Feedforward weight matrix (M x N).
        cell_type_indices: Array of cell type indices for recurrent neurons.
        input_cell_type_indices: Array of cell type indices for input neurons.
        cell_type_names: Names of recurrent cell types.
        input_cell_type_names: Names of input cell types.
        recurrent_g_bar_by_type: Total g_bar for each recurrent cell type.
        feedforward_g_bar_by_type: Total g_bar for each feedforward cell type.
        recurrent_tau_rise_by_type: Rise time constant for each recurrent cell type.
        recurrent_tau_decay_by_type: Decay time constant for each recurrent cell type.
        feedforward_tau_rise_by_type: Rise time constant for each feedforward cell type.
        feedforward_tau_decay_by_type: Decay time constant for each feedforward cell type.
        recurrent_E_syn_by_type: Reversal potential for each recurrent cell type.
        feedforward_E_syn_by_type: Reversal potential for each feedforward cell type.
        mean_membrane_potential: Mean membrane potential in mV (for driving force calculation).
        ax: Matplotlib axes to plot on.
    """
    # Prepare data: sum weights and calculate synaptic drive for each source
    conductance_sums_list = []
    input_labels = []
    colors = []

    # Feedforward synaptic drive by input type
    for i, input_type_name in enumerate(input_cell_type_names):
        input_mask = input_cell_type_indices == i
        ff_weight_sums = feedforward_weights[input_mask, :].sum(axis=0)
        if (
            feedforward_g_bar_by_type is None
            or input_type_name not in feedforward_g_bar_by_type
        ):
            raise ValueError(
                f"g_bar value for feedforward input type '{input_type_name}' is missing. "
                "This is a critical error - all cell types must have g_bar values."
            )
        if (
            input_type_name not in feedforward_tau_rise_by_type
            or input_type_name not in feedforward_tau_decay_by_type
        ):
            raise ValueError(
                f"Tau values for feedforward input type '{input_type_name}' are missing. "
                "This is a critical error - all cell types must have tau_rise and tau_decay values."
            )
        g_bar = feedforward_g_bar_by_type[input_type_name]
        tau_rise = feedforward_tau_rise_by_type[input_type_name]
        tau_decay = feedforward_tau_decay_by_type[input_type_name]
        if input_type_name not in feedforward_E_syn_by_type:
            raise ValueError(
                f"E_syn value for feedforward input type '{input_type_name}' is missing. "
                "This is a critical error - all cell types must have E_syn values."
            )
        E_syn = feedforward_E_syn_by_type[input_type_name]

        # Calculate synaptic drive: (g_bar * weight * (tau_decay - tau_rise)) / peak_norm * driving_force
        # Peak norm: r^(r/(1-r)) - r^(1/(1-r))
        r = tau_rise / tau_decay
        peak_norm = r ** (r / (1 - r)) - r ** (1 / (1 - r))
        area = tau_decay - tau_rise
        # Synaptic drive in charge units: (g_bar * weight * area / peak_norm) * (V_mem - E_syn)
        # Units: (nS * unitless * ms / unitless) * mV = pA * ms = fC
        driving_force = mean_membrane_potential - E_syn
        ff_drive = (ff_weight_sums * g_bar * area / peak_norm) * driving_force
        conductance_sums_list.append(ff_drive)
        input_labels.append(input_type_name)
        colors.append("#00AA00")  # Green for feedforward/mitral

    # Recurrent synaptic drive by cell type
    for i, cell_type_name in enumerate(cell_type_names):
        cell_mask = cell_type_indices == i
        rec_weight_sums = weights[cell_mask, :].sum(axis=0)
        if (
            recurrent_g_bar_by_type is None
            or cell_type_name not in recurrent_g_bar_by_type
        ):
            raise ValueError(
                f"g_bar value for recurrent cell type '{cell_type_name}' is missing. "
                "This is a critical error - all cell types must have g_bar values."
            )
        if (
            cell_type_name not in recurrent_tau_rise_by_type
            or cell_type_name not in recurrent_tau_decay_by_type
        ):
            raise ValueError(
                f"Tau values for recurrent cell type '{cell_type_name}' are missing. "
                "This is a critical error - all cell types must have tau_rise and tau_decay values."
            )
        g_bar = recurrent_g_bar_by_type[cell_type_name]
        tau_rise = recurrent_tau_rise_by_type[cell_type_name]
        tau_decay = recurrent_tau_decay_by_type[cell_type_name]
        if cell_type_name not in recurrent_E_syn_by_type:
            raise ValueError(
                f"E_syn value for recurrent cell type '{cell_type_name}' is missing. "
                "This is a critical error - all cell types must have E_syn values."
            )
        E_syn = recurrent_E_syn_by_type[cell_type_name]

        # Calculate synaptic drive with driving force
        # Peak norm: r^(r/(1-r)) - r^(1/(1-r))
        r = tau_rise / tau_decay
        peak_norm = r ** (r / (1 - r)) - r ** (1 / (1 - r))
        area = tau_decay - tau_rise
        # Synaptic drive in charge units: (g_bar * weight * area / peak_norm) * (V_mem - E_syn)
        # Units: (nS * unitless * ms / unitless) * mV = pA * ms = fC
        driving_force = mean_membrane_potential - E_syn
        rec_drive = (rec_weight_sums * g_bar * area / peak_norm) * driving_force
        conductance_sums_list.append(rec_drive)
        input_labels.append(cell_type_name)

        # Assign colors: excitatory=red, inhibitory=blue
        if "excitatory" in cell_type_name.lower() or "exc" in cell_type_name.lower():
            colors.append("#FF0000")
        elif "inhibitory" in cell_type_name.lower() or "inh" in cell_type_name.lower():
            colors.append("#0000FF")
        else:
            colors.append("#888888")  # Gray for unknown types

    # Create violin plots
    positions = np.arange(len(input_labels))
    parts = ax.violinplot(
        conductance_sums_list,
        positions=positions,
        widths=0.6,
        showmeans=True,
        showmedians=True,
    )

    # Color violin bodies to match pie chart
    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    # Style the mean and median lines
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(2)
    parts["cmedians"].set_color("darkgray")
    parts["cmedians"].set_linewidth(2)
    parts["cbars"].set_color("black")
    parts["cbars"].set_linewidth(1.5)
    parts["cmaxes"].set_color("black")
    parts["cmaxes"].set_linewidth(1.5)
    parts["cmins"].set_color("black")
    parts["cmins"].set_linewidth(1.5)

    # Configure axes
    ax.set_xticks(positions)
    ax.set_xticklabels(input_labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Synaptic Drive (fC)", fontsize=10)
    ax.set_title("Synaptic Drive Distributions", fontsize=11, pad=10, fontweight="bold")
    ax.set_ylim(bottom=0)  # Start y-axis at zero
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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

    # Use green colormap for feedforward (mitral cells)
    # Just show weight magnitude (all mitral cells are excitatory)

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
        feedforward_weights[:n_input_plot, :n_output_plot],
        cmap="Greens",
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    # Always show colorbar legend
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    cbar.set_label("Weight", rotation=270, labelpad=15)
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


def plot_weight_distribution_by_input_type(
    recurrent_weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    connectome_mask: NDArray[np.bool_],
    feedforward_mask: NDArray[np.bool_],
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot violin plot showing distribution of weight values by input cell type.

    Args:
        recurrent_weights (NDArray[np.float32]): Recurrent weight matrix (N x N).
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for recurrent neurons.
        input_cell_type_indices (NDArray[np.int32]): Array of cell type indices for input neurons.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.
        connectome_mask (NDArray[np.bool_]): Binary mask for valid recurrent connections.
        feedforward_mask (NDArray[np.bool_]): Binary mask for valid feedforward connections.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

    # Define color mapping for cell types (use more saturated colors matching legends)
    color_map = {
        "mitral": "#00AA00",  # Saturated green
        "excitatory": "#DD0000",  # Saturated red
        "inhibitory": "#0000DD",  # Saturated blue
    }

    # Collect non-zero weights for each input cell type
    weight_data = []
    labels = []
    colors = []

    # Process feedforward input types
    for i, cell_type_name in enumerate(input_cell_type_names):
        # Get weights from this input type and apply mask
        type_mask = input_cell_type_indices == i
        type_weights = feedforward_weights[type_mask, :]
        masked_weights = type_weights * feedforward_mask[type_mask, :]
        non_zero_weights = masked_weights[masked_weights > 0]

        if len(non_zero_weights) > 0:
            weight_data.append(non_zero_weights)
            labels.append(cell_type_name)

            # Assign color based on cell type name
            color = "gray"  # default
            for key, val in color_map.items():
                if key.lower() in cell_type_name.lower():
                    color = val
                    break
            colors.append(color)

    # Process recurrent cell types (as input sources)
    for i, cell_type_name in enumerate(cell_type_names):
        # Get weights from this recurrent cell type and apply mask
        type_mask = cell_type_indices == i
        type_weights = recurrent_weights[type_mask, :]
        masked_weights = type_weights * connectome_mask[type_mask, :]
        non_zero_weights = masked_weights[masked_weights > 0]

        if len(non_zero_weights) > 0:
            weight_data.append(non_zero_weights)
            labels.append(cell_type_name)

            # Assign color based on cell type name
            color = "gray"  # default
            for key, val in color_map.items():
                if key.lower() in cell_type_name.lower():
                    color = val
                    break
            colors.append(color)

    # Create violin plot
    if weight_data:
        parts = ax.violinplot(
            weight_data,
            positions=np.arange(len(weight_data)),
            showmeans=True,
            showmedians=False,
            widths=0.7,
            bw_method=0.5,  # Smoothing parameter (smaller = smoother)
        )

        # Color violin bodies
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)

        # Style only mean lines
        if "cmeans" in parts:
            parts["cmeans"].set_edgecolor("black")
            parts["cmeans"].set_linewidth(1.5)

        # Remove the range bars
        for partname in ("cbars", "cmins", "cmaxes"):
            if partname in parts:
                parts[partname].set_visible(False)

        # Set y-axis limits based on percentiles to exclude extreme outliers
        all_weights = np.concatenate(weight_data)
        y_min = np.percentile(all_weights, 1)  # 1st percentile
        y_max = np.percentile(all_weights, 99)  # 99th percentile
        # Add some padding
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Weight Value")
    ax.set_title("Weight Distribution by Input Type", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None


def plot_weight_statistics_matrix(
    recurrent_weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    ax_mean: plt.Axes | None = None,
    ax_std: plt.Axes | None = None,
) -> tuple[plt.Figure | None, plt.Figure | None]:
    """Plot mean and std of weights by presynaptic to postsynaptic cell type.

    Args:
        recurrent_weights (NDArray[np.float32]): Recurrent weight matrix (N x N).
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for recurrent neurons.
        input_cell_type_indices (NDArray[np.int32]): Array of cell type indices for input neurons.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.
        ax_mean (plt.Axes | None): Matplotlib axes for mean plot. If None, creates new figure.
        ax_std (plt.Axes | None): Matplotlib axes for std plot. If None, creates new figure.

    Returns:
        tuple[plt.Figure | None, plt.Figure | None]: Matplotlib figure objects if axes are None, otherwise None.
    """
    # Define color mapping for cell types
    color_map = {
        "mitral": "#00AA00",  # Saturated green
        "excitatory": "#DD0000",  # Saturated red
        "inhibitory": "#0000DD",  # Saturated blue
    }

    # Combined presynaptic types (feedforward + recurrent)
    all_pre_types = list(input_cell_type_names) + list(cell_type_names)
    n_pre_types = len(all_pre_types)
    n_post_types = len(cell_type_names)

    # Initialize matrices
    mean_matrix = np.zeros((n_pre_types, n_post_types))
    std_matrix = np.zeros((n_pre_types, n_post_types))

    # Process feedforward connections
    for i, pre_type_name in enumerate(input_cell_type_names):
        pre_mask = input_cell_type_indices == i
        for j, post_type_name in enumerate(cell_type_names):
            post_mask = cell_type_indices == j
            # Get weights from pre type i to all post cells of type j
            # Sum over presynaptic cells to get total input per postsynaptic cell
            weights = feedforward_weights[pre_mask, :][:, post_mask]
            sums_per_post_cell = weights.sum(axis=0)  # Sum over presynaptic cells
            if len(sums_per_post_cell) > 0:
                mean_matrix[i, j] = sums_per_post_cell.mean()
                std_matrix[i, j] = sums_per_post_cell.std()

    # Process recurrent connections
    for i, pre_type_name in enumerate(cell_type_names):
        pre_idx = len(input_cell_type_names) + i
        pre_mask = cell_type_indices == i
        for j, post_type_name in enumerate(cell_type_names):
            post_mask = cell_type_indices == j
            # Get weights from pre type i to all post cells of type j
            # Sum over presynaptic cells to get total input per postsynaptic cell
            weights = recurrent_weights[pre_mask, :][:, post_mask]
            sums_per_post_cell = weights.sum(axis=0)  # Sum over presynaptic cells
            if len(sums_per_post_cell) > 0:
                mean_matrix[pre_idx, j] = sums_per_post_cell.mean()
                std_matrix[pre_idx, j] = sums_per_post_cell.std()

    # Assign row colors based on presynaptic type
    row_colors = []
    for type_name in all_pre_types:
        color = "gray"
        for key, val in color_map.items():
            if key.lower() in type_name.lower():
                color = val
                break
        row_colors.append(color)

    # Assign column colors based on postsynaptic type
    col_colors = []
    for type_name in cell_type_names:
        color = "gray"
        for key, val in color_map.items():
            if key.lower() in type_name.lower():
                color = val
                break
        col_colors.append(color)

    # Compute individual color scales for each matrix
    vmax_mean = int(np.ceil(mean_matrix.max()))
    vmax_std = int(np.ceil(std_matrix.max()))

    # Plot mean matrix
    if ax_mean is None:
        fig_mean, ax_mean = plt.subplots(figsize=(8, 6))
        return_fig_mean = True
    else:
        fig_mean = ax_mean.get_figure()
        return_fig_mean = False

    im_mean = ax_mean.imshow(
        mean_matrix,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=vmax_mean,
    )
    ax_mean.set_yticks(np.arange(n_pre_types))
    ax_mean.set_yticklabels(all_pre_types)
    ax_mean.set_xticks(np.arange(n_post_types))
    ax_mean.set_xticklabels(cell_type_names, rotation=45, ha="right")
    ax_mean.set_xlabel("Postsynaptic (Recurrent) Cell Type")
    ax_mean.set_ylabel("Presynaptic Cell Type")
    ax_mean.set_title("Mean Total Input by Connection Type", fontweight="bold")

    # Color y-axis labels (presynaptic)
    for i, (tick_label, color) in enumerate(zip(ax_mean.get_yticklabels(), row_colors)):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    # Color x-axis labels (postsynaptic)
    for i, (tick_label, color) in enumerate(zip(ax_mean.get_xticklabels(), col_colors)):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    # Colorbar on left plot
    cbar_mean = plt.colorbar(im_mean, ax=ax_mean)
    cbar_mean.set_label("", rotation=270, labelpad=15)

    # Plot std matrix
    if ax_std is None:
        fig_std, ax_std = plt.subplots(figsize=(8, 6))
        return_fig_std = True
    else:
        fig_std = ax_std.get_figure()
        return_fig_std = False

    im_std = ax_std.imshow(
        std_matrix,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=vmax_std,
    )
    ax_std.set_yticks([])  # No y-axis ticks
    ax_std.set_xticks(np.arange(n_post_types))
    ax_std.set_xticklabels(cell_type_names, rotation=45, ha="right")
    ax_std.set_xlabel("Postsynaptic (Recurrent) Cell Type")
    ax_std.set_title("Std of Total Input by Connection Type", fontweight="bold")

    # Color x-axis labels (postsynaptic)
    for i, (tick_label, color) in enumerate(zip(ax_std.get_xticklabels(), col_colors)):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    # Single colorbar on right plot only
    cbar_std = plt.colorbar(im_std, ax=ax_std)
    cbar_std.set_label("", rotation=270, labelpad=15)  # No label

    if return_fig_mean:
        plt.tight_layout()
    if return_fig_std:
        plt.tight_layout()

    return (fig_mean if return_fig_mean else None, fig_std if return_fig_std else None)

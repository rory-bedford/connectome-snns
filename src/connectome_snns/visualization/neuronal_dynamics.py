"""Neuronal dynamics visualization functions."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from numpy.typing import NDArray

from connectome_snns.configs.conductance_based import (
    EXCITATORY_SYNAPSE_TYPES,
    INHIBITORY_SYNAPSE_TYPES,
)


def _round_to_nice_limit(value: float) -> float:
    """Round a value up to a nice round number (power of 10 times 1, 2, or 5).

    Args:
        value (float): The value to round up

    Returns:
        float: Rounded value
    """
    if value <= 0:
        return 1.0

    # Find the order of magnitude
    magnitude = 10 ** np.floor(np.log10(value))

    # Normalize to range [1, 10)
    normalized = value / magnitude

    # Round up to nearest nice number (1, 2, 5, 10)
    if normalized <= 1:
        nice_normalized = 1
    elif normalized <= 2:
        nice_normalized = 2
    elif normalized <= 5:
        nice_normalized = 5
    else:
        nice_normalized = 10

    return nice_normalized * magnitude


def plot_membrane_voltages(
    voltages: NDArray[np.float32],
    spikes: NDArray[np.int32],
    neuron_types: NDArray[np.int32],
    delta_t: float,
    neuron_params: dict,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
    y_min: float = -100.0,
    y_max: float = 0.0,
    y_tick_step: float = 50.0,
    figsize: tuple[float, float] = (12, 12),
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> plt.Figure | None:
    """
    Visualize membrane voltage traces with spike markers.

    Args:
        voltages (NDArray[np.float32]): Voltage array with shape (batch, time, neurons).
        spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        neuron_types (NDArray[np.int32]): Array indicating neuron type indices (0, 1, 2, ...).
        delta_t (float): Time step in milliseconds.
        neuron_params (dict): Dictionary mapping cell type indices to parameters
            {'threshold': float, 'rest': float, 'name': str, 'sign': int}.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        y_min (float): Minimum y-axis value in mV. Defaults to -100.0.
        y_max (float): Maximum y-axis value in mV. Defaults to 0.0.
        y_tick_step (float): Step size for y-axis ticks. Defaults to 50.0.
        figsize (tuple[float, float]): Figure size. Defaults to (12, 12).
        ax (plt.Axes | list[plt.Axes] | None): Matplotlib axes to plot on.
            Can be a single axis, list of axes (one per neuron), or None to create new figure.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """

    n_steps = voltages.shape[1]
    n_steps_plot = int(n_steps * fraction)

    # Handle axes parameter
    if ax is None:
        fig, axes = plt.subplots(n_neurons_plot, 1, figsize=figsize, sharex=True)
        # Ensure axes is always iterable
        if n_neurons_plot == 1:
            axes = [axes]
        return_fig = True
    elif isinstance(ax, list):
        # List of axes provided
        if len(ax) != n_neurons_plot:
            raise ValueError(f"Expected {n_neurons_plot} axes, got {len(ax)}")
        axes = ax
        fig = axes[0].get_figure()
        return_fig = False
    else:
        # Single axis provided - create our own figure
        fig, axes = plt.subplots(n_neurons_plot, 1, figsize=figsize, sharex=True)
        # Ensure axes is always iterable
        if n_neurons_plot == 1:
            axes = [axes]
        return_fig = False

    # Time axis for the last n_steps_plot timesteps (aligned to end of simulation)
    time_axis = (
        np.arange(n_steps - n_steps_plot, n_steps) * delta_t * 1e-3
    )  # Convert to seconds

    for neuron_id in range(n_neurons_plot):
        voltage_trace = voltages[0, -n_steps_plot:, neuron_id]
        spike_times_neuron = np.where(spikes[0, -n_steps_plot:, neuron_id])[0]

        # Plot voltage trace
        axes[neuron_id].plot(
            time_axis, voltage_trace, linewidth=0.5, color="black", rasterized=True
        )

        # Get neuron-specific parameters
        cell_type_idx = neuron_types[neuron_id]
        params = neuron_params[cell_type_idx]
        threshold = params["threshold"]
        rest = params["rest"]

        # Add threshold and rest lines
        from connectome_snns.visualization import EXCITATORY_COLOR, INHIBITORY_COLOR

        axes[neuron_id].axhline(
            y=threshold,
            color=EXCITATORY_COLOR,
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
            label="Threshold",
        )
        axes[neuron_id].axhline(
            y=rest,
            color=INHIBITORY_COLOR,
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
            label="Rest",
        )

        # Mark spike times with vertical lines from threshold to zero
        if len(spike_times_neuron) > 0:
            # Convert spike indices to absolute time (aligned to end of simulation)
            spike_times_s = (
                (n_steps - n_steps_plot + spike_times_neuron) * delta_t * 1e-3
            )
            # Vectorized plotting of spike markers
            spike_x = np.repeat(spike_times_s, 2)
            spike_y = np.tile([threshold, 0], len(spike_times_s))
            axes[neuron_id].plot(
                spike_x.reshape(-1, 2).T,
                spike_y.reshape(-1, 2).T,
                color="black",
                linewidth=0.5,
                alpha=0.7,
                zorder=5,
                rasterized=True,
            )

        # Set ylabel
        ylabel = "Membrane Potential (mV)"
        axes[neuron_id].set_ylabel(ylabel)
        axes[neuron_id].tick_params()
        axes[neuron_id].set_ylim(y_min, y_max)
        # Generate y-ticks based on y_tick_step
        y_ticks = np.arange(y_min, y_max + y_tick_step / 2, y_tick_step)
        axes[neuron_id].set_yticks(y_ticks)
        axes[neuron_id].grid(True, alpha=0.3)

        # Add legend to first subplot only, floating just above the top-right corner
        if neuron_id == 0:
            axes[neuron_id].legend(
                loc="lower right",
                bbox_to_anchor=(1.0, 1.0),
                fancybox=True,
                facecolor="white",
                edgecolor="gray",
                framealpha=0.9,
            )

    axes[-1].set_xlabel("Time (s)")

    # Set uniform tight xlim across all subplots with minimal extension for last tick
    start_time_s = (n_steps - n_steps_plot) * delta_t * 1e-3
    end_time_s = n_steps * delta_t * 1e-3
    for ax in axes:
        ax.set_xlim(start_time_s, end_time_s + 0.01)  # Add 0.01s for tick visibility
        ax.margins(x=0)

    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None


def plot_synaptic_currents(
    I_exc: NDArray[np.float32],
    I_inh: NDArray[np.float32],
    I_tot: NDArray[np.float32],
    delta_t: float,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
    show_total: bool = False,
    neuron_types: NDArray[np.int32] | None = None,
    neuron_params: dict | None = None,
    figsize: tuple[float, float] = (12, 12),
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> plt.Figure | None:
    """
    Visualize excitatory and inhibitory synaptic currents.

    Args:
        I_exc (NDArray[np.float32]): Excitatory current array with shape (batch, time, neurons).
        I_inh (NDArray[np.float32]): Inhibitory current array with shape (batch, time, neurons).
        I_tot (NDArray[np.float32]): Total current array with shape (batch, time, neurons).
        delta_t (float): Time step in milliseconds.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        show_total (bool): Whether to show total current trace in grey. Defaults to False.
        neuron_types (NDArray[np.int32] | None): Array indicating neuron type indices. Defaults to None.
        neuron_params (dict | None): Dictionary mapping cell type indices to parameters. Defaults to None.
        figsize (tuple[float, float]): Figure size. Defaults to (12, 12).
        ax (plt.Axes | list[plt.Axes] | None): Matplotlib axes to plot on. If None, creates new figure.
            If list, should contain n_neurons_plot axes.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """

    n_steps = I_exc.shape[1]
    n_steps_plot = int(n_steps * fraction)

    # Automatically compute nice round y-axis limits based on data
    # Collect all current values to compute 98th percentile (excluding top 2% outliers)
    all_currents = []
    for neuron_id in range(n_neurons_plot):
        I_exc_trace = I_exc[0, -n_steps_plot:, neuron_id]
        I_inh_trace = I_inh[0, -n_steps_plot:, neuron_id]
        all_currents.extend(np.abs(I_exc_trace))
        all_currents.extend(np.abs(I_inh_trace))

    # Use 98th percentile instead of max to avoid outliers
    max_current = np.percentile(all_currents, 98)

    y_lim = _round_to_nice_limit(max_current)

    # Handle axes
    if isinstance(ax, list):
        if len(ax) != n_neurons_plot:
            raise ValueError(f"Expected {n_neurons_plot} axes, got {len(ax)}")
        axes = ax
        fig = axes[0].get_figure()
        return_fig = False
    elif ax is None:
        fig, axes = plt.subplots(n_neurons_plot, 1, figsize=figsize, sharex=True)
        # Ensure axes is always iterable
        if n_neurons_plot == 1:
            axes = [axes]
        return_fig = True
    else:
        # Single axis provided - treat as legacy behavior
        fig, axes = plt.subplots(n_neurons_plot, 1, figsize=figsize, sharex=True)
        # Ensure axes is always iterable
        if n_neurons_plot == 1:
            axes = [axes]
        return_fig = False
    # Time axis for the last n_steps_plot timesteps (aligned to end of simulation)
    time_axis = (
        np.arange(n_steps - n_steps_plot, n_steps) * delta_t * 1e-3
    )  # Convert to seconds

    for neuron_id in range(n_neurons_plot):
        # Extract excitatory and inhibitory currents for this neuron
        I_exc_trace = I_exc[0, -n_steps_plot:, neuron_id]
        I_inh_trace = I_inh[0, -n_steps_plot:, neuron_id]
        I_tot_trace = I_tot[0, -n_steps_plot:, neuron_id]

        # Compute mean total current over full simulation (not just plotted portion)
        I_total_full = I_tot[0, :, neuron_id]
        mean_total = I_total_full.mean()

        # Plot total current in grey
        axes[neuron_id].plot(
            time_axis,
            I_tot_trace,
            linewidth=0.8,
            color="gray",
            alpha=0.7,
            label="Total",
            rasterized=True,
        )

        # Plot excitatory and inhibitory currents
        from connectome_snns.visualization import EXCITATORY_COLOR, INHIBITORY_COLOR

        axes[neuron_id].plot(
            time_axis,
            I_exc_trace,
            linewidth=0.8,
            color=EXCITATORY_COLOR,
            alpha=0.7,
            label="Excitatory",
            rasterized=True,
        )
        axes[neuron_id].plot(
            time_axis,
            I_inh_trace,
            linewidth=0.8,
            color=INHIBITORY_COLOR,
            alpha=0.7,
            label="Inhibitory + Leak",
            rasterized=True,
        )

        # Add zero line
        axes[neuron_id].axhline(
            y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.3
        )

        # Add mean total current as text annotation in top left corner
        axes[neuron_id].text(
            0.02,
            0.95,
            f"mean current = {mean_total:.2f} pA",
            transform=axes[neuron_id].transAxes,
            va="top",
            ha="left",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
            ),
        )

        # Set ylabel
        ylabel = "Input Current (pA)"
        axes[neuron_id].set_ylabel(ylabel)
        axes[neuron_id].tick_params()
        axes[neuron_id].set_ylim(-y_lim, y_lim)
        axes[neuron_id].grid(True, alpha=0.3)

        # Add legend to first subplot only, floating just above the top-right corner
        if neuron_id == 0:
            axes[neuron_id].legend(
                loc="lower right",
                bbox_to_anchor=(1.0, 1.0),
                fancybox=True,
                facecolor="white",
                edgecolor="gray",
                framealpha=0.9,
            )

    axes[-1].set_xlabel("Time (s)")

    # Set uniform tight xlim across all subplots with minimal extension for last tick
    start_time_s = (n_steps - n_steps_plot) * delta_t * 1e-3
    end_time_s = n_steps * delta_t * 1e-3
    for ax in axes:
        ax.set_xlim(start_time_s, end_time_s + 0.01)  # Add 0.01s for tick visibility
        ax.margins(x=0)

    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None


def plot_spike_trains(
    spikes: NDArray[np.int32],
    dt: float,
    cell_type_indices: NDArray[np.int32] | None = None,
    cell_type_names: list[str] | None = None,
    cell_type: str | None = None,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
    random_seed: int = 42,
    title: str | None = None,
    ylabel: str = "Neuron ID",
    figsize: tuple[float, float] = (12, 4),
    ax: plt.Axes | None = None,
    n_compared: int | None = None,
    show_spike_histogram: bool = False,
    cell_type_colors: dict[int, str] | None = None,
) -> plt.Figure | None:
    """Plot spike trains with optional cell type coloring.

    This unified function can plot spike trains for any cell type, with special
    handling for known types like "mitral" (black by default) or multiple cell
    types (colored by type).

    Args:
        spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        dt (float): Time step in milliseconds.
        cell_type_indices (NDArray[np.int32] | None): Array of cell type indices for each neuron.
            If None, all neurons are treated as the same type. Defaults to None.
        cell_type_names (list[str] | None): Names of cell types. Required if cell_type_indices
            is provided. Defaults to None.
        cell_type (str | None): Name of single cell type (e.g., "mitral"). If "mitral",
            uses black color by default. Defaults to None.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        random_seed (int): Random seed for shuffling neurons when cell_type_indices
            is provided. Defaults to 42.
        title (str | None): Custom title for the plot. If None, generates default title
            based on cell type. Defaults to None.
        ylabel (str): Label for y-axis. Defaults to "Neuron ID".
        figsize (tuple[float, float]): Figure size. Defaults to (12, 4).
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
        n_compared (int | None): Number of spike trains being compared (e.g., 2 for
            teacher vs student). When set, neurons are grouped by index with gaps
            between groups and shared shading within each group. Defaults to None.
        show_spike_histogram (bool): If True, show a horizontal histogram of spike
            counts on the right side of the plot. Defaults to False.
        cell_type_colors (dict[int, str] | None): Optional mapping from cell type index
            to color string. Overrides default colors when provided. Use -1 for feedforward.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    if ax is None:
        if show_spike_histogram:
            fig, (ax_to_use, ax_hist) = plt.subplots(
                1, 2, figsize=figsize, width_ratios=[4, 1], sharey=True
            )
        else:
            fig, ax_to_use = plt.subplots(figsize=figsize)
            ax_hist = None
        return_fig = True
    else:
        fig = ax.get_figure()
        ax_to_use = ax
        ax_hist = None
        return_fig = False

    # Calculate number of timesteps to plot (common to both branches)
    n_steps = spikes.shape[1]
    n_steps_plot = int(n_steps * fraction)

    # First, determine final n_neurons_plot and prepare spike data
    if cell_type_indices is not None and cell_type_names is not None:
        n_cell_types = len(cell_type_names)

        # Subtract 1 from n_cell_types if first name is "Feedforward" (already accounted for in colors_map)
        recurrent_start_idx = (
            1 if (n_cell_types > 0 and cell_type_names[0] == "Feedforward") else 0
        )

        if cell_type_colors is not None:
            colors_map = cell_type_colors
        else:
            # Define colors: gray for feedforward, red exc, blue inh, then tab10 overflow
            colors_map = {-1: "#808080"}  # Gray for feedforward neurons (index -1)
            from connectome_snns.visualization import EXCITATORY_COLOR, INHIBITORY_COLOR

            base_colors = [EXCITATORY_COLOR, INHIBITORY_COLOR]

            n_recurrent_types = n_cell_types - recurrent_start_idx

            if n_recurrent_types <= len(base_colors):
                recurrent_colors = base_colors[:n_recurrent_types]
            else:
                cmap = plt.cm.get_cmap("tab10")
                additional_colors = [
                    cmap(i) for i in range(n_recurrent_types - len(base_colors))
                ]
                recurrent_colors = base_colors + additional_colors

            # Map recurrent cell type indices to colors (0, 1, 2, ... -> colors)
            for i in range(n_recurrent_types):
                colors_map[i] = recurrent_colors[i]

        # Shuffle or select neuron indices
        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
            total_neurons = spikes.shape[2]
            n_neurons_plot = min(n_neurons_plot, total_neurons)
            shuffled_indices = rng.permutation(total_neurons)[:n_neurons_plot]
        else:
            total_neurons = spikes.shape[2]
            n_neurons_plot = min(n_neurons_plot, total_neurons)
            shuffled_indices = np.arange(n_neurons_plot)

        # Extract subset of spikes for selected neurons (last n_steps_plot timesteps)
        spikes_subset = np.take(spikes[0, -n_steps_plot:, :], shuffled_indices, axis=1)
        cell_types_subset = cell_type_indices[shuffled_indices]

        # Build spike time lists and colors per neuron
        spike_times_per_neuron = []
        colors_per_neuron = []
        for neuron_idx in range(n_neurons_plot):
            spike_indices = np.where(spikes_subset[:, neuron_idx])[0]
            spike_times_abs = (spike_indices + (n_steps - n_steps_plot)) * dt * 1e-3
            spike_times_per_neuron.append(spike_times_abs)
            colors_per_neuron.append(colors_map[cell_types_subset[neuron_idx]])

        use_cell_type_colors = True
    else:
        # Single cell type case
        spikes_subset = spikes[0, -n_steps_plot:, :n_neurons_plot]
        color = "black"

        spike_times_per_neuron = []
        for neuron_idx in range(n_neurons_plot):
            spike_indices = np.where(spikes_subset[:, neuron_idx])[0]
            spike_times_abs = (spike_indices + (n_steps - n_steps_plot)) * dt * 1e-3
            spike_times_per_neuron.append(spike_times_abs)

        use_cell_type_colors = False

    # Now calculate y-positions and shading with final n_neurons_plot
    gap_size = 0.3  # Gap between groups when comparing
    if n_compared is not None:
        # Comparison mode: group neurons by index with gaps between groups
        n_groups = n_neurons_plot // n_compared
        y_positions = []
        for group_idx in range(n_groups):
            group_start = group_idx * (n_compared + gap_size)
            for within_group in range(n_compared):
                y_positions.append(group_start + within_group)
        y_positions = np.array(y_positions[:n_neurons_plot])

        # Shading: alternate by group (neuron index), not by individual neuron
        # Extend shading symmetrically to include half the gap on each side
        for group_idx in range(0, n_groups, 2):
            group_start = group_idx * (n_compared + gap_size)
            ax_to_use.axhspan(
                group_start - 0.5 - gap_size / 2,
                group_start + n_compared - 0.5 + gap_size / 2,
                color="#f5f5f5",
                zorder=0,
            )
    else:
        # Standard mode: simple integer positions
        y_positions = np.arange(n_neurons_plot)

        # Add alternating row shading for visual separation
        for i in range(0, n_neurons_plot, 2):
            ax_to_use.axhspan(i - 0.5, i + 0.5, color="#f5f5f5", zorder=0)

    # Plot the spike trains
    if use_cell_type_colors:
        ax_to_use.eventplot(
            spike_times_per_neuron,
            colors=colors_per_neuron,
            lineoffsets=y_positions,
            linelengths=0.6,
            linewidths=0.8,
            rasterized=True,
        )

        # Create legend with cell type names (reversed so first type is at bottom)
        legend_elements = []
        for i in range(n_cell_types):
            if i == 0 and cell_type_names[0] == "Feedforward":
                color_idx = -1
            else:
                color_idx = i - recurrent_start_idx
            legend_elements.append(
                Patch(
                    facecolor=colors_map[color_idx],
                    label=cell_type_names[i].capitalize(),
                )
            )
        ax_to_use.legend(handles=legend_elements[::-1], loc="upper right")

        ax_to_use.set_yticks([])
        default_title = "Spike Trains (colored by cell type)"
        ylabel = ""
    else:
        ax_to_use.eventplot(
            spike_times_per_neuron,
            colors=color,
            lineoffsets=y_positions,
            linelengths=0.6,
            linewidths=0.8,
            rasterized=True,
        )

        if n_compared is not None:
            ax_to_use.set_yticks([])
        else:
            ax_to_use.set_yticks(range(n_neurons_plot))
        default_title = (
            f"Sample {cell_type.title() if cell_type else ''} Spike Trains".strip()
        )

    ax_to_use.set_xlabel("Time (s)")
    ax_to_use.set_ylabel(ylabel)
    ax_to_use.set_title(title if title is not None else default_title)
    ax_to_use.tick_params()
    ax_to_use.set_ylim(-0.5, y_positions[-1] + 0.5)
    ax_to_use.yaxis.grid(False)  # Remove horizontal grid lines

    # Set tight xlim with minimal extension for last tick
    start_time_s = (n_steps - n_steps_plot) * dt * 1e-3
    end_time_s = n_steps * dt * 1e-3
    ax_to_use.set_xlim(start_time_s, end_time_s + 0.01)  # Add 0.01s for tick visibility
    ax_to_use.margins(x=0)
    ax_to_use.margins(x=0)

    # Plot spike count histogram if requested
    if ax_hist is not None:
        # Compute spike counts per neuron
        spike_counts = np.array([len(times) for times in spike_times_per_neuron])

        # Plot horizontal bars with matching colors
        if use_cell_type_colors:
            ax_hist.barh(
                y_positions,
                spike_counts,
                height=0.6,
                color=colors_per_neuron,
                edgecolor="none",
            )
        else:
            ax_hist.barh(
                y_positions,
                spike_counts,
                height=0.6,
                color=color,
                edgecolor="none",
            )

        ax_hist.set_xlabel("Spike Count")
        ax_hist.tick_params()
        ax_hist.yaxis.grid(False)
        ax_hist.xaxis.grid(True, alpha=0.3)
        ax_hist.set_xlim(0, None)

    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None


def plot_mitral_cell_spikes(
    input_spikes: NDArray[np.int32],
    dt: float,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
) -> plt.Figure:
    """Plot sample mitral cell spike trains.

    This is a convenience wrapper around plot_spike_trains for backward compatibility.

    Args:
        input_spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        dt (float): Time step in milliseconds.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.

    Returns:
        plt.Figure: Matplotlib figure object containing the mitral cell spike trains.
    """
    return plot_spike_trains(
        spikes=input_spikes,
        dt=dt,
        cell_type="mitral",
        n_neurons_plot=n_neurons_plot,
        fraction=fraction,
    )


def plot_dp_network_spikes(
    output_spikes: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float,
    n_neurons_plot: int = 20,
    fraction: float = 1.0,
    random_seed: int = 42,
) -> plt.Figure:
    """Plot sample Dp network spike trains colored by cell type.

    This is a convenience wrapper around plot_spike_trains for backward compatibility.

    Args:
        output_spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step in milliseconds.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 20.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        random_seed (int): Random seed for shuffling neurons. Defaults to 42.

    Returns:
        plt.Figure: Matplotlib figure object containing the spike trains.
    """
    return plot_spike_trains(
        spikes=output_spikes,
        dt=dt,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        n_neurons_plot=n_neurons_plot,
        fraction=fraction,
        random_seed=random_seed,
        figsize=(12, 6),
    )


def plot_synaptic_conductances(
    recurrent_conductances: NDArray[np.float32],
    feedforward_conductances: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_synapse_names: dict[str, list[str]],
    feedforward_synapse_names: dict[str, list[str]],
    dt: float,
    neuron_id: int = 0,
    fraction: float = 1.0,
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> plt.Figure | None:
    """Plot synaptic conductances for a single neuron, grouped into 3 subplots (E, I, Feedforward).

    Plots individual synapse traces (AMPA, NMDA, GABA_A, GABA_B, etc.) grouped by type with unified legend.

    Args:
        recurrent_conductances (NDArray[np.float32]): Recurrent conductances with shape (batch, time, neurons, synapses).
        feedforward_conductances (NDArray[np.float32]): Feedforward conductances with shape (batch, time, neurons, synapses).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.
        recurrent_synapse_names (dict[str, list[str]]): Synapse names for each recurrent cell type.
        feedforward_synapse_names (dict[str, list[str]]): Synapse names for each feedforward cell type.
        dt (float): Time step in milliseconds.
        neuron_id (int): Index of neuron to plot. Defaults to 0.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        ax (plt.Axes | list[plt.Axes] | None): Matplotlib axes to plot on. If None, creates new figure.
            If list, should contain 3 axes (E, I, Feedforward).

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    # Create time array - use the maximum of recurrent and feedforward timesteps
    n_steps_rec = recurrent_conductances.shape[1]
    n_steps_ff = feedforward_conductances.shape[1]
    n_steps = max(n_steps_rec, n_steps_ff)
    n_steps_plot = int(n_steps * fraction)

    # Build synapse lists by category
    # Conductance array is organized by PRESYNAPTIC cell type, not postsynaptic
    # So we iterate through ALL cell types and their synapse types
    exc_synapses = []  # (idx, name)
    inh_synapses = []

    synapse_idx = 0
    for cell_name in cell_type_names:
        synapse_names_list = recurrent_synapse_names[cell_name]
        for syn_name in synapse_names_list:
            if syn_name in EXCITATORY_SYNAPSE_TYPES:
                exc_synapses.append((synapse_idx, syn_name))
            elif syn_name in INHIBITORY_SYNAPSE_TYPES:
                inh_synapses.append((synapse_idx, syn_name))
            synapse_idx += 1

    # Feedforward synapses
    ff_synapses = []
    for input_idx, input_cell_name in enumerate(input_cell_type_names):
        ff_syn_names = feedforward_synapse_names[input_cell_name]
        # For feedforward, we need to track which synapse indices they are
        for ff_syn_idx, ff_syn_name in enumerate(ff_syn_names):
            # Calculate global ff index
            global_ff_idx = (
                sum(
                    len(feedforward_synapse_names[input_cell_type_names[i]])
                    for i in range(input_idx)
                )
                + ff_syn_idx
            )
            ff_synapses.append((global_ff_idx, f"{input_cell_name} {ff_syn_name}"))

    # Calculate separate slicing for recurrent and feedforward to handle different lengths
    # We want to plot the last n_steps_plot from the total n_steps timeline
    rec_start_idx = max(0, n_steps_rec - n_steps_plot)
    ff_start_idx = max(0, n_steps_ff - n_steps_plot)

    # Create time axes for each - they should align to the same absolute time
    time_axis_rec = (
        np.arange(n_steps - (n_steps_rec - rec_start_idx), n_steps) * dt * 1e-3
    )
    time_axis_ff = np.arange(n_steps - (n_steps_ff - ff_start_idx), n_steps) * dt * 1e-3

    # Collect conductance traces separately: exc+FF share a y-axis, inh is free
    exc_ff_conductances = []
    for syn_idx, _ in exc_synapses:
        exc_ff_conductances.extend(
            recurrent_conductances[0, rec_start_idx:, neuron_id, syn_idx].flatten()
        )
    for syn_idx, _ in ff_synapses:
        exc_ff_conductances.extend(
            feedforward_conductances[0, ff_start_idx:, neuron_id, syn_idx].flatten()
        )

    inh_conductances = []
    for syn_idx, _ in inh_synapses:
        inh_conductances.extend(
            recurrent_conductances[0, rec_start_idx:, neuron_id, syn_idx].flatten()
        )

    max_exc_ff = np.percentile(exc_ff_conductances, 98) if exc_ff_conductances else 1.0
    y_lim_exc_ff = _round_to_nice_limit(max_exc_ff)
    y_lim_inh = y_lim_exc_ff * 10

    # Handle axes - expect 3 axes for E, I, FF
    if isinstance(ax, list):
        if len(ax) != 3:
            raise ValueError(f"Expected 3 axes (E, I, FF), got {len(ax)}")
        axes = ax
        fig = axes[0].get_figure()
        return_fig = False
    elif ax is None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
        return_fig = True
    else:
        # Single axis provided - create 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
        return_fig = False

    # Create unified color mapping for synapse types (not cell types)
    # Collect all unique synapse types present
    all_synapse_types = set()
    for _, syn_name in exc_synapses:
        all_synapse_types.add(syn_name)
    for _, syn_name in inh_synapses:
        all_synapse_types.add(syn_name)
    # For feedforward, extract just the synapse type (not cell type prefix)
    for _, syn_label in ff_synapses:
        syn_type = syn_label.split()[-1]  # Get last word (AMPA, NMDA, etc.)
        all_synapse_types.add(syn_type)

    # Assign consistent colors using canonical synapse palette; fall back to Set1 for unknowns
    from connectome_snns.visualization.colors import SYNAPSE_COLORS

    _fallback_cmap = plt.colormaps["Set1"]
    _unknown_types = sorted(t for t in all_synapse_types if t not in SYNAPSE_COLORS)
    synapse_color_map = {**SYNAPSE_COLORS}
    for i, syn_type in enumerate(_unknown_types):
        synapse_color_map[syn_type] = _fallback_cmap(i % 9)

    # Track which synapse types we've added to legend (for unified legend)
    legend_handles = {}

    # Plot excitatory conductances
    for syn_idx, syn_name in exc_synapses:
        g_trace = recurrent_conductances[0, rec_start_idx:, neuron_id, syn_idx]
        lines = axes[0].plot(
            time_axis_rec,
            g_trace,
            color=synapse_color_map[syn_name],
            linewidth=1,
            alpha=0.8,
            rasterized=True,
        )
        if syn_name not in legend_handles and len(lines) > 0:
            legend_handles[syn_name] = lines[0]
    axes[0].set_ylim(0, y_lim_exc_ff)
    axes[0].tick_params()
    # Add label in top left
    axes[0].text(
        0.02,
        0.98,
        "Excitatory",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
    )

    # Plot inhibitory conductances
    for syn_idx, syn_name in inh_synapses:
        g_trace = recurrent_conductances[0, rec_start_idx:, neuron_id, syn_idx]
        lines = axes[1].plot(
            time_axis_rec,
            g_trace,
            color=synapse_color_map[syn_name],
            linewidth=1,
            alpha=0.8,
            rasterized=True,
        )
        if syn_name not in legend_handles and len(lines) > 0:
            legend_handles[syn_name] = lines[0]
    axes[1].set_ylabel("Conductance (µS)")
    axes[1].set_ylim(0, y_lim_inh)
    axes[1].tick_params()
    # Add label in top left
    axes[1].text(
        0.02,
        0.98,
        "Inhibitory",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
    )

    # Plot feedforward conductances
    for syn_idx, syn_label in ff_synapses:
        syn_type = syn_label.split()[-1]  # Extract synapse type
        g_trace = feedforward_conductances[0, ff_start_idx:, neuron_id, syn_idx]
        lines = axes[2].plot(
            time_axis_ff,
            g_trace,
            color=synapse_color_map[syn_type],
            linewidth=1,
            alpha=0.8,
            rasterized=True,
        )
        if syn_type not in legend_handles and len(lines) > 0:
            legend_handles[syn_type] = lines[0]
    axes[2].set_ylim(0, y_lim_exc_ff)
    axes[2].tick_params()
    # Add label in top left
    axes[2].text(
        0.02,
        0.98,
        "Feedforward",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
    )

    # Add unified legend, floating just above the top-right corner of the top subplot
    if legend_handles:
        axes[0].legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            fancybox=True,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.9,
        )

    # Set common x-axis properties
    axes[-1].set_xlabel("Time (s)")
    # Use the max time from both rec and ff
    max_time = max(
        time_axis_rec[-1] if len(time_axis_rec) > 0 else 0,
        time_axis_ff[-1] if len(time_axis_ff) > 0 else 0,
    )
    min_time = min(
        time_axis_rec[0] if len(time_axis_rec) > 0 else max_time,
        time_axis_ff[0] if len(time_axis_ff) > 0 else max_time,
    )

    # Remove x margins so plots span full time range
    # Hide tick labels for top two subplots
    for i, ax in enumerate(axes):
        ax.set_xlim(min_time, max_time)
        ax.margins(x=0)
        if i < len(axes) - 1:  # Not the last subplot
            ax.set_xticklabels([])

    # Set tight xlim with minimal extension for last tick
    for ax in axes:
        ax.set_xlim(min_time, max_time + 0.01)  # Add 0.01s for tick visibility
        ax.margins(x=0)

    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None

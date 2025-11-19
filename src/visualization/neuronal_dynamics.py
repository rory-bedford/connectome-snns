"""Neuronal dynamics visualization functions."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from numpy.typing import NDArray

from src.parameter_loaders import (
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
        return_fig = False

    # Time axis for the last n_steps_plot timesteps (aligned to end of simulation)
    time_axis = (
        np.arange(n_steps - n_steps_plot, n_steps) * delta_t * 1e-3
    )  # Convert to seconds

    for neuron_id in range(n_neurons_plot):
        voltage_trace = voltages[0, -n_steps_plot:, neuron_id]
        spike_times_neuron = np.where(spikes[0, -n_steps_plot:, neuron_id])[0]

        # Plot voltage trace
        axes[neuron_id].plot(time_axis, voltage_trace, linewidth=0.5, color="black")

        # Get neuron-specific parameters
        cell_type_idx = neuron_types[neuron_id]
        params = neuron_params[cell_type_idx]
        threshold = params["threshold"]
        rest = params["rest"]

        # Add threshold and rest lines
        axes[neuron_id].axhline(
            y=threshold,
            color="#FF0000",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
            label="Threshold",
        )
        axes[neuron_id].axhline(
            y=rest,
            color="#0000FF",
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
            )

        # Set ylabel
        ylabel = "Membrane Potential (mV)"
        axes[neuron_id].set_ylabel(ylabel, fontsize=10)
        axes[neuron_id].tick_params(labelsize=9)
        axes[neuron_id].set_ylim(-80, -20)
        axes[neuron_id].set_yticks([-80, -70, -60, -50, -40, -30, -20])
        axes[neuron_id].grid(True, alpha=0.3)

        # Add legend to first subplot only
        if neuron_id == 0:
            axes[neuron_id].legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)", fontsize=10)

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
        return_fig = True
    else:
        # Single axis provided - treat as legacy behavior
        fig, axes = plt.subplots(n_neurons_plot, 1, figsize=figsize, sharex=True)
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
        )

        # Plot excitatory and inhibitory currents
        axes[neuron_id].plot(
            time_axis,
            I_exc_trace,
            linewidth=0.8,
            color="#FF0000",
            alpha=0.7,
            label="Excitatory",
        )
        axes[neuron_id].plot(
            time_axis,
            I_inh_trace,
            linewidth=0.8,
            color="#0000FF",
            alpha=0.7,
            label="Inhibitory + Leak",
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
            fontsize=8,
            va="top",
            ha="left",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
            ),
        )

        # Set ylabel
        ylabel = "Input Current (pA)"
        axes[neuron_id].set_ylabel(ylabel, fontsize=10)
        axes[neuron_id].tick_params(labelsize=9)
        axes[neuron_id].set_ylim(-y_lim, y_lim)
        axes[neuron_id].grid(True, alpha=0.3)

        # Add legend to first subplot only
        if neuron_id == 0:
            axes[neuron_id].legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)", fontsize=10)

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

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    if ax is None:
        fig, ax_to_use = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        fig = ax.get_figure()
        ax_to_use = ax
        return_fig = False

    # Calculate number of timesteps to plot (common to both branches)
    n_steps = spikes.shape[1]
    n_steps_plot = int(n_steps * fraction)

    # Handle multiple cell types case
    if cell_type_indices is not None and cell_type_names is not None:
        n_cell_types = len(cell_type_names)

        # Define colors: gray for feedforward (-1 index), then red, blue, then tab10 colormap
        colors_map = {-1: "#808080"}  # Gray for feedforward neurons (index -1)
        base_colors = ["#FF0000", "#0000FF"]

        # Subtract 1 from n_cell_types if first name is "Feedforward" (already accounted for in colors_map)
        recurrent_start_idx = (
            1 if (n_cell_types > 0 and cell_type_names[0] == "Feedforward") else 0
        )
        n_recurrent_types = n_cell_types - recurrent_start_idx

        if n_recurrent_types <= 2:
            recurrent_colors = base_colors[:n_recurrent_types]
        else:
            cmap = plt.cm.get_cmap("tab10")
            additional_colors = [cmap(i) for i in range(n_recurrent_types - 2)]
            recurrent_colors = base_colors + additional_colors

        # Map recurrent cell type indices to colors (0, 1, 2, ... -> colors)
        for i in range(n_recurrent_types):
            colors_map[i] = recurrent_colors[i]

        # Shuffle or select neuron indices
        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
            total_neurons = spikes.shape[2]
            # Ensure we don't try to plot more neurons than exist
            n_neurons_plot = min(n_neurons_plot, total_neurons)
            shuffled_indices = rng.permutation(total_neurons)[:n_neurons_plot]
        else:
            # No shuffling - use neurons as provided
            total_neurons = spikes.shape[2]
            n_neurons_plot = min(n_neurons_plot, total_neurons)
            shuffled_indices = np.arange(n_neurons_plot)

        # Extract subset of spikes for selected neurons (last n_steps_plot timesteps)
        # Use explicit indexing with np.take to ensure correct shape
        spikes_subset = np.take(spikes[0, -n_steps_plot:, :], shuffled_indices, axis=1)
        cell_types_subset = cell_type_indices[shuffled_indices]

        # np.where returns (time_indices, neuron_indices) for shape (time, neurons)
        spike_times, neuron_ids = np.where(spikes_subset)

        # Vectorized color mapping using numpy indexing
        spike_colors = [colors_map[cell_types_subset[nid]] for nid in neuron_ids]

        # Convert spike times to absolute time (aligned to end of simulation)
        spike_times_abs = (n_steps - n_steps_plot + spike_times) * dt * 1e-3
        ax_to_use.scatter(spike_times_abs, neuron_ids, s=1, c=spike_colors)

        # Create legend with cell type names
        legend_elements = []
        for i in range(n_cell_types):
            # Map cell type name index to actual color index
            if i == 0 and cell_type_names[0] == "Feedforward":
                color_idx = -1  # Feedforward uses -1
            else:
                color_idx = i - recurrent_start_idx  # Recurrent types
            legend_elements.append(
                Patch(
                    facecolor=colors_map[color_idx],
                    label=cell_type_names[i].capitalize(),
                )
            )
        ax_to_use.legend(handles=legend_elements, loc="upper right", fontsize=9)

        ax_to_use.set_yticks([])  # Remove y-axis tick labels
        default_title = "Spike Trains (colored by cell type)"
        ylabel = ""
    else:
        # Single cell type case - show last n_steps_plot timesteps
        spike_times, neuron_ids = np.where(spikes[0, -n_steps_plot:, :n_neurons_plot])

        # Determine color based on cell_type
        if cell_type is not None and cell_type.lower() == "mitral":
            color = "black"
        else:
            # For unknown cell types, use a default color
            color = "black"

        # Convert spike times to absolute time (aligned to end of simulation)
        spike_times_abs = (n_steps - n_steps_plot + spike_times) * dt * 1e-3
        ax_to_use.scatter(spike_times_abs, neuron_ids, s=1, color=color)
        ax_to_use.set_yticks(range(n_neurons_plot))
        default_title = (
            f"Sample {cell_type.title() if cell_type else ''} Spike Trains".strip()
        )

    ax_to_use.set_xlabel("Time (s)", fontsize=10)
    ax_to_use.set_ylabel(ylabel, fontsize=10)
    ax_to_use.set_title(title if title is not None else default_title, fontsize=11)
    ax_to_use.tick_params(labelsize=9)
    ax_to_use.set_ylim(-0.5, n_neurons_plot - 0.5)

    # Set tight xlim with minimal extension for last tick
    start_time_s = (n_steps - n_steps_plot) * dt * 1e-3
    end_time_s = n_steps * dt * 1e-3
    ax_to_use.set_xlim(start_time_s, end_time_s + 0.01)  # Add 0.01s for tick visibility
    ax_to_use.margins(x=0)
    ax_to_use.margins(x=0)
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

    # Collect all conductance traces to compute y-axis limit
    all_conductances = []
    for syn_idx, _ in exc_synapses:
        g_trace = recurrent_conductances[0, rec_start_idx:, neuron_id, syn_idx]
        all_conductances.extend(g_trace.flatten())
    for syn_idx, _ in inh_synapses:
        g_trace = recurrent_conductances[0, rec_start_idx:, neuron_id, syn_idx]
        all_conductances.extend(g_trace.flatten())
    for syn_idx, _ in ff_synapses:
        g_trace = feedforward_conductances[0, ff_start_idx:, neuron_id, syn_idx]
        all_conductances.extend(g_trace.flatten())

    # Compute y-axis limit
    max_g = np.percentile(all_conductances, 98) if len(all_conductances) > 0 else 1.0
    if max_g <= 0:
        y_lim = 1.0
    else:
        magnitude = 10 ** np.floor(np.log10(max_g))
        normalized = max_g / magnitude
        if normalized <= 1:
            nice_normalized = 1
        elif normalized <= 2:
            nice_normalized = 2
        elif normalized <= 5:
            nice_normalized = 5
        else:
            nice_normalized = 10
        y_lim = nice_normalized * magnitude

    # Handle axes - expect 3 axes for E, I, FF
    if isinstance(ax, list):
        if len(ax) != 3:
            raise ValueError(f"Expected 3 axes (E, I, FF), got {len(ax)}")
        axes = ax
        fig = axes[0].get_figure()
        return_fig = False
    elif ax is None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True, sharey=True)
        return_fig = True
    else:
        # Single axis provided - create 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True, sharey=True)
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

    # Assign consistent colors to synapse types using a more discriminable palette
    synapse_types_list = sorted(all_synapse_types)
    # Use Set1 colormap for better discrimination
    cmap = plt.colormaps["Set1"]
    synapse_color_map = {
        syn_type: cmap(i % 9) for i, syn_type in enumerate(synapse_types_list)
    }

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
        )
        if syn_name not in legend_handles and len(lines) > 0:
            legend_handles[syn_name] = lines[0]
    axes[0].set_ylim(0, y_lim)
    axes[0].tick_params(labelsize=9)
    # Add label in top left
    axes[0].text(
        0.02,
        0.98,
        "Excitatory",
        transform=axes[0].transAxes,
        fontsize=9,
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
        )
        if syn_name not in legend_handles and len(lines) > 0:
            legend_handles[syn_name] = lines[0]
    axes[1].set_ylabel("Conductance (µS)", fontsize=10)
    axes[1].set_ylim(0, y_lim)
    axes[1].tick_params(labelsize=9)
    # Add label in top left
    axes[1].text(
        0.02,
        0.98,
        "Inhibitory",
        transform=axes[1].transAxes,
        fontsize=9,
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
        )
        if syn_type not in legend_handles and len(lines) > 0:
            legend_handles[syn_type] = lines[0]
    axes[2].set_ylim(0, y_lim)
    axes[2].tick_params(labelsize=9)
    # Add label in top left
    axes[2].text(
        0.02,
        0.98,
        "Feedforward",
        transform=axes[2].transAxes,
        fontsize=9,
        va="top",
        ha="left",
    )

    # Add unified legend on top subplot (top right)
    if legend_handles:
        axes[0].legend(
            legend_handles.values(),
            legend_handles.keys(),
            fontsize=9,
            loc="upper right",
        )

    # Set common x-axis properties
    axes[-1].set_xlabel("Time (s)", fontsize=10)
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

"""Neuronal dynamics visualization functions."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from numpy.typing import NDArray
import torch
from typing import Union


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
    voltages: Union[NDArray[np.float32], torch.Tensor],
    spikes: Union[NDArray[np.int32], torch.Tensor],
    neuron_types: Union[NDArray[np.int32], torch.Tensor],
    delta_t: float,
    duration: float,
    neuron_params: dict,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
    y_min: float = -100.0,
    y_max: float = 0.0,
    y_tick_step: float = 50.0,
    figsize: tuple[float, float] = (12, 12),
) -> plt.Figure:
    """
    Visualize membrane voltage traces with spike markers.

    Args:
        voltages (Union[NDArray[np.float32], torch.Tensor]): Voltage array with shape (batch, time, neurons).
        spikes (Union[NDArray[np.int32], torch.Tensor]): Spike array with shape (batch, time, neurons).
        neuron_types (Union[NDArray[np.int32], torch.Tensor]): Array indicating neuron type indices (0, 1, 2, ...).
        delta_t (float): Time step in milliseconds.
        duration (float): Total duration in milliseconds.
        neuron_params (dict): Dictionary mapping cell type indices to parameters
            {'threshold': float, 'rest': float, 'name': str, 'sign': int}.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        y_min (float): Minimum y-axis value in mV. Defaults to -100.0.
        y_max (float): Maximum y-axis value in mV. Defaults to 0.0.
        y_tick_step (float): Step size for y-axis ticks. Defaults to 50.0.
        figsize (tuple[float, float]): Figure size. Defaults to (12, 12).

    Returns:
        plt.Figure: Matplotlib figure object containing the voltage traces.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(voltages, torch.Tensor):
        voltages = voltages.detach().cpu().numpy()
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()
    if isinstance(neuron_types, torch.Tensor):
        neuron_types = neuron_types.detach().cpu().numpy()

    n_steps = voltages.shape[1]
    n_steps_plot = int(n_steps * fraction)

    y_ticks = np.arange(y_min, y_max + 1, y_tick_step)

    fig, axes = plt.subplots(n_neurons_plot, 1, figsize=figsize, sharex=True)
    time_axis = np.arange(n_steps_plot) * delta_t * 1e-3  # Convert to seconds

    for neuron_id in range(n_neurons_plot):
        voltage_trace = voltages[0, :n_steps_plot, neuron_id]
        spike_times_neuron = np.where(spikes[0, :n_steps_plot, neuron_id])[0]

        # Plot voltage trace
        axes[neuron_id].plot(time_axis, voltage_trace, linewidth=0.5, color="black")

        # Get neuron-specific parameters
        cell_type_idx = neuron_types[neuron_id]
        params = neuron_params[cell_type_idx]
        threshold = params["threshold"]
        rest = params["rest"]
        cell_name = params["name"]

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
            spike_times_s = spike_times_neuron * delta_t * 1e-3
            for spike_t in spike_times_s:
                axes[neuron_id].plot(
                    [spike_t, spike_t],
                    [threshold, 0],
                    color="black",
                    linewidth=0.5,
                    alpha=0.7,
                    zorder=5,
                )

        # Create ylabel with cell type info
        ylabel = f"Neuron {neuron_id} ({cell_name})\nVoltage (mV)"
        axes[neuron_id].set_ylabel(ylabel, fontsize=9)
        axes[neuron_id].set_xlim(0, duration * 1e-3 * fraction)
        axes[neuron_id].set_ylim(y_min, y_max)
        axes[neuron_id].set_yticks(y_ticks)
        axes[neuron_id].grid(True, alpha=0.3)

        # Add legend to first subplot only
        if neuron_id == 0:
            axes[neuron_id].legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Membrane Potential Traces (First {n_neurons_plot} Neurons)", fontsize=12
    )
    plt.tight_layout()

    return fig


def plot_synaptic_currents(
    I_exc: Union[NDArray[np.float32], torch.Tensor],
    I_inh: Union[NDArray[np.float32], torch.Tensor],
    delta_t: float,
    duration: float,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
    show_total: bool = False,
    neuron_types: Union[NDArray[np.int32], torch.Tensor, None] = None,
    neuron_params: dict | None = None,
    figsize: tuple[float, float] = (12, 12),
) -> plt.Figure:
    """
    Visualize excitatory and inhibitory synaptic currents.

    Args:
        I_exc (Union[NDArray[np.float32], torch.Tensor]): Excitatory current array with shape (batch, time, neurons).
        I_inh (Union[NDArray[np.float32], torch.Tensor]): Inhibitory current array with shape (batch, time, neurons).
        delta_t (float): Time step in milliseconds.
        duration (float): Total duration in milliseconds.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        show_total (bool): Whether to show total current trace in grey. Defaults to False.
        neuron_types (Union[NDArray[np.int32], torch.Tensor, None]): Array indicating neuron type indices. Defaults to None.
        neuron_params (dict | None): Dictionary mapping cell type indices to parameters. Defaults to None.
        figsize (tuple[float, float]): Figure size. Defaults to (12, 12).

    Returns:
        plt.Figure: Matplotlib figure object containing the current traces.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(I_exc, torch.Tensor):
        I_exc = I_exc.detach().cpu().numpy()
    if isinstance(I_inh, torch.Tensor):
        I_inh = I_inh.detach().cpu().numpy()
    if isinstance(neuron_types, torch.Tensor):
        neuron_types = neuron_types.detach().cpu().numpy()

    n_steps = I_exc.shape[1]
    n_steps_plot = int(n_steps * fraction)

    # Automatically compute nice round y-axis limits based on data
    # Collect all current values to compute 98th percentile (excluding top 2% outliers)
    all_currents = []
    for neuron_id in range(n_neurons_plot):
        I_exc_trace = I_exc[0, :n_steps_plot, neuron_id]
        I_inh_trace = I_inh[0, :n_steps_plot, neuron_id]
        all_currents.extend(np.abs(I_exc_trace))
        all_currents.extend(np.abs(I_inh_trace))

    # Use 98th percentile instead of max to avoid outliers
    max_current = np.percentile(all_currents, 98)

    y_lim = _round_to_nice_limit(max_current)

    fig, axes = plt.subplots(n_neurons_plot, 1, figsize=figsize, sharex=True)
    time_axis = np.arange(n_steps_plot) * delta_t * 1e-3  # Convert to seconds

    for neuron_id in range(n_neurons_plot):
        # Extract excitatory and inhibitory currents for this neuron
        I_exc_trace = I_exc[0, :n_steps_plot, neuron_id]
        I_inh_trace = I_inh[0, :n_steps_plot, neuron_id]
        I_total_trace = I_exc_trace + I_inh_trace

        # Compute mean total current over full simulation (not just plotted portion)
        I_total_full = I_exc[0, :, neuron_id] + I_inh[0, :, neuron_id]
        mean_total = I_total_full.mean()

        # Plot total current in grey (optional)
        if show_total:
            axes[neuron_id].plot(
                time_axis,
                I_total_trace,
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
            label="Inhibitory",
        )

        # Add zero line
        axes[neuron_id].axhline(
            y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.3
        )

        # Add horizontal line at mean total current
        axes[neuron_id].axhline(
            y=mean_total, color="black", linestyle="--", linewidth=0.8, alpha=0.5
        )

        # Add mean total current as text annotation on the right
        axes[neuron_id].text(
            1.04,
            0.5,
            f"mean current = {mean_total:.2f} pA",
            transform=axes[neuron_id].transAxes,
            fontsize=7,
            va="center",
            ha="left",
            color="black",
        )

        # Add ylabel with cell type info
        cell_type_idx = neuron_types[neuron_id] if neuron_types is not None else None
        if cell_type_idx is not None and neuron_params is not None:
            cell_name = neuron_params[cell_type_idx]["name"]
            ylabel = f"Neuron {neuron_id} ({cell_name})\nCurrent (pA)"
        else:
            ylabel = f"Neuron {neuron_id}\nCurrent (pA)"

        axes[neuron_id].set_ylabel(ylabel, fontsize=9)
        axes[neuron_id].set_xlim(0, duration * 1e-3 * fraction)
        axes[neuron_id].set_ylim(-y_lim, y_lim)
        axes[neuron_id].grid(True, alpha=0.3)

        # Add legend to first subplot only
        if neuron_id == 0:
            axes[neuron_id].legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Synaptic Current Inputs (First {n_neurons_plot} Neurons)", fontsize=12
    )
    plt.tight_layout()

    return fig


def plot_spike_trains(
    spikes: NDArray[np.int32],
    dt: float,
    duration: float,
    cell_type_indices: NDArray[np.int32] | None = None,
    cell_type_names: list[str] | None = None,
    cell_type: str | None = None,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
    random_seed: int = 42,
    title: str | None = None,
    ylabel: str = "Neuron ID",
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    """Plot spike trains with optional cell type coloring.

    This unified function can plot spike trains for any cell type, with special
    handling for known types like "mitral" (black by default) or multiple cell
    types (colored by type).

    Args:
        spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        dt (float): Time step in milliseconds.
        duration (float): Total duration in milliseconds.
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

    Returns:
        plt.Figure: Matplotlib figure object containing the spike trains.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Handle multiple cell types case
    if cell_type_indices is not None and cell_type_names is not None:
        n_cell_types = len(cell_type_names)

        # Define colors: red, blue, then tab10 colormap for additional types
        base_colors = ["#FF0000", "#0000FF"]
        if n_cell_types <= 2:
            colors_map = base_colors[:n_cell_types]
        else:
            cmap = plt.cm.get_cmap("tab10")
            additional_colors = [cmap(i) for i in range(n_cell_types - 2)]
            colors_map = base_colors + additional_colors

        # Shuffle neuron indices
        rng = np.random.RandomState(random_seed)
        total_neurons = spikes.shape[2]
        shuffled_indices = rng.permutation(total_neurons)[:n_neurons_plot]

        # Extract subset of spikes for selected neurons
        spikes_subset = spikes[0][:, shuffled_indices]
        cell_types_subset = cell_type_indices[shuffled_indices]

        spike_times, neuron_ids = np.where(spikes_subset)

        # Color spikes by cell type
        spike_colors = [colors_map[cell_types_subset[nid]] for nid in neuron_ids]

        ax.scatter(spike_times * dt * 1e-3, neuron_ids, s=1, c=spike_colors)

        # Create legend with cell type names
        legend_elements = [
            Patch(facecolor=colors_map[i], label=cell_type_names[i])
            for i in range(n_cell_types)
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        ax.set_yticks([])  # Remove y-axis tick labels
        default_title = "Spike Trains (colored by cell type)"
        ylabel = "Neuron (shuffled)"
    else:
        # Single cell type case
        spike_times, neuron_ids = np.where(spikes[0, :, :n_neurons_plot])

        # Determine color based on cell_type
        if cell_type is not None and cell_type.lower() == "mitral":
            color = "black"
        else:
            # For unknown cell types, use a default color
            color = "black"

        ax.scatter(spike_times * dt * 1e-3, neuron_ids, s=1, color=color)
        ax.set_yticks(range(n_neurons_plot))
        default_title = (
            f"Sample {cell_type.title() if cell_type else ''} Spike Trains".strip()
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else default_title)
    ax.set_ylim(-0.5, n_neurons_plot - 0.5)
    ax.set_xlim(0, duration * 1e-3 * fraction)
    plt.tight_layout()

    return fig


def plot_mitral_cell_spikes(
    input_spikes: NDArray[np.int32],
    dt: float,
    duration: float,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
) -> plt.Figure:
    """Plot sample mitral cell spike trains.

    This is a convenience wrapper around plot_spike_trains for backward compatibility.

    Args:
        input_spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        dt (float): Time step in milliseconds.
        duration (float): Total duration in milliseconds.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.

    Returns:
        plt.Figure: Matplotlib figure object containing the mitral cell spike trains.
    """
    return plot_spike_trains(
        spikes=input_spikes,
        dt=dt,
        duration=duration,
        cell_type="mitral",
        n_neurons_plot=n_neurons_plot,
        fraction=fraction,
    )


def plot_dp_network_spikes(
    output_spikes: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float,
    duration: float,
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
        duration (float): Total duration in milliseconds.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 20.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        random_seed (int): Random seed for shuffling neurons. Defaults to 42.

    Returns:
        plt.Figure: Matplotlib figure object containing the spike trains.
    """
    return plot_spike_trains(
        spikes=output_spikes,
        dt=dt,
        duration=duration,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        n_neurons_plot=n_neurons_plot,
        fraction=fraction,
        random_seed=random_seed,
        figsize=(12, 6),
    )


def plot_synaptic_conductances(
    output_conductances: NDArray[np.float32],
    input_conductances: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_synapse_names: dict[str, list[str]],
    feedforward_synapse_names: dict[str, list[str]],
    dt: float,
    duration: float,
    neuron_id: int = 0,
    fraction: float = 1.0,
) -> plt.Figure:
    """Plot synaptic conductances for a single neuron, with each synapse type on a separate subplot.

    Shows both recurrent and feedforward conductances in separate subplots, one for each synapse type.

    Args:
        output_conductances (NDArray[np.float32]): Recurrent conductances with shape (batch, time, neurons, synapses).
        input_conductances (NDArray[np.float32]): Feedforward conductances with shape (batch, time, neurons, synapses).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.
        recurrent_synapse_names (dict[str, list[str]]): Synapse names for each recurrent cell type.
        feedforward_synapse_names (dict[str, list[str]]): Synapse names for each feedforward cell type.
        dt (float): Time step in milliseconds.
        duration (float): Total duration in milliseconds.
        neuron_id (int): Index of neuron to plot. Defaults to 0.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.

    Returns:
        plt.Figure: Matplotlib figure object containing the conductance traces.
    """
    # Build list of all synapse types (recurrent + feedforward)
    all_synapse_labels = []

    # Recurrent synapse types
    for cell_type in cell_type_names:
        synapse_names = recurrent_synapse_names[cell_type]
        for syn_name in synapse_names:
            all_synapse_labels.append(f"{cell_type} {syn_name}")

    # Feedforward synapse types
    ff_start_idx = len(all_synapse_labels)
    for cell_type in input_cell_type_names:
        synapse_names = feedforward_synapse_names[cell_type]
        for syn_name in synapse_names:
            all_synapse_labels.append(f"{cell_type} {syn_name}")

    # Create time array
    n_steps = output_conductances.shape[1]
    n_steps_plot = int(n_steps * fraction)
    time_axis = np.arange(n_steps_plot) * dt * 1e-3  # Convert to seconds

    # Color palette for different synapse types
    cmap = plt.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(len(all_synapse_labels))]

    # Get cell type for this neuron
    cell_type_idx = cell_type_indices[neuron_id]
    cell_name = cell_type_names[cell_type_idx]

    # Automatically compute nice round y-axis limit based on data
    # Collect all conductance values to compute 98th percentile (excluding top 2% outliers)
    all_conductances = []
    for syn_idx in range(output_conductances.shape[3]):
        g_trace = output_conductances[0, :n_steps_plot, neuron_id, syn_idx]
        all_conductances.extend(g_trace)
    for syn_idx in range(input_conductances.shape[3]):
        g_trace = input_conductances[0, :n_steps_plot, neuron_id, syn_idx]
        all_conductances.extend(g_trace)

    # Use 98th percentile instead of max to avoid outliers
    max_conductance = np.percentile(all_conductances, 98)

    # Round to nice limit
    if max_conductance <= 0:
        y_lim = 1.0
    else:
        magnitude = 10 ** np.floor(np.log10(max_conductance))
        normalized = max_conductance / magnitude
        if normalized <= 1:
            nice_normalized = 1
        elif normalized <= 2:
            nice_normalized = 2
        elif normalized <= 5:
            nice_normalized = 5
        else:
            nice_normalized = 10
        y_lim = nice_normalized * magnitude

    # Total number of synapse types
    n_synapses = output_conductances.shape[3] + input_conductances.shape[3]

    # Create figure with subplots for each synapse type
    fig, axes = plt.subplots(
        n_synapses, 1, figsize=(14, 2 * n_synapses), sharex=True, sharey=True
    )
    if n_synapses == 1:
        axes = [axes]

    # Plot recurrent conductances
    for syn_idx in range(output_conductances.shape[3]):
        ax = axes[syn_idx]
        g_trace = output_conductances[0, :n_steps_plot, neuron_id, syn_idx]
        ax.plot(
            time_axis,
            g_trace,
            linewidth=1.0,
            color=colors[syn_idx],
            alpha=0.8,
        )
        ax.set_ylabel("Conductance (nS)", fontsize=9)
        ax.set_ylim(0, y_lim)
        ax.grid(True, alpha=0.3)
        ax.set_title(all_synapse_labels[syn_idx], fontsize=10, loc="left")

    # Plot feedforward conductances
    for syn_idx in range(input_conductances.shape[3]):
        ax = axes[output_conductances.shape[3] + syn_idx]
        g_trace = input_conductances[0, :n_steps_plot, neuron_id, syn_idx]
        ax.plot(
            time_axis,
            g_trace,
            linewidth=1.0,
            color=colors[ff_start_idx + syn_idx],
            alpha=0.8,
        )
        ax.set_ylabel("Conductance (nS)", fontsize=9)
        ax.set_ylim(0, y_lim)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            all_synapse_labels[ff_start_idx + syn_idx], fontsize=10, loc="left"
        )

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(0, duration * 1e-3 * fraction)
    fig.suptitle(
        f"Synaptic Conductances - Neuron {neuron_id} ({cell_name})", fontsize=14
    )
    plt.tight_layout()

    return fig

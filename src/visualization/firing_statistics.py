"""Visualization functions for firing statistics and spike train analysis."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from analysis.firing_statistics import (
    compute_spike_train_cv,
    compute_spike_train_fano_factor,
)


def plot_fano_factor_vs_window_size(
    spike_trains: NDArray[np.int32],
    window_sizes: list[int] | np.ndarray,
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float = 1.0,
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> Figure:
    """
    Plot mean Fano factor across neurons as a function of window size, split by cell type.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        window_sizes (list[int] | np.ndarray): List of window sizes (in steps) to evaluate.
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step duration in milliseconds for converting steps to time.
            Defaults to 1.0.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    # Convert window sizes from steps to seconds
    window_sizes_s = np.array(window_sizes) * dt * 1e-3  # Convert ms to s

    n_cell_types = len(cell_type_names)

    # Define colors: red, blue, then tab10 colormap for additional types
    base_colors = ["#FF0000", "#0000FF"]
    if n_cell_types <= 2:
        colors_map = base_colors[:n_cell_types]
    else:
        cmap = plt.cm.get_cmap("tab10")
        additional_colors = [cmap(i) for i in range(n_cell_types - 2)]
        colors_map = base_colors + additional_colors

    # Handle axes parameter
    if ax is None:
        fig, axes = plt.subplots(
            1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
        )
        return_fig = True
        if n_cell_types == 1:
            axes = [axes]
    elif isinstance(ax, list):
        # List of axes provided
        if len(ax) != n_cell_types:
            raise ValueError(f"Expected {n_cell_types} axes, got {len(ax)}")
        axes = ax
        fig = axes[0].get_figure()
        return_fig = False
    else:
        # Single axis provided - create our own figure
        fig, axes = plt.subplots(
            1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
        )
        return_fig = False
        if n_cell_types == 1:
            axes = [axes]

    # Plot each cell type
    for cell_type_idx in range(n_cell_types):
        ax = axes[cell_type_idx] if n_cell_types > 1 or ax is None else axes[0]

        mean_fano_factors = []
        std_fano_factors = []

        # Get neurons of this cell type
        cell_type_mask = cell_type_indices == cell_type_idx

        for window_size in window_sizes:
            fano_factors = compute_spike_train_fano_factor(spike_trains, window_size)
            # Extract fano factors for this cell type across all batches
            fano_cell_type = fano_factors[:, cell_type_mask].flatten()
            fano_valid = fano_cell_type[~np.isnan(fano_cell_type)]

            if len(fano_valid) > 0:
                mean_fano_factors.append(float(fano_valid.mean()))
                std_fano_factors.append(float(fano_valid.std()))
            else:
                mean_fano_factors.append(np.nan)
                std_fano_factors.append(np.nan)

        mean_fano_factors = np.array(mean_fano_factors)
        std_fano_factors = np.array(std_fano_factors)

        # Check if we have any valid data to plot
        has_valid_data = np.any(~np.isnan(mean_fano_factors))

        if has_valid_data:
            # Plot mean line with colored dashed line
            ax.plot(
                window_sizes_s,
                mean_fano_factors,
                linestyle="--",
                linewidth=2,
                color=colors_map[cell_type_idx],
                alpha=0.8,
                zorder=2,
            )

            # Plot points and error bars in black
            ax.errorbar(
                window_sizes_s,
                mean_fano_factors,
                yerr=std_fano_factors,
                fmt="o",
                capsize=5,
                markersize=8,
                color="black",
                ecolor="black",
                capthick=1.5,
                elinewidth=1.5,
                alpha=0.6,
                zorder=3,
            )

            # Fill between mean +/- std with transparent color
            ax.fill_between(
                window_sizes_s,
                mean_fano_factors - std_fano_factors,
                mean_fano_factors + std_fano_factors,
                color=colors_map[cell_type_idx],
                alpha=0.2,
                zorder=1,
            )

            ax.set_xlabel("Window Size (s)", fontsize=10)
            ax.set_ylabel("Mean Fano Factor", fontsize=10)
            ax.set_title(f"{cell_type_names[cell_type_idx].capitalize()}", fontsize=11)
            ax.tick_params(labelsize=9)
            # Only set log scale if we have positive window sizes
            if len(window_sizes_s) > 0 and np.all(np.array(window_sizes_s) > 0):
                ax.set_xscale("log")
            ax.axhline(
                1.0, color="black", linestyle="-", linewidth=1.5, alpha=0.7, zorder=1
            )
            ax.grid(True, alpha=0.3, which="both")
        else:
            # No valid data - display message
            ax.text(
                0.5,
                0.5,
                "No spikes detected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.set_xlabel("Window Size (s)", fontsize=10)
            ax.set_ylabel("Mean Fano Factor", fontsize=10)
            ax.set_title(f"{cell_type_names[cell_type_idx].capitalize()}", fontsize=11)
            ax.tick_params(labelsize=9)

    # Ensure shared y-axis range across all subplots
    if n_cell_types > 1:
        all_ylims = [ax.get_ylim() for ax in axes]
        global_ymin = min(ylim[0] for ylim in all_ylims)
        global_ymax = max(ylim[1] for ylim in all_ylims)
        for ax in axes:
            ax.set_ylim(global_ymin, global_ymax)

    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None


def plot_cv_histogram(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float = 1.0,
    bins: int = 50,
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> Figure:
    """
    Plot histogram of CV values across all neurons, split by cell type.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step duration in milliseconds for computing ISIs in seconds.
            Defaults to 1.0.
        bins (int): Number of histogram bins.
        ax (plt.Axes | list[plt.Axes] | None): Matplotlib axes to plot on.
            Can be a single axis, list of axes (one per cell type), or None to create new figure.

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    n_cell_types = len(cell_type_names)

    # Define colors: red, blue, then tab10 colormap for additional types
    base_colors = ["#FF0000", "#0000FF"]
    if n_cell_types <= 2:
        colors_map = base_colors[:n_cell_types]
    else:
        cmap = plt.cm.get_cmap("tab10")
        additional_colors = [cmap(i) for i in range(n_cell_types - 2)]
        colors_map = base_colors + additional_colors

    # Compute CV for all neurons (convert dt from ms to s)
    cv_values = compute_spike_train_cv(spike_trains, dt=dt * 1e-3)

    # Collect all CV values to determine global x-axis range and shared bins
    all_cv_valid = []
    for i in range(n_cell_types):
        cell_type_mask = cell_type_indices == i
        cv_cell_type = cv_values[:, cell_type_mask].flatten()
        cv_valid = cv_cell_type[~np.isnan(cv_cell_type)]
        if len(cv_valid) > 0:
            all_cv_valid.extend(cv_valid)

    # Determine shared x-axis range and bins
    if len(all_cv_valid) > 0:
        x_min = min(all_cv_valid)
        x_max = max(all_cv_valid)
        # Add 5% padding
        x_range = x_max - x_min
        x_min = max(0, x_min - 0.05 * x_range)
        x_max = x_max + 0.05 * x_range
        bin_edges = np.linspace(x_min, x_max, bins + 1)
    else:
        bin_edges = bins  # Fallback to automatic binning if no data
        x_min, x_max = 0, 2

    # Handle axes parameter
    if ax is None:
        fig, axes = plt.subplots(
            1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
        )
        return_fig = True
        if n_cell_types == 1:
            axes = [axes]
    elif isinstance(ax, list):
        # List of axes provided
        if len(ax) != n_cell_types:
            raise ValueError(f"Expected {n_cell_types} axes, got {len(ax)}")
        axes = ax
        fig = axes[0].get_figure()
        return_fig = False
    else:
        # Single axis provided - create our own figure
        fig, axes = plt.subplots(
            1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
        )
        return_fig = False
        if n_cell_types == 1:
            axes = [axes]

    # Plot each cell type
    for i in range(n_cell_types):
        ax_i = axes[i]

        # Get CV values for this cell type across all batches
        cell_type_mask = cell_type_indices == i
        cv_cell_type = cv_values[:, cell_type_mask].flatten()
        cv_valid = cv_cell_type[~np.isnan(cv_cell_type)]

        if len(cv_valid) > 0:
            # Plot histogram with shared bins
            ax_i.hist(
                cv_valid,
                bins=bin_edges,
                alpha=0.6,
                edgecolor="black",
                linewidth=1.2,
                color=colors_map[i],
                label=f"n={len(cv_valid)}",
            )
            ax_i.axvline(
                cv_valid.mean(),
                color="gray",
                linestyle="--",
                linewidth=2,
                alpha=0.6,
                label=f"Mean: {cv_valid.mean():.2f}",
            )
            ax_i.set_xlabel("Coefficient of Variation (CV)", fontsize=10)
            ax_i.set_ylabel("Count", fontsize=10)
            ax_i.set_title(f"{cell_type_names[i].capitalize()}", fontsize=11)
            ax_i.tick_params(labelsize=9)
            ax_i.set_xlim(x_min, x_max)
            ax_i.legend(fontsize=9)
            ax_i.grid(True, alpha=0.3, axis="y")
        else:
            ax_i.text(
                0.5,
                0.5,
                "No valid CV values",
                ha="center",
                va="center",
                transform=ax_i.transAxes,
            )
            ax_i.tick_params(labelsize=9)
            ax_i.set_title(f"{cell_type_names[i].capitalize()}", fontsize=11)
            ax_i.set_xlim(x_min, x_max)

    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None


def plot_isi_histogram(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float = 1.0,
    bins: int = 50,
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> Figure:
    """
    Plot histogram of inter-spike intervals (ISIs) split by cell type.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step duration in milliseconds for converting time steps to seconds.
            Defaults to 1.0.
        bins (int): Number of histogram bins.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    n_cell_types = len(cell_type_names)

    # Define colors: red, blue, then tab10 colormap for additional types
    base_colors = ["#FF0000", "#0000FF"]
    if n_cell_types <= 2:
        colors_map = base_colors[:n_cell_types]
    else:
        cmap = plt.cm.get_cmap("tab10")
        additional_colors = [cmap(i) for i in range(n_cell_types - 2)]
        colors_map = base_colors + additional_colors

    batch_size, n_steps, n_neurons = spike_trains.shape

    # Create time indices in seconds (dt is in milliseconds)
    time_step = dt * 1e-3

    # Vectorized ISI computation - collect ISIs for each cell type
    isis_by_type = [[] for _ in range(n_cell_types)]

    # Process all neurons and batches - vectorized over batches where possible
    for cell_type_idx in range(n_cell_types):
        cell_type_mask = cell_type_indices == cell_type_idx
        neuron_indices = np.where(cell_type_mask)[0]

        for neuron_idx in neuron_indices:
            # Process all batches for this neuron
            for batch_idx in range(batch_size):
                # Find spike indices for this neuron in this batch
                spike_indices = np.where(spike_trains[batch_idx, :, neuron_idx] > 0)[0]

                # Need at least 2 spikes to compute ISIs
                if len(spike_indices) >= 2:
                    # Compute ISIs directly from indices (more efficient)
                    isis = np.diff(spike_indices.astype(np.float32)) * time_step
                    isis_by_type[cell_type_idx].append(isis)

    # Collect all ISIs across cell types to determine global x-axis limit and shared bins
    all_isis_combined = []
    for i in range(n_cell_types):
        if len(isis_by_type[i]) > 0:
            all_isis_array = np.concatenate(isis_by_type[i])
            all_isis_combined.append(all_isis_array)

    # Compute 99.5th percentile for x-axis limit and create shared bin edges
    if len(all_isis_combined) > 0:
        all_isis_flat = np.concatenate(all_isis_combined)
        x_limit = np.percentile(all_isis_flat, 99.5)
        # Create shared bin edges from 0 to x_limit
        bin_edges = np.linspace(0, x_limit, bins + 1)
    else:
        x_limit = None
        bin_edges = bins  # Fallback to automatic binning if no data

    # Handle axes parameter
    if ax is None:
        fig, axes = plt.subplots(
            1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
        )
        return_fig = True
        if n_cell_types == 1:
            axes = [axes]
    elif isinstance(ax, list):
        # List of axes provided
        if len(ax) != n_cell_types:
            raise ValueError(f"Expected {n_cell_types} axes, got {len(ax)}")
        axes = ax
        fig = axes[0].get_figure()
        return_fig = False
    else:
        # Single axis provided - create our own figure
        fig, axes = plt.subplots(
            1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
        )
        return_fig = False
        if n_cell_types == 1:
            axes = [axes]

    # Plot each cell type
    for i in range(n_cell_types):
        ax_i = axes[i]

        if len(isis_by_type[i]) > 0:
            # Concatenate all ISIs for this cell type
            isis_np = np.concatenate(isis_by_type[i])

            # Plot histogram with shared bin edges
            ax_i.hist(
                isis_np,
                bins=bin_edges,
                alpha=0.6,
                edgecolor="black",
                linewidth=1.2,
                color=colors_map[i],
                label=f"n={len(isis_np)}",
            )
            ax_i.axvline(
                isis_np.mean(),
                color="gray",
                linestyle="--",
                linewidth=2,
                alpha=0.6,
                label=f"Mean: {isis_np.mean():.4f} s",
            )
            ax_i.set_xlabel("Inter-Spike Interval (s)", fontsize=10)
            ax_i.set_ylabel("Count", fontsize=10)
            ax_i.set_title(f"{cell_type_names[i].capitalize()}", fontsize=11)
            ax_i.tick_params(labelsize=9)

            # Set x-axis limit to 98th percentile for better resolution
            if x_limit is not None:
                ax_i.set_xlim(0, x_limit)

            ax_i.legend(fontsize=9)
            ax_i.grid(True, alpha=0.3, axis="y")
        else:
            ax_i.text(
                0.5,
                0.5,
                "No ISIs found",
                ha="center",
                va="center",
                transform=ax_i.transAxes,
            )
            ax_i.tick_params(labelsize=9)
            ax_i.set_title(f"{cell_type_names[i].capitalize()}", fontsize=11)

    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None


def plot_psth(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    window_size: float,
    dt: float = 1.0,
    ax: plt.Axes | None = None,
    title: str | None = None,
    input_spike_trains: NDArray[np.int32] | None = None,
) -> Figure:
    """
    Plot Peri-Stimulus Time Histogram (PSTH) split by cell type with overlapping windows.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        window_size (float): Window size for PSTH bins in milliseconds.
        dt (float): Time step duration in milliseconds. Defaults to 1.0.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
        title (str | None): Optional custom title. If None, uses default title.
        input_spike_trains (NDArray[np.int32] | None): Optional feedforward input spikes with shape (batch_size, n_steps, n_inputs).

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    batch_size, n_steps, n_neurons = spike_trains.shape
    n_cell_types = len(cell_type_names)

    # Define colors: red, blue, then tab10 colormap for additional types
    base_colors = ["#FF0000", "#0000FF"]
    if n_cell_types <= 2:
        colors_map = base_colors[:n_cell_types]
    else:
        cmap = plt.cm.get_cmap("tab10")
        additional_colors = [cmap(i) for i in range(n_cell_types - 2)]
        colors_map = base_colors + additional_colors

    # Convert window size to timesteps
    window_steps = int(np.round(window_size / dt))

    # Create time array for bin centers (one bin per timestep)
    time_centers = np.arange(n_steps) * dt * 1e-3  # Convert to seconds

    # Create figure
    if ax is None:
        fig, ax_to_use = plt.subplots(figsize=(12, 6))
        return_fig = True
    else:
        fig = ax.get_figure()
        ax_to_use = ax
        return_fig = False

    # Calculate PSTH for each cell type using vectorized convolution
    for cell_type_idx in range(n_cell_types):
        cell_type_mask = cell_type_indices == cell_type_idx
        n_neurons_type = cell_type_mask.sum()

        if n_neurons_type == 0:
            continue

        # Sum spikes across neurons of this type and all batches
        spike_counts = spike_trains[:, :, cell_type_mask].sum(
            axis=(0, 2)
        )  # Shape: (n_steps,)

        # Use convolution for efficient sliding window operation
        # Create uniform kernel that sums over the window (not averages)
        kernel = np.ones(window_steps)

        # Apply convolution with 'same' mode to keep same length
        # This gives us the total spike count in each overlapping window
        convolved = np.convolve(spike_counts, kernel, mode="same")

        # Convert to firing rate: spikes per window -> Hz
        # Normalize by window duration (in seconds), number of neurons, and number of batches
        firing_rates = convolved / (window_size * 1e-3) / n_neurons_type / batch_size

        # Plot PSTH
        cell_name_capitalized = cell_type_names[cell_type_idx].capitalize()
        ax_to_use.plot(
            time_centers,
            firing_rates,
            color=colors_map[cell_type_idx],
            linewidth=1,
            alpha=0.6,
            label=f"{cell_name_capitalized} (n={n_neurons_type})",
        )

    # Plot feedforward input if provided
    if input_spike_trains is not None:
        n_inputs = input_spike_trains.shape[2]
        spike_counts_ff = input_spike_trains.sum(
            axis=(0, 2)
        )  # Sum across batches and inputs
        kernel = np.ones(window_steps)
        convolved_ff = np.convolve(spike_counts_ff, kernel, mode="same")
        firing_rates_ff = convolved_ff / (window_size * 1e-3) / n_inputs / batch_size
        ax_to_use.plot(
            time_centers,
            firing_rates_ff,
            color="#808080",  # Gray
            linewidth=1,
            alpha=0.6,
            label=f"Feedforward (n={n_inputs})",
        )

    ax_to_use.set_xlabel("Time (s)", fontsize=10)
    ax_to_use.set_ylabel("Firing Rate (Hz)", fontsize=10)
    ax_to_use.tick_params(labelsize=9)

    if title is None:
        title = "Population Activity"
    ax_to_use.set_title(title, fontsize=11)

    # Set xlim to span full time range with no margins
    # Set tight xlim with minimal extension for last tick
    start_time_s = 0
    end_time_s = (n_steps - 1) * dt * 1e-3
    ax_to_use.set_xlim(start_time_s, end_time_s + 0.01)  # Add 0.01s for tick visibility
    ax_to_use.margins(x=0)

    ax_to_use.legend(loc="upper right", fontsize=9)
    ax_to_use.grid(True, alpha=0.3)
    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None


def plot_firing_rate_distribution(
    output_spikes: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float,
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> plt.Figure | None:
    """Plot distribution of firing rates in the Dp network by cell type.

    Args:
        output_spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step in milliseconds.
        ax (plt.Axes | list[plt.Axes] | None): Matplotlib axes to plot on.
            Can be a single axis, list of axes (one per cell type), or None to create new figure.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    n_cell_types = len(cell_type_names)

    # Define colors: red, blue, then tab10 colormap for additional types
    base_colors = ["#FF0000", "#0000FF"]
    if n_cell_types <= 2:
        colors_map = base_colors[:n_cell_types]
    else:
        cmap = plt.cm.get_cmap("tab10")
        additional_colors = [cmap(i) for i in range(n_cell_types - 2)]
        colors_map = base_colors + additional_colors

    # Calculate duration from data shape
    n_timesteps = output_spikes.shape[1]
    duration_s = n_timesteps * dt * 1e-3  # Duration in seconds

    # Calculate firing rates (spikes per second)
    spike_counts = output_spikes[0].sum(axis=0)  # Total spikes per neuron
    firing_rates = spike_counts / duration_s

    # Filter out zero firing rates for log scale
    firing_rates_nonzero = firing_rates[firing_rates > 0]

    # Handle axes parameter
    if ax is None:
        fig, axes = plt.subplots(
            1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
        )
        return_fig = True
        if n_cell_types == 1:
            axes = [axes]
    elif isinstance(ax, list):
        # List of axes provided
        if len(ax) != n_cell_types:
            raise ValueError(f"Expected {n_cell_types} axes, got {len(ax)}")
        axes = ax
        fig = axes[0].get_figure()
        return_fig = False
    else:
        # Single axis provided - create our own figure
        fig, axes = plt.subplots(
            1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
        )
        return_fig = False
        if n_cell_types == 1:
            axes = [axes]

    # Calculate global x-axis limits
    if len(firing_rates_nonzero) > 0:
        global_x_min = firing_rates_nonzero.min()
        global_x_max = firing_rates_nonzero.max()
    else:
        global_x_min = 0.1
        global_x_max = 10.0

    # Create logarithmically-spaced bins
    log_bins = np.logspace(np.log10(global_x_min), np.log10(global_x_max), 30)

    # Plot each cell type
    for i in range(n_cell_types):
        ax = axes[i]
        cell_type_rates = firing_rates[cell_type_indices == i]
        cell_type_rates_nonzero = cell_type_rates[cell_type_rates > 0]
        n_silent = (cell_type_rates == 0).sum()

        # Build legend handles and labels in the correct order
        legend_handles = []
        legend_labels = []

        # 1. Plot histogram for non-zero rates (color)
        hist_patches = ax.hist(
            cell_type_rates_nonzero,
            bins=log_bins,
            alpha=0.6,
            color=colors_map[i],
            edgecolor="black",
        )
        legend_handles.append(hist_patches[2][0])  # Get first patch as handle
        legend_labels.append(f"n={len(cell_type_rates)}")

        # 2. Add mean line
        if len(cell_type_rates_nonzero) > 0:
            mean_rate = cell_type_rates_nonzero.mean()
            mean_line = ax.axvline(
                mean_rate,
                alpha=0.6,
                color="gray",
                linestyle="--",
                linewidth=2,
            )
            legend_handles.append(mean_line)
            legend_labels.append(f"Mean = {mean_rate:.2f} Hz")

        # 3. Add silent neurons
        if n_silent > 0:
            # Place bar just to the left of the visible range with annotation
            bar_position = global_x_min * 0.6
            bar_width = global_x_min * 0.2
            silent_bar = ax.bar(
                bar_position,
                n_silent,
                width=bar_width / 2,
                alpha=0.6,
                color=colors_map[i],
                edgecolor="black",
                linewidth=1.0,
                hatch="//",
            )
            # Add text annotation below the bar
            ax.text(
                bar_position,
                -ax.get_ylim()[1] * 0.05,
                "0",
                ha="center",
                va="top",
                fontsize=9,
            )
            legend_handles.append(silent_bar)
            legend_labels.append(f"Silent: {n_silent}")
        else:
            # Add legend entry for zero silent neurons
            empty_handle = ax.plot([], [], " ")[0]
            legend_handles.append(empty_handle)
            legend_labels.append("Silent: 0")

        ax.set_xlabel("Firing Rate (Hz)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(cell_type_names[i].capitalize(), fontsize=11)
        ax.tick_params(labelsize=9)

        # Use regular log scale - bar at 0 will be handled separately
        # Only set log scale if we have positive values
        if len(cell_type_rates_nonzero) > 0 and global_x_min > 0:
            ax.set_xscale("log")
            ax.set_xlim(global_x_min * 0.5, global_x_max)

            # Set nice round x-ticks: 0.01, 0.1, 1, 10, 100, etc. (excluding 0)
            log_min = np.floor(np.log10(global_x_min))
            log_max = np.ceil(np.log10(global_x_max))

            xticks = []
            for i_tick in range(int(log_min), int(log_max) + 1):
                tick_val = 10**i_tick
                # Exclude tick at 0 or below 0
                if (
                    tick_val > 0
                    and tick_val >= global_x_min * 0.9
                    and tick_val <= global_x_max
                ):
                    xticks.append(tick_val)

            ax.set_xticks(xticks)

            # Format tick labels nicely
            xticklabels = []
            for tick in xticks:
                if tick < 1:
                    xticklabels.append(f"{tick:.2f}")
                elif tick < 10:
                    xticklabels.append(f"{tick:.0f}")
                else:
                    xticklabels.append(f"{int(tick)}")
            ax.set_xticklabels(xticklabels)
        else:
            ax.set_xlim(0, 1)

        ax.legend(legend_handles, legend_labels, fontsize=9)
        ax.grid(True, alpha=0.3)

    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None

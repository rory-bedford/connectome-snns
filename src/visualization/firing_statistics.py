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
    ax: plt.Axes | None = None,
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
        Figure: Matplotlib figure object.
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

    # Create subplots
    if ax is None:
        fig, axes = plt.subplots(
            1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
        )
        if n_cell_types == 1:
            axes = [axes]
    else:
        fig = ax.get_figure()
        axes = [ax]

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

        ax.set_xlabel("Window Size (s)", fontsize=12)
        ax.set_ylabel("Mean Fano Factor", fontsize=12)
        ax.set_title(f"{cell_type_names[cell_type_idx]}", fontsize=14)
        # Only set log scale if we have positive window sizes
        if len(window_sizes_s) > 0 and np.all(np.array(window_sizes_s) > 0):
            ax.set_xscale("log")
        ax.axhline(
            1.0, color="black", linestyle="-", linewidth=1.5, alpha=0.7, zorder=1
        )
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Fano Factor vs Window Size", fontsize=16, y=1.02)
    plt.tight_layout()

    return fig


def plot_cv_histogram(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float = 1.0,
    bins: int = 50,
    ax: plt.Axes | None = None,
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
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        Figure: Matplotlib figure object.
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

    # Create subplots
    if ax is None:
        fig, axes = plt.subplots(
            1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
        )
        if n_cell_types == 1:
            axes = [axes]
    else:
        fig = ax.get_figure()
        axes = [ax]

    # Plot each cell type
    for i in range(n_cell_types):
        ax = axes[i] if n_cell_types > 1 or ax is None else axes[0]

        # Get CV values for this cell type across all batches
        cell_type_mask = cell_type_indices == i
        cv_cell_type = cv_values[:, cell_type_mask].flatten()
        cv_valid = cv_cell_type[~np.isnan(cv_cell_type)]

        if len(cv_valid) > 0:
            # Plot histogram
            ax.hist(
                cv_valid,
                bins=bins,
                alpha=0.6,
                edgecolor="black",
                linewidth=1.2,
                color=colors_map[i],
                label=f"n={len(cv_valid)}",
            )
            ax.axvline(
                cv_valid.mean(),
                color=colors_map[i],
                linestyle="--",
                linewidth=2,
                alpha=0.6,
                label=f"Mean: {cv_valid.mean():.2f}",
            )
            ax.set_xlabel("Coefficient of Variation (CV)", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(f"{cell_type_names[i]}", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(
                0.5,
                0.5,
                "No valid CV values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{cell_type_names[i]}", fontsize=14)

    plt.tight_layout()
    return fig


def plot_isi_histogram(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float = 1.0,
    bins: int = 50,
    ax: plt.Axes | None = None,
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
        Figure: Matplotlib figure object.
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
    time_indices = np.arange(n_steps, dtype=np.float32) * dt * 1e-3

    # Collect ISIs for each cell type
    isis_by_type = [[] for _ in range(n_cell_types)]

    for batch_idx in range(batch_size):
        for neuron_idx in range(n_neurons):
            # Find time indices where spikes occur for this neuron
            spike_indices = np.where(spike_trains[batch_idx, :, neuron_idx] > 0)[0]

            # Need at least 2 spikes to compute ISIs
            if len(spike_indices) < 2:
                continue

            # Convert to time units
            spike_times = time_indices[spike_indices]

            # Compute inter-spike intervals
            isis = spike_times[1:] - spike_times[:-1]

            # Add to appropriate cell type
            cell_type = cell_type_indices[neuron_idx]
            isis_by_type[cell_type].append(isis)

    # Collect all ISIs across cell types to determine global x-axis limit
    all_isis_combined = []
    for i in range(n_cell_types):
        if len(isis_by_type[i]) > 0:
            all_isis_array = np.concatenate(isis_by_type[i])
            all_isis_combined.append(all_isis_array)

    # Compute 99.5th percentile for x-axis limit
    if len(all_isis_combined) > 0:
        all_isis_flat = np.concatenate(all_isis_combined)
        x_limit = np.percentile(all_isis_flat, 99.5)
    else:
        x_limit = None

    # Create subplots
    if ax is None:
        fig, axes = plt.subplots(
            1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
        )
        if n_cell_types == 1:
            axes = [axes]
    else:
        fig = ax.get_figure()
        axes = [ax]

    # Plot each cell type
    for i in range(n_cell_types):
        ax = axes[i] if n_cell_types > 1 or ax is None else axes[0]

        if len(isis_by_type[i]) > 0:
            # Concatenate all ISIs for this cell type
            isis_np = np.concatenate(isis_by_type[i])

            # Plot histogram with x-limit based on 99.5th percentile
            ax.hist(
                isis_np,
                bins=bins,
                alpha=0.6,
                edgecolor="black",
                linewidth=1.2,
                color=colors_map[i],
                label=f"n={len(isis_np)}",
            )
            ax.axvline(
                isis_np.mean(),
                color=colors_map[i],
                linestyle="--",
                linewidth=2,
                alpha=0.6,
                label=f"Mean: {isis_np.mean():.4f} s",
            )
            ax.set_xlabel("Inter-Spike Interval (s)", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(f"{cell_type_names[i]}", fontsize=14)

            # Set x-axis limit to 98th percentile for better resolution
            if x_limit is not None:
                ax.set_xlim(0, x_limit)

            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(
                0.5,
                0.5,
                "No ISIs found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{cell_type_names[i]}", fontsize=14)

    plt.tight_layout()
    return fig


def plot_firing_rate_distribution(
    output_spikes: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    duration: float,
) -> plt.Figure:
    """Plot distribution of firing rates in the Dp network by cell type.

    Args:
        output_spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        duration (float): Total duration in milliseconds.

    Returns:
        plt.Figure: Matplotlib figure object containing the firing rate distribution.
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

    # Calculate firing rates (spikes per second)
    spike_counts = output_spikes[0].sum(axis=0)  # Total spikes per neuron
    firing_rates = spike_counts / (duration * 1e-3)  # Convert duration from ms to s

    # Filter out zero firing rates for log scale
    firing_rates_nonzero = firing_rates[firing_rates > 0]

    # Create subplots
    fig, axes = plt.subplots(
        1, n_cell_types, figsize=(6 * n_cell_types, 5), sharey=True
    )
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

        # Plot histogram for non-zero rates
        ax.hist(
            cell_type_rates_nonzero,
            bins=log_bins,
            alpha=0.6,
            color=colors_map[i],
            label=f"n={len(cell_type_rates)}",
            edgecolor="black",
        )

        # Add a separate bar for silent neurons at position 0 on the x-axis
        if n_silent > 0:
            # Place bar just to the left of the visible range with annotation
            bar_position = global_x_min * 0.6
            bar_width = global_x_min * 0.2
            ax.bar(
                bar_position,
                n_silent,
                width=bar_width / 2,
                alpha=0.6,
                color=colors_map[i],
                edgecolor="black",
                linewidth=1.0,
                label=f"Silent: {n_silent}",
                hatch="//",
            )
            # Add text annotation below the bar
            ax.text(
                bar_position,
                -ax.get_ylim()[1] * 0.05,
                "0",
                ha="center",
                va="top",
                fontsize=10,
            )

        # Add mean line
        if len(cell_type_rates_nonzero) > 0:
            mean_rate = cell_type_rates_nonzero.mean()
            ax.axvline(
                mean_rate,
                alpha=0.6,
                color=colors_map[i],
                linestyle="--",
                linewidth=2,
                label=f"Mean = {mean_rate:.2f} Hz",
            )

        ax.set_xlabel("Firing Rate (Hz)")
        ax.set_title(cell_type_names[i])

        # Use regular log scale - bar at 0 will be handled separately
        # Only set log scale if we have positive values
        if len(cell_type_rates_nonzero) > 0 and global_x_min > 0:
            ax.set_xscale("log")
            ax.set_xlim(global_x_min * 0.5, global_x_max)

            # Set nice round x-ticks: 0.01, 0.1, 1, 10, 100, etc.
            log_min = np.floor(np.log10(global_x_min))
            log_max = np.ceil(np.log10(global_x_max))

            xticks = []
            for i_tick in range(int(log_min), int(log_max) + 1):
                tick_val = 10**i_tick
                if tick_val >= global_x_min * 0.9 and tick_val <= global_x_max:
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

        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Number of Neurons")
    fig.suptitle("Firing Rate Distribution (log scale)", fontsize=14, y=1.02)
    plt.tight_layout()

    return fig

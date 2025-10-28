"""Visualization functions for firing statistics and spike train analysis."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from analysis.firing_statistics import (
    compute_spike_train_cv,
    compute_spike_train_fano_factor,
)


def plot_fano_factor_vs_window_size(
    spike_trains: torch.Tensor,
    window_sizes: list[int] | np.ndarray,
    ax: plt.Axes | None = None,
) -> Figure:
    """
    Plot mean Fano factor across all neurons as a function of window size.

    Args:
        spike_trains (torch.Tensor): Spike trains of shape (batch_size, n_steps, n_neurons).
        window_sizes (list[int] | np.ndarray): List of window sizes to evaluate.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        Figure: Matplotlib figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    mean_fano_factors = []
    std_fano_factors = []

    for window_size in window_sizes:
        fano_factors = compute_spike_train_fano_factor(spike_trains, window_size)
        # Flatten across batch and neurons, ignore NaNs
        fano_flat = fano_factors.flatten()
        fano_valid = fano_flat[~torch.isnan(fano_flat)]

        if len(fano_valid) > 0:
            mean_fano_factors.append(fano_valid.mean().item())
            std_fano_factors.append(fano_valid.std().item())
        else:
            mean_fano_factors.append(np.nan)
            std_fano_factors.append(np.nan)

    mean_fano_factors = np.array(mean_fano_factors)
    std_fano_factors = np.array(std_fano_factors)

    # Plot with error bars
    ax.errorbar(
        window_sizes,
        mean_fano_factors,
        yerr=std_fano_factors,
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Window Size (steps)", fontsize=12)
    ax.set_ylabel("Mean Fano Factor", fontsize=12)
    ax.set_title("Fano Factor vs Window Size", fontsize=14)
    ax.grid(True, alpha=0.3)

    return fig


def plot_cv_histogram(
    spike_trains: torch.Tensor,
    dt: float = 1.0,
    bins: int = 50,
    ax: plt.Axes | None = None,
) -> Figure:
    """
    Plot histogram of CV values across all neurons.

    Args:
        spike_trains (torch.Tensor): Spike trains of shape (batch_size, n_steps, n_neurons).
        dt (float): Time step duration for converting time steps to actual time units.
        bins (int): Number of histogram bins.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        Figure: Matplotlib figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    # Compute CV for all neurons
    cv_values = compute_spike_train_cv(spike_trains, dt=dt)

    # Flatten across batch and neurons, ignore NaNs
    cv_flat = cv_values.flatten()
    cv_valid = cv_flat[~torch.isnan(cv_flat)]

    if len(cv_valid) > 0:
        cv_np = cv_valid.cpu().numpy()

        # Plot histogram
        ax.hist(cv_np, bins=bins, alpha=0.7, edgecolor="black", linewidth=1.2)
        ax.axvline(
            cv_np.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {cv_np.mean():.2f}",
        )
        ax.set_xlabel("Coefficient of Variation (CV)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Distribution of CV Values Across Neurons", fontsize=14)
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

    return fig


def plot_isi_histogram(
    spike_trains: torch.Tensor,
    dt: float = 1.0,
    bins: int = 50,
    ax: plt.Axes | None = None,
) -> Figure:
    """
    Plot histogram of inter-spike intervals (ISIs) pooled across all neurons.

    Args:
        spike_trains (torch.Tensor): Spike trains of shape (batch_size, n_steps, n_neurons).
        dt (float): Time step duration for converting time steps to actual time units.
        bins (int): Number of histogram bins.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        Figure: Matplotlib figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    batch_size, n_steps, n_neurons = spike_trains.shape
    device = spike_trains.device

    # Create time indices
    time_indices = torch.arange(n_steps, device=device, dtype=torch.float32) * dt

    # Collect all ISIs across all neurons and batches
    all_isis = []

    for batch_idx in range(batch_size):
        for neuron_idx in range(n_neurons):
            # Find time indices where spikes occur for this neuron
            spike_indices = torch.where(spike_trains[batch_idx, :, neuron_idx] > 0)[0]

            # Need at least 2 spikes to compute ISIs
            if len(spike_indices) < 2:
                continue

            # Convert to time units
            spike_times = time_indices[spike_indices]

            # Compute inter-spike intervals
            isis = spike_times[1:] - spike_times[:-1]
            all_isis.append(isis)

    if len(all_isis) > 0:
        # Concatenate all ISIs
        all_isis_tensor = torch.cat(all_isis)
        isis_np = all_isis_tensor.cpu().numpy()

        # Plot histogram
        ax.hist(isis_np, bins=bins, alpha=0.7, edgecolor="black", linewidth=1.2)
        ax.axvline(
            isis_np.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {isis_np.mean():.2f}",
        )
        ax.set_xlabel("Inter-Spike Interval (time units)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Distribution of ISIs Across All Neurons", fontsize=14)
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

    return fig

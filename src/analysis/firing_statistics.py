"""Analysis functions for spike train statistics and firing rates."""

import numpy as np
from numpy.typing import NDArray


def compute_spike_train_cv(
    spike_trains: NDArray[np.int32], dt: float = 1.0
) -> NDArray[np.float32]:
    """
    Compute coefficient of variation (CV) for spike trains for each neuron.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons)
            or (batch_size, n_patterns, n_steps, n_neurons).
        dt (float): Time step duration in the desired time units (e.g., seconds or milliseconds).
            Since CV is dimensionless, the units cancel out, but consistent units should be used.
            Defaults to 1.0.

    Returns:
        NDArray[np.float32]: CV values of shape (batch_size, n_neurons) or
            (batch_size, n_patterns, n_neurons). Returns NaN for neurons with fewer than 3 spikes.
    """
    # Determine input dimensions
    if spike_trains.ndim == 4:
        batch_size, n_patterns, n_steps, n_neurons = spike_trains.shape
        output_shape = (batch_size, n_patterns, n_neurons)
    else:
        batch_size, n_steps, n_neurons = spike_trains.shape
        output_shape = (batch_size, n_neurons)
        # Add pattern dimension for uniform processing
        spike_trains = spike_trains[:, np.newaxis, :, :]  # (batch, 1, steps, neurons)
        n_patterns = 1

    # Initialize CV values with NaN
    cv_values = np.full(output_shape, np.nan, dtype=np.float32)

    # Create time indices
    time_indices = np.arange(n_steps, dtype=np.float32) * dt

    # Loop over batch, patterns, and neurons
    for batch_idx in range(batch_size):
        for pattern_idx in range(n_patterns):
            for neuron_idx in range(n_neurons):
                # Find time indices where spikes occur
                spike_indices = np.where(
                    spike_trains[batch_idx, pattern_idx, :, neuron_idx] > 0
                )[0]

                # Need at least 3 spikes to compute CV with unbiased std (at least 2 ISIs)
                if len(spike_indices) < 3:
                    continue

                # Convert to time units
                spike_times = time_indices[spike_indices]

                # Compute inter-spike intervals
                isis = spike_times[1:] - spike_times[:-1]

                # Compute CV = std(ISI) / mean(ISI)
                mean_isi = isis.mean()
                if mean_isi > 0:
                    std_isi = isis.std(ddof=1)  # unbiased std
                    if output_shape == (batch_size, n_neurons):
                        cv_values[batch_idx, neuron_idx] = std_isi / mean_isi
                    else:
                        cv_values[batch_idx, pattern_idx, neuron_idx] = (
                            std_isi / mean_isi
                        )

    return cv_values


def compute_spike_train_fano_factor(
    spike_trains: NDArray[np.int32], window_size: int
) -> NDArray[np.float32]:
    """
    Compute Fano factor for spike trains for each neuron.

    The Fano factor is computed as the variance of spike counts divided by the
    mean spike count across time windows. Fano factor is computed separately
    for each neuron in each batch.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        window_size (int): Size of the time window (in steps) for counting spikes.

    Returns:
        NDArray[np.float32]: Fano factor values of shape (batch_size, n_neurons).
            Returns NaN for neurons with zero mean spike count.
    """
    batch_size, n_steps, n_neurons = spike_trains.shape

    # Initialize Fano factor values with NaN
    fano_values = np.full((batch_size, n_neurons), np.nan, dtype=np.float32)

    # Calculate number of complete windows
    n_windows = n_steps // window_size

    if n_windows < 1:
        return fano_values

    # Vectorized computation: reshape all data at once
    # Trim to complete windows and reshape
    trimmed_length = n_windows * window_size
    trimmed_spikes = spike_trains[
        :, :trimmed_length, :
    ]  # (batch_size, trimmed_length, n_neurons)

    # Reshape to windows: (batch_size, n_windows, window_size, n_neurons)
    windowed_spikes = trimmed_spikes.reshape(
        batch_size, n_windows, window_size, n_neurons
    )

    # Sum over window dimension to get spike counts per window
    # Shape: (batch_size, n_windows, n_neurons)
    spike_counts = windowed_spikes.sum(axis=2)

    # Compute mean and variance across windows for each neuron in each batch
    # Shape: (batch_size, n_neurons)
    mean_counts = spike_counts.mean(axis=1)  # Mean across windows
    var_counts = spike_counts.var(axis=1, ddof=1)  # Variance across windows

    # Compute Fano factor = var/mean, avoiding division by zero
    # Only compute where mean > 0 and we have more than 1 window
    valid_mask = (mean_counts > 0) & (n_windows > 1)
    fano_values[valid_mask] = var_counts[valid_mask] / mean_counts[valid_mask]

    return fano_values


def compute_firing_rate_by_cell_type(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    duration: float,
) -> dict[int, dict[str, float]]:
    """
    Compute mean and standard deviation of firing rates by cell type.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (NDArray[np.int32]): Cell type indices of shape (n_neurons,).
            Each value indicates the cell type index for that neuron.
        duration (float): Duration of the spike train in milliseconds.

    Returns:
        dict[int, dict[str, float]]: Dictionary mapping cell type index to a dict with:
            - "mean_firing_rate_hz": Mean firing rate across neurons of this type (Hz)
            - "std_firing_rate_hz": Standard deviation of firing rates (Hz)
            - "n_silent_cells": Number of cells that don't fire at all
    """
    batch_size, n_steps, n_neurons = spike_trains.shape

    # Compute total spike count per neuron across all timesteps and batches
    # Sum over time and batch dimensions
    total_spikes = spike_trains.sum(axis=(0, 1))  # Shape: (n_neurons,)

    # Convert duration from ms to seconds for Hz calculation
    duration_sec = duration / 1000.0

    # Compute firing rate in Hz for each neuron
    firing_rates_hz = total_spikes / (duration_sec * batch_size)  # Shape: (n_neurons,)

    # Get unique cell types
    unique_cell_types = np.unique(cell_type_indices)

    # Compute statistics by cell type
    stats_by_type = {}
    for cell_type in unique_cell_types.tolist():
        # Get indices of neurons belonging to this cell type
        mask = cell_type_indices == cell_type
        cell_type_rates = firing_rates_hz[mask]

        # Compute mean and std
        mean_rate = float(cell_type_rates.mean())
        std_rate = (
            float(cell_type_rates.std(ddof=1)) if len(cell_type_rates) > 1 else 0.0
        )

        # Count cells that don't fire at all (firing rate = 0)
        n_silent = int((cell_type_rates == 0).sum())

        stats_by_type[cell_type] = {
            "mean_firing_rate_hz": mean_rate,
            "std_firing_rate_hz": std_rate,
            "n_silent_cells": n_silent,
        }

    return stats_by_type


def compute_cv_by_cell_type(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    dt: float = 1.0,
) -> dict[int, dict[str, float]]:
    """
    Compute mean and standard deviation of coefficient of variation (CV) by cell type.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (NDArray[np.int32]): Cell type indices of shape (n_neurons,).
            Each value indicates the cell type index for that neuron.
        dt (float): Time step duration in the desired time units (e.g., seconds or milliseconds).
            Defaults to 1.0.

    Returns:
        dict[int, dict[str, float]]: Dictionary mapping cell type index to a dict with:
            - "mean_cv": Mean CV across neurons of this type
            - "std_cv": Standard deviation of CV values
    """
    # Compute CV for all neurons
    cv_values = compute_spike_train_cv(
        spike_trains, dt=dt
    )  # Shape: (batch_size, n_neurons)

    # Average CV across batches for each neuron
    # Suppress warning for all-NaN slices (expected when neurons have no spikes)
    with np.errstate(invalid="ignore"):
        cv_per_neuron = np.nanmean(cv_values, axis=0)  # Shape: (n_neurons,)

    # Get unique cell types
    unique_cell_types = np.unique(cell_type_indices)

    # Compute statistics by cell type
    stats_by_type = {}
    for cell_type in unique_cell_types.tolist():
        # Get indices of neurons belonging to this cell type
        mask = cell_type_indices == cell_type
        cell_type_cvs = cv_per_neuron[mask]

        # Filter out NaN values
        valid_cvs = cell_type_cvs[~np.isnan(cell_type_cvs)]

        if len(valid_cvs) > 0:
            mean_cv = float(valid_cvs.mean())
            std_cv = float(valid_cvs.std(ddof=1)) if len(valid_cvs) > 1 else 0.0
        else:
            mean_cv = float("nan")
            std_cv = float("nan")

        stats_by_type[cell_type] = {
            "mean_cv": mean_cv,
            "std_cv": std_cv,
        }

    return stats_by_type

"""Analysis functions for spike train statistics and firing rates."""

import torch


def compute_spike_train_cv(spike_trains: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """
    Compute coefficient of variation (CV) for spike trains for each neuron.

    The CV is computed as the standard deviation of inter-spike intervals (ISIs)
    divided by the mean ISI. CV is computed separately for each neuron in each batch.

    Args:
        spike_trains (torch.Tensor): Spike trains of shape (batch_size, n_steps, n_neurons).
        dt (float): Time step duration for converting time steps to actual time units.
            Defaults to 1.0 (time in steps).

    Returns:
        torch.Tensor: CV values of shape (batch_size, n_neurons).
            Returns NaN for neurons with fewer than 2 spikes.
    """
    batch_size, n_steps, n_neurons = spike_trains.shape
    device = spike_trains.device

    # Initialize CV values with NaN
    cv_values = torch.full((batch_size, n_neurons), float("nan"), device=device)

    # Create time indices
    time_indices = torch.arange(n_steps, device=device, dtype=torch.float32) * dt

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

            # Compute CV = std(ISI) / mean(ISI)
            mean_isi = isis.mean()
            if mean_isi > 0:
                std_isi = isis.std(unbiased=True)
                cv_values[batch_idx, neuron_idx] = std_isi / mean_isi

    return cv_values


def compute_spike_train_fano_factor(
    spike_trains: torch.Tensor, window_size: int
) -> torch.Tensor:
    """
    Compute Fano factor for spike trains for each neuron.

    The Fano factor is computed as the variance of spike counts divided by the
    mean spike count across time windows. Fano factor is computed separately
    for each neuron in each batch.

    Args:
        spike_trains (torch.Tensor): Spike trains of shape (batch_size, n_steps, n_neurons).
        window_size (int): Size of the time window (in steps) for counting spikes.

    Returns:
        torch.Tensor: Fano factor values of shape (batch_size, n_neurons).
            Returns NaN for neurons with zero mean spike count.
    """
    batch_size, n_steps, n_neurons = spike_trains.shape
    device = spike_trains.device

    # Initialize Fano factor values with NaN
    fano_values = torch.full((batch_size, n_neurons), float("nan"), device=device)

    # Calculate number of complete windows
    n_windows = n_steps // window_size

    if n_windows < 1:
        return fano_values

    for batch_idx in range(batch_size):
        for neuron_idx in range(n_neurons):
            # Extract spike train for this neuron
            neuron_spikes = spike_trains[
                batch_idx, : n_windows * window_size, neuron_idx
            ]

            # Reshape into windows and count spikes per window
            # Shape: (n_windows, window_size) -> (n_windows,)
            spike_counts = (
                neuron_spikes.reshape(n_windows, window_size).sum(dim=1).float()
            )

            # Compute Fano factor = var(spike_counts) / mean(spike_counts)
            mean_count = spike_counts.mean()
            if mean_count > 0:
                var_count = spike_counts.var(unbiased=True)
                fano_values[batch_idx, neuron_idx] = var_count / mean_count

    return fano_values

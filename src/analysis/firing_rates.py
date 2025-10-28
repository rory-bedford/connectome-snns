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

    Example:
        >>> spike_trains = torch.zeros(2, 1000, 50)  # 2 batches, 1000 steps, 50 neurons
        >>> spike_trains[0, [10, 25, 60], 0] = 1  # Add some spikes to neuron 0
        >>> cv = compute_spike_train_cv(spike_trains, dt=0.1)
        >>> print(cv.shape)
        torch.Size([2, 50])
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

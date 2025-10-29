"""Analysis functions for spike train statistics and firing rates."""

import torch


def compute_spike_train_cv(spike_trains: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """
    Compute coefficient of variation (CV) for spike trains for each neuron.

    The CV is computed as the standard deviation of inter-spike intervals (ISIs)
    divided by the mean ISI. CV is computed separately for each neuron in each batch.

    Args:
        spike_trains (torch.Tensor): Spike trains of shape (batch_size, n_steps, n_neurons).
        dt (float): Time step duration in the desired time units (e.g., seconds or milliseconds).
            Since CV is dimensionless, the units cancel out, but consistent units should be used.
            Defaults to 1.0.

    Returns:
        torch.Tensor: CV values of shape (batch_size, n_neurons).
            Returns NaN for neurons with fewer than 3 spikes.
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
            if mean_count > 0 and n_windows > 1:
                var_count = spike_counts.var(unbiased=True)
                fano_values[batch_idx, neuron_idx] = var_count / mean_count

    return fano_values


def compute_firing_rate_by_cell_type(
    spike_trains: torch.Tensor,
    cell_type_indices: torch.Tensor,
    duration: float,
) -> dict[int, dict[str, float]]:
    """
    Compute mean and standard deviation of firing rates by cell type.

    Args:
        spike_trains (torch.Tensor): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (torch.Tensor): Cell type indices of shape (n_neurons,).
            Each value indicates the cell type index for that neuron.
        duration (float): Duration of the spike train in milliseconds.

    Returns:
        dict[int, dict[str, float]]: Dictionary mapping cell type index to a dict with:
            - "mean_firing_rate_hz": Mean firing rate across neurons of this type (Hz)
            - "std_firing_rate_hz": Standard deviation of firing rates (Hz)
    """
    batch_size, n_steps, n_neurons = spike_trains.shape

    # Compute total spike count per neuron across all timesteps and batches
    # Sum over time and batch dimensions
    total_spikes = spike_trains.sum(dim=(0, 1))  # Shape: (n_neurons,)

    # Convert duration from ms to seconds for Hz calculation
    duration_sec = duration / 1000.0

    # Compute firing rate in Hz for each neuron
    firing_rates_hz = total_spikes / (duration_sec * batch_size)  # Shape: (n_neurons,)

    # Get unique cell types
    unique_cell_types = torch.unique(cell_type_indices)

    # Compute statistics by cell type
    stats_by_type = {}
    for cell_type in unique_cell_types.tolist():
        # Get indices of neurons belonging to this cell type
        mask = cell_type_indices == cell_type
        cell_type_rates = firing_rates_hz[mask]

        # Compute mean and std
        mean_rate = cell_type_rates.mean().item()
        std_rate = (
            cell_type_rates.std(unbiased=True).item()
            if len(cell_type_rates) > 1
            else 0.0
        )

        stats_by_type[cell_type] = {
            "mean_firing_rate_hz": mean_rate,
            "std_firing_rate_hz": std_rate,
        }

    return stats_by_type


def compute_cv_by_cell_type(
    spike_trains: torch.Tensor,
    cell_type_indices: torch.Tensor,
    dt: float = 1.0,
) -> dict[int, dict[str, float]]:
    """
    Compute mean and standard deviation of coefficient of variation (CV) by cell type.

    Args:
        spike_trains (torch.Tensor): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (torch.Tensor): Cell type indices of shape (n_neurons,).
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
    cv_per_neuron = cv_values.nanmean(dim=0)  # Shape: (n_neurons,)

    # Get unique cell types
    unique_cell_types = torch.unique(cell_type_indices)

    # Compute statistics by cell type
    stats_by_type = {}
    for cell_type in unique_cell_types.tolist():
        # Get indices of neurons belonging to this cell type
        mask = cell_type_indices == cell_type
        cell_type_cvs = cv_per_neuron[mask]

        # Filter out NaN values
        valid_cvs = cell_type_cvs[~torch.isnan(cell_type_cvs)]

        if len(valid_cvs) > 0:
            mean_cv = valid_cvs.mean().item()
            std_cv = valid_cvs.std(unbiased=True).item() if len(valid_cvs) > 1 else 0.0
        else:
            mean_cv = float("nan")
            std_cv = float("nan")

        stats_by_type[cell_type] = {
            "mean_cv": mean_cv,
            "std_cv": std_cv,
        }

    return stats_by_type

"""Memory-efficient firing rate computation for larger-than-memory spike datasets.

This module provides functions to compute firing rates from spike data stored in
Zarr format using Dask for chunked computation. Designed for datasets that don't
fit entirely in memory but produce firing rate arrays that do.
"""

import dask.array as da
import numpy as np
import zarr
from pathlib import Path
from typing import Union


def compute_firing_rates_from_zarr(
    zarr_path: Union[Path, str],
    dataset_name: str,
    dt: float,
) -> np.ndarray:
    """Compute firing rates from Zarr spike data using chunked Dask computation.

    This function efficiently computes firing rates from spike data stored in Zarr
    format without loading the entire array into memory. It processes data in chunks
    using Dask, then returns the final result as a NumPy array that fits in memory.

    Args:
        zarr_path (Path | str): Path to the Zarr store containing spike data.
        dataset_name (str): Name of the dataset within the Zarr store (e.g., "input_spikes", "output_spikes").
        dt (float): Timestep in milliseconds.

    Returns:
        np.ndarray: Firing rates in Hz with shape (batch_size, n_patterns, n_neurons).
            This array is the result of summing over the time dimension and dividing
            by the total duration in seconds.

    Notes:
        - Input spike data expected shape: (batch_size, n_patterns, n_timesteps, n_neurons)
        - Output firing rate shape: (batch_size, n_patterns, n_neurons)
        - Firing rate = spike_count / duration_in_seconds
        - Uses Dask for chunked computation to handle spike arrays larger than memory
        - Returns NumPy array (assumes firing rates fit in memory)

    Example:
        >>> from pathlib import Path
        >>> firing_rates = compute_firing_rates_from_zarr(
        ...     zarr_path=Path("results/spike_data.zarr"),
        ...     dataset_name="output_spikes",
        ...     dt=1.0
        ... )
        >>> firing_rates.shape
        (10, 5, 1000)  # (batch_size, n_patterns, n_neurons)
    """
    # Load zarr array
    root = zarr.open(zarr_path, mode="r")
    spikes_zarr = root[dataset_name]

    # Get dimensions
    batch_size, n_patterns, n_timesteps, n_neurons = spikes_zarr.shape
    duration_s = n_timesteps * dt * 1e-3  # Convert ms to seconds

    # Wrap in Dask array for chunked computation
    spikes_dask = da.from_zarr(spikes_zarr)

    # Sum over time dimension: (batch_size, n_patterns, n_neurons)
    spike_counts = spikes_dask.sum(axis=2)

    # Convert to firing rates (Hz) and compute to NumPy array
    firing_rates = (spike_counts / duration_s).compute()

    return firing_rates

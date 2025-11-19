"""Analysis utilities for network statistics."""

import pandas as pd
from numpy.typing import NDArray
import numpy as np

from .firing_statistics import (
    compute_firing_rate_by_cell_type,
    compute_cv_by_cell_type,
)
from .voltage_statistics import compute_membrane_potential_by_cell_type


def compute_network_statistics(
    output_spikes: NDArray,
    output_voltages: NDArray,
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    duration: float,
    dt: float,
) -> pd.DataFrame:
    """
    Compute all network statistics and return as a DataFrame.

    Combines firing rate, CV, and voltage statistics into one comprehensive
    DataFrame with one row per cell type.

    Args:
        output_spikes (NDArray): Spike trains of shape (batch_size, n_steps, n_neurons)
        output_voltages (NDArray): Membrane voltages of shape (batch_size, n_steps, n_neurons)
        cell_type_indices (NDArray[np.int32]): Cell type indices of shape (n_neurons,)
        cell_type_names (list[str]): Names of cell types
        duration (float): Simulation duration in milliseconds
        dt (float): Timestep in milliseconds

    Returns:
        pd.DataFrame: DataFrame with columns for cell type and all computed statistics
    """
    # Compute all statistics
    firing_rate_stats = compute_firing_rate_by_cell_type(
        spike_trains=output_spikes,
        cell_type_indices=cell_type_indices,
        duration=duration,
    )

    cv_stats = compute_cv_by_cell_type(
        spike_trains=output_spikes,
        cell_type_indices=cell_type_indices,
        dt=dt,
    )

    voltage_stats = compute_membrane_potential_by_cell_type(
        voltages=output_voltages,
        cell_type_indices=cell_type_indices,
    )

    # Combine all statistics into a single DataFrame
    combined_data = []
    for cell_type_idx in firing_rate_stats.keys():
        row = {
            "cell_type": cell_type_idx,
            "cell_type_name": cell_type_names[cell_type_idx],
            # Firing rate statistics
            "mean_firing_rate_hz": firing_rate_stats[cell_type_idx][
                "mean_firing_rate_hz"
            ],
            "std_firing_rate_hz": firing_rate_stats[cell_type_idx][
                "std_firing_rate_hz"
            ],
            "n_silent_cells": firing_rate_stats[cell_type_idx]["n_silent_cells"],
            # CV statistics
            "mean_cv": cv_stats[cell_type_idx]["mean_cv"],
            "std_cv": cv_stats[cell_type_idx]["std_cv"],
            # Voltage statistics
            "mean_of_mean_voltages": voltage_stats[cell_type_idx]["mean_of_means"],
            "std_of_mean_voltages": voltage_stats[cell_type_idx]["std_of_means"],
            "mean_of_std_voltages": voltage_stats[cell_type_idx]["mean_of_stds"],
            "std_of_std_voltages": voltage_stats[cell_type_idx]["std_of_stds"],
        }
        combined_data.append(row)

    return pd.DataFrame(combined_data)


__all__ = ["compute_network_statistics"]

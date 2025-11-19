"""
PyTorch DataLoaders for generating input spike trains.

This module provides DataLoader implementations for generating various types
of spike train inputs for spiking neural network simulations, compatible with
PyTorch's training pipeline.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Union, Tuple


class PoissonSpikeDataset(Dataset):
    """
    Dataset for generating Poisson spike trains.

    This dataset generates spike trains on-the-fly according to a homogeneous
    Poisson process with specified firing rates for each neuron. Each call to
    __getitem__ returns a new randomly generated spike train chunk.

    Args:
        firing_rates (Union[np.ndarray, torch.Tensor]): Firing rates in Hz for each neuron (num_neurons,).
        chunk_size (int): Number of simulation time steps per chunk.
        dt (float): Simulation time step in milliseconds.
        device (Union[str, torch.device]): Device to generate spikes on. Default: "cpu".

    Example:
        >>> firing_rates = np.array([10.0, 10.0, 20.0, 20.0])
        >>> dataset = PoissonSpikeDataset(
        ...     firing_rates=firing_rates,
        ...     chunk_size=1000,
        ...     dt=1.0,
        ...     device="cuda"
        ... )
        >>> # Generate spikes indefinitely
        >>> for i in range(100):
        ...     spikes = dataset[i]  # shape: (chunk_size, num_neurons)
    """

    def __init__(
        self,
        firing_rates: Union[np.ndarray, torch.Tensor],
        chunk_size: int,
        dt: float,
        device: Union[str, torch.device] = "cpu",
    ):
        self.chunk_size = chunk_size
        self.dt = dt
        self.device = device

        # Convert firing rates to torch tensor on device
        if isinstance(firing_rates, np.ndarray):
            self.firing_rates = torch.from_numpy(firing_rates).float().to(device)
        elif isinstance(firing_rates, torch.Tensor):
            self.firing_rates = firing_rates.float().to(device)
        else:
            raise TypeError("firing_rates must be a numpy array or torch tensor")

        self.num_neurons = len(self.firing_rates)

        # Pre-compute spike probabilities: rate (Hz) * dt (ms) * 1e-3 (ms->s conversion)
        self.spike_probs = self.firing_rates * dt * 1e-3  # Shape: (num_neurons,)

    def __len__(self) -> int:
        """Return a large number to allow indefinite iteration."""
        return 2**31 - 1  # Max int32 value for effectively infinite iteration

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Generate a single chunk of Poisson spike trains.

        Args:
            idx (int): Index of the chunk (not used in generation, but required by Dataset).

        Returns:
            torch.Tensor: Boolean tensor of shape (chunk_size, num_neurons) with spike times.
        """
        # Generate random values and compare with spike probabilities
        random_vals = torch.rand(self.chunk_size, self.num_neurons, device=self.device)
        spikes = random_vals < self.spike_probs

        return spikes


class PoissonOdourDataset(Dataset):
    """
    Dataset for generating Poisson spike trains with multiple input patterns (e.g., odours).

    This dataset generates spikes for multiple distinct input patterns, cycling through them
    indefinitely. Each __getitem__ call returns spikes for ONE pattern. The index cycles
    through patterns repeatedly, allowing infinite iteration.

    Args:
        firing_rates (Union[np.ndarray, torch.Tensor]): Firing rates in Hz for each pattern.
            Shape: (n_patterns, n_neurons)
        chunk_size (float): Duration of each chunk in milliseconds.
        dt (float): Simulation time step in milliseconds.
        device (Union[str, torch.device]): Device to generate spikes on. Default: "cpu".

    Example:
        >>> firing_rates = np.random.uniform(5, 20, (10, 100))  # 10 patterns, 100 neurons
        >>> dataset = PoissonOdourDataset(
        ...     firing_rates=firing_rates,
        ...     chunk_size=100.0,
        ...     dt=1.0,
        ...     device="cuda"
        ... )
        >>> # Use with DataLoader and collate_pattern_batches
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=10, collate_fn=collate_pattern_batches)
        >>> for spikes, pattern_indices in loader:
        ...     # spikes shape: (1, 10, n_steps, n_neurons) - 1 trial of 10 patterns
        ...     # Cycles through patterns indefinitely: 0,1,2,...,9,0,1,2,...
        ...     pass
    """

    def __init__(
        self,
        firing_rates: Union[np.ndarray, torch.Tensor],
        chunk_size: float,
        dt: float,
        device: Union[str, torch.device] = "cpu",
    ):
        # Convert firing rates to torch tensor on device
        if isinstance(firing_rates, np.ndarray):
            self.firing_rates = torch.from_numpy(firing_rates).float().to(device)
        elif isinstance(firing_rates, torch.Tensor):
            self.firing_rates = firing_rates.float().to(device)
        else:
            raise TypeError("firing_rates must be a numpy array or torch tensor")

        if self.firing_rates.ndim != 2:
            raise ValueError(
                "firing_rates must be 2D array of shape (n_patterns, n_neurons)"
            )

        self.n_patterns = self.firing_rates.shape[0]
        self.n_neurons = self.firing_rates.shape[1]
        self.chunk_size = chunk_size
        self.dt = dt
        self.device = device
        self.n_steps = int(chunk_size / dt)

        # Pre-compute spike probabilities: rate (Hz) * dt (ms) * 1e-3 (ms->s conversion)
        self.spike_probs = self.firing_rates * dt * 1e-3

    def __len__(self) -> int:
        """Return a large number to allow indefinite iteration."""
        return 2**31 - 1  # Effectively infinite

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Generate Poisson spikes for one pattern, cycling through patterns indefinitely.

        Args:
            idx (int): Flat index that cycles through patterns.

        Returns:
            Tuple of (spikes, pattern_idx) where:
                - spikes: torch.Tensor of shape (n_steps, n_neurons)
                - pattern_idx: int indicating which pattern this is (0 to n_patterns-1)
        """
        # Cycle through patterns indefinitely
        pattern_idx = idx % self.n_patterns

        # Select spike probabilities for this pattern
        spike_probs = self.spike_probs[pattern_idx]

        # Generate Poisson spikes: (n_steps, n_neurons)
        random_vals = torch.rand(self.n_steps, self.n_neurons, device=self.device)
        spikes = random_vals < spike_probs

        return spikes, pattern_idx


def collate_pattern_batches(batch):
    """
    Collate function for organizing pattern data with batch dimension.

    The DataLoader fetches batch_size * n_patterns items. This function organizes them
    into (batch_size, n_patterns, n_steps, n_neurons) where batch_size represents
    independent repeats/trials of all patterns.

    Args:
        batch: List of (spikes, pattern_idx) tuples from PoissonOdourDataset.
               Each spikes tensor has shape (n_steps, n_neurons).
               Length of batch = batch_size * n_patterns

    Returns:
        Tuple of (batched_spikes, pattern_indices) where:
            - batched_spikes: torch.Tensor of shape (batch_size, n_patterns, n_steps, n_neurons)
            - pattern_indices: List of unique pattern indices

    Example:
        >>> # DataLoader with batch_size=5, dataset has 10 patterns
        >>> # DataLoader fetches 50 items total (5 repeats × 10 patterns)
        >>> # batch = [(spikes0, 0), (spikes1, 1), ..., (spikes9, 9),  # First repeat
        >>> #          (spikes0, 0), (spikes1, 1), ..., (spikes9, 9),  # Second repeat
        >>> #          ...]  # 5 repeats total
        >>> # Each spikes has shape (n_steps, n_neurons)
        >>> #
        >>> # Result after collation:
        >>> # batched_spikes.shape = (5, 10, n_steps, n_neurons)
        >>> #                         ↑  ↑   ↑       ↑
        >>> #                         │  │   │       └─ neurons
        >>> #                         │  │   └───────── time steps
        >>> #                         │  └───────────── patterns
        >>> #                         └──────────────── batch (repeats/trials)
    """
    spikes_list = []
    pattern_indices = []

    for spikes, pattern_idx in batch:
        spikes_list.append(spikes)  # Each is (n_steps, n_neurons)
        pattern_indices.append(pattern_idx)

    # Stack all items: (batch_size * n_patterns, n_steps, n_neurons)
    all_stacked = torch.stack(spikes_list, dim=0)

    # Determine n_patterns from the cycling pattern indices
    # Find where pattern_idx repeats (e.g., [0,1,2,...,9,0,1,2,...] -> n_patterns=10)
    unique_patterns = []
    for idx in pattern_indices:
        if idx not in unique_patterns:
            unique_patterns.append(idx)
        else:
            break  # Pattern cycled, we found n_patterns
    n_patterns = len(unique_patterns)

    # Reshape to (batch_size, n_patterns, n_steps, n_neurons)
    batch_size = len(batch) // n_patterns
    n_steps = all_stacked.shape[1]
    n_neurons = all_stacked.shape[2]
    batched_spikes = all_stacked.reshape(batch_size, n_patterns, n_steps, n_neurons)

    return batched_spikes, unique_patterns

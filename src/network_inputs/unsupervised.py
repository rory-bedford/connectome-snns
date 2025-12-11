"""
Unsupervised input generation for spiking neural networks.

This module provides datasets for generating spike trains on-the-fly during
network simulation and training. Supports both homogeneous Poisson processes
(constant rates) and inhomogeneous Poisson processes (time-varying rates).

These datasets are designed for unsupervised learning scenarios where spike
patterns are generated dynamically rather than loaded from disk.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Union, Tuple
from .rate_processes import OrnsteinUhlenbeckRateProcess


class HomogeneousPoissonSpikeDataset(Dataset):
    """
    Dataset for generating Poisson spike trains with optional multiple patterns.

    This dataset generates spike trains on-the-fly according to a homogeneous
    Poisson process. Supports both single and multiple pattern cases:
    - 1D firing_rates (n_neurons,) - always returns pattern_idx 0
    - 2D firing_rates (n_patterns, n_neurons) - cycles through patterns

    Args:
        firing_rates (Union[np.ndarray, torch.Tensor]): Firing rates in Hz.
            Shape: (n_neurons,) or (n_patterns, n_neurons).
            1D inputs are automatically reshaped to (1, n_neurons).
        chunk_size (float): Duration of each chunk in milliseconds.
        dt (float): Simulation time step in milliseconds.

    Example:
        >>> # Single pattern (homogeneous)
        >>> firing_rates = np.array([10.0, 10.0, 20.0, 20.0])
        >>> dataset = HomogeneousPoissonSpikeDataset(firing_rates, chunk_size=100.0, dt=1.0)
        >>> spikes, pattern_idx = dataset[0]  # pattern_idx always 0
        >>>
        >>> # Multiple patterns (e.g., odours)
        >>> firing_rates = np.random.uniform(5, 20, (10, 100))
        >>> dataset = HomogeneousPoissonSpikeDataset(firing_rates, chunk_size=100.0, dt=1.0)
        >>> spikes, pattern_idx = dataset[15]  # pattern_idx = 15 % 10 = 5
    """

    def __init__(
        self,
        firing_rates: Union[np.ndarray, torch.Tensor],
        chunk_size: float,
        dt: float,
    ):
        # Convert firing rates to torch tensor (CPU)
        if isinstance(firing_rates, np.ndarray):
            firing_rates = torch.from_numpy(firing_rates).float()
        elif isinstance(firing_rates, torch.Tensor):
            firing_rates = firing_rates.float().cpu()
        else:
            raise TypeError("firing_rates must be a numpy array or torch tensor")

        # Auto-reshape 1D to 2D: (n_neurons,) -> (1, n_neurons)
        if firing_rates.ndim == 1:
            firing_rates = firing_rates.unsqueeze(0)
        elif firing_rates.ndim != 2:
            raise ValueError(
                "firing_rates must be 1D (n_neurons,) or 2D (n_patterns, n_neurons)"
            )

        self.firing_rates = firing_rates
        self.n_patterns = self.firing_rates.shape[0]
        self.n_neurons = self.firing_rates.shape[1]
        self.chunk_size = chunk_size
        self.dt = dt
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
        random_vals = torch.rand(self.n_steps, self.n_neurons)
        spikes = random_vals < spike_probs

        return spikes, pattern_idx


def collate_pattern_batches(batch):
    """
    Collate function for organizing pattern data with batch dimension.

    The DataLoader fetches batch_size * n_patterns items. This function organizes them
    into (batch_size, n_patterns, n_steps, n_neurons) where batch_size represents
    independent repeats/trials of all patterns.

    Args:
        batch: List of (spikes, pattern_idx) tuples from HomogeneousPoissonDataset.
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


def _collate_single_pattern(batch):
    """
    Collate function for single pattern case (1D firing rates).

    Args:
        batch: List of (spikes, pattern_idx) tuples.
               Each spikes tensor has shape (n_steps, n_neurons).

    Returns:
        Tuple of (batched_spikes, pattern_indices) where:
            - batched_spikes: torch.Tensor of shape (batch_size, n_steps, n_neurons)
            - pattern_indices: List containing single pattern index [0]
    """
    spikes_list = []
    for spikes, pattern_idx in batch:
        spikes_list.append(spikes)  # Each is (n_steps, n_neurons)

    # Stack into (batch_size, n_steps, n_neurons)
    batched_spikes = torch.stack(spikes_list, dim=0)

    return batched_spikes, [0]


class HomogeneousPoissonSpikeDataLoader(DataLoader):
    """
    DataLoader for HomogeneousPoissonSpikeDataset with pattern-aware batching.

    This DataLoader wraps HomogeneousPoissonSpikeDataset and handles both single and
    multiple pattern cases:
    - Single pattern (1D firing_rates): Returns (batch_size, n_steps, n_neurons)
    - Multiple patterns (2D firing_rates): Returns (batch_size, n_patterns, n_steps, n_neurons)

    For multiple patterns, the batch_size parameter represents the number of independent
    repeats/trials of all patterns. The actual number of items fetched is batch_size * n_patterns.

    Args:
        firing_rates (Union[np.ndarray, torch.Tensor]): Firing rates in Hz.
            Shape: (n_neurons,) or (n_patterns, n_neurons).
        chunk_size (float): Duration of each chunk in milliseconds.
        dt (float): Simulation time step in milliseconds.
        batch_size (int): Number of independent samples (single pattern) or repeats (multiple patterns).
        device (Union[str, torch.device]): Device to generate spikes on. Default: "cpu".
        num_workers (int): Number of subprocesses for data loading. Default: 0.
        **kwargs: Additional keyword arguments passed to DataLoader.

    Example:
        >>> # Single pattern case (1D firing rates)
        >>> firing_rates = np.array([10.0, 10.0, 20.0, 20.0])
        >>> dataloader = HomogeneousPoissonSpikeDataLoader(
        ...     firing_rates=firing_rates,
        ...     chunk_size=100.0,
        ...     dt=1.0,
        ...     batch_size=32,
        ...     device="cuda"
        ... )
        >>> for batched_spikes, pattern_indices in dataloader:
        ...     # batched_spikes.shape = (32, n_steps, 4)
        ...     break
        >>>
        >>> # Multiple patterns case (2D firing rates)
        >>> firing_rates = np.random.uniform(5, 20, (10, 100))
        >>> dataloader = HomogeneousPoissonSpikeDataLoader(
        ...     firing_rates=firing_rates,
        ...     chunk_size=100.0,
        ...     dt=1.0,
        ...     batch_size=5,
        ...     device="cuda"
        ... )
        >>> for batched_spikes, pattern_indices in dataloader:
        ...     # batched_spikes.shape = (5, 10, n_steps, 100)
        ...     #                         ↑  ↑   ↑       ↑
        ...     #                         │  │   │       └─ neurons
        ...     #                         │  │   └───────── time steps
        ...     #                         │  └───────────── patterns
        ...     #                         └──────────────── batch (repeats)
        ...     break
    """

    def __init__(
        self,
        firing_rates: Union[np.ndarray, torch.Tensor],
        chunk_size: float,
        dt: float,
        batch_size: int,
        device: Union[str, torch.device] = "cpu",
        num_workers: int = 0,
        **kwargs,
    ):
        # Store device for later use
        self.device = torch.device(device)

        # Create dataset (no device handling in dataset)
        dataset = HomogeneousPoissonSpikeDataset(
            firing_rates=firing_rates,
            chunk_size=chunk_size,
            dt=dt,
        )

        # Get number of patterns from dataset
        n_patterns = dataset.n_patterns

        # Choose collate function and batch size based on number of patterns
        if n_patterns == 1:
            # Single pattern: use simple batching
            collate_fn = _collate_single_pattern
            actual_batch_size = batch_size
        else:
            # Multiple patterns: use pattern-aware batching
            collate_fn = collate_pattern_batches
            actual_batch_size = batch_size * n_patterns

        # Initialize DataLoader with appropriate collate function
        super().__init__(
            dataset=dataset,
            batch_size=actual_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            **kwargs,
        )


class InhomogeneousPoissonSpikeDataset(Dataset):
    """
    Dataset for generating inhomogeneous Poisson spikes from time-varying rates.

    Takes an OrnsteinUhlenbeckRateProcess as input and generates spikes by sampling
    from an inhomogeneous Poisson process with the time-varying rates.

    This allows separating rate dynamics (OU process in pattern space)
    from spike generation, enabling flexible composition of rate processes
    with Poisson sampling.

    Args:
        rate_process: OrnsteinUhlenbeckRateProcess that generates time-varying rates.
        return_rates: If True, __getitem__ returns tuple (spikes, rates, weights)
            for diagnostic/visualization purposes. If False, returns only spikes. Default: False.
            Note: This requires the underlying rate_process to have return_rates=True.

    Attributes:
        dt: Timestep in milliseconds (inherited from rate_process).
        n_neurons: Number of neurons (inherited from rate_process).
        chunk_size: Chunk size (inherited from rate_process).

    Example:
        >>> from src.network_inputs.rate_processes import OrnsteinUhlenbeckRateProcess
        >>> from src.network_inputs.odourants import generate_odour_firing_rates
        >>>
        >>> # Generate odour patterns
        >>> patterns = generate_odour_firing_rates(...)  # (20, 5000)
        >>>
        >>> # OU-driven inhomogeneous Poisson spikes
        >>> rate_process = OrnsteinUhlenbeckRateProcess(
        ...     patterns=patterns,
        ...     chunk_size=1000,
        ...     dt=0.1,
        ...     tau=50.0,
        ...     r_mean=1.0,
        ...     sigma=0.5
        ... )
        >>> dataset = InhomogeneousPoissonSpikeDataset(rate_process)
        >>> spikes = dataset[0]  # Shape: (1000, 5000)
        >>> spikes.dtype
        torch.bool
        >>>
        >>> # With diagnostics enabled
        >>> rate_process_diag = OrnsteinUhlenbeckRateProcess(
        ...     patterns=patterns,
        ...     chunk_size=1000,
        ...     dt=0.1,
        ...     tau=50.0,
        ...     return_rates=True
        ... )
        >>> dataset_diag = InhomogeneousPoissonSpikeDataset(
        ...     rate_process=rate_process_diag,
        ...     return_rates=True
        ... )
        >>> spikes, rates, weights = dataset_diag[0]
        >>> # spikes: (1000, 5000), rates: (1000, 5000), weights: (1000, 20)
    """

    def __init__(
        self,
        rate_process: OrnsteinUhlenbeckRateProcess,
        return_rates: bool = False,
    ):
        self.rate_process = rate_process
        self.return_rates = return_rates
        self.dt = rate_process.dt
        self.n_neurons = rate_process.n_neurons
        self.chunk_size = rate_process.chunk_size

    def __len__(self) -> int:
        """Inherit length from rate process (effectively infinite)."""
        return len(self.rate_process)

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate inhomogeneous Poisson spikes.

        Args:
            idx: Index (passed to rate process).

        Returns:
            If return_rates=False:
                Binary spike tensor of shape (chunk_size, n_neurons).
            If return_rates=True:
                Tuple of (spikes, rates, weights) where:
                - spikes: shape (chunk_size, n_neurons) - binary spike tensor
                - rates: shape (chunk_size, n_neurons) - firing rates in Hz
                - weights: shape (chunk_size, n_patterns) - normalized mixing weights
        """
        # Get time-varying rates from process
        rate_output = self.rate_process[idx]

        if self.return_rates:
            # Unpack diagnostics from rate process
            rates, weights = rate_output
        else:
            rates = rate_output

        # Convert rates to probabilities: p = rate * dt / 1000 (dt in ms, rate in Hz)
        spike_probs = rates * self.dt / 1000.0

        # Sample spikes
        spikes = torch.rand_like(spike_probs) < spike_probs

        if self.return_rates:
            return spikes, rates, weights
        else:
            return spikes


class InhomogeneousPoissonSpikeDataLoader:
    """
    DataLoader for inhomogeneous Poisson spike generation.

    Wraps an InhomogeneousPoissonSpikeDataset and provides batched iteration
    with appropriate collation. Generates batches of spike trains where temporal
    dynamics vary according to the underlying rate process.

    Args:
        rate_process: OrnsteinUhlenbeckRateProcess that generates time-varying firing rates.
        batch_size: Number of independent samples per batch.
        device: Device for tensor operations.
        return_rates: If True, iterator yields tuple (spikes, rates, weights)
            for diagnostic/visualization purposes. If False, yields only spikes. Default: False.
            Note: This requires the underlying rate_process to have return_rates=True.

    Attributes:
        dataset: The underlying InhomogeneousPoissonSpikeDataset.
        batch_size: Number of samples per batch.
        dt: Timestep in milliseconds.
        n_neurons: Number of neurons.
        chunk_size: Chunk size per sample.

    Example:
        >>> from src.network_inputs.rate_processes import OrnsteinUhlenbeckRateProcess
        >>>
        >>> rate_process = OrnsteinUhlenbeckRateProcess(
        ...     patterns=patterns,  # (20, 5000)
        ...     chunk_size=1000,
        ...     dt=0.1,
        ...     tau=50.0,
        ...     r_mean=1.0,
        ...     sigma=0.5
        ... )
        >>>
        >>> dataloader = InhomogeneousPoissonSpikeDataLoader(
        ...     rate_process=rate_process,
        ...     batch_size=32,
        ...     device="cuda"
        ... )
        >>>
        >>> for spikes in dataloader:
        ...     # spikes.shape: (32, 1000, 5000)
        ...     # Each sample has independent OU-driven temporal dynamics
        ...     break
        >>>
        >>> # With diagnostics enabled
        >>> rate_process_diag = OrnsteinUhlenbeckRateProcess(
        ...     patterns=patterns,
        ...     chunk_size=1000,
        ...     dt=0.1,
        ...     tau=50.0,
        ...     return_rates=True
        ... )
        >>> dataloader_diag = InhomogeneousPoissonSpikeDataLoader(
        ...     rate_process=rate_process_diag,
        ...     batch_size=32,
        ...     device="cuda",
        ...     return_rates=True
        ... )
        >>> for spikes, rates, weights in dataloader_diag:
        ...     # spikes: (32, 1000, 5000), rates: (32, 1000, 5000)
        ...     # weights: (32, 1000, 20)
        ...     break
    """

    def __init__(
        self,
        rate_process: OrnsteinUhlenbeckRateProcess,
        batch_size: int,
        device: Union[str, torch.device] = "cpu",
        return_rates: bool = False,
    ):
        self.dataset = InhomogeneousPoissonSpikeDataset(
            rate_process=rate_process, return_rates=return_rates
        )
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.return_rates = return_rates

        # Expose useful attributes
        self.dt = self.dataset.dt
        self.n_neurons = self.dataset.n_neurons
        self.chunk_size = self.dataset.chunk_size

    def __iter__(self):
        """Infinite iterator over batches."""
        idx = 0
        while True:
            if self.return_rates:
                # Collect diagnostics
                spikes_batch = []
                rates_batch = []
                weights_batch = []

                for _ in range(self.batch_size):
                    spikes, rates, weights = self.dataset[idx]
                    spikes_batch.append(spikes)
                    rates_batch.append(rates)
                    weights_batch.append(weights)
                    idx += 1

                # Stack into (batch_size, chunk_size, n_neurons) and (batch_size, chunk_size, n_patterns)
                spikes_tensor = torch.stack(spikes_batch, dim=0).to(self.device)
                rates_tensor = torch.stack(rates_batch, dim=0).to(self.device)
                weights_tensor = torch.stack(weights_batch, dim=0).to(self.device)

                yield spikes_tensor, rates_tensor, weights_tensor
            else:
                # Only collect spikes
                batch = []
                for _ in range(self.batch_size):
                    spikes = self.dataset[idx]
                    batch.append(spikes)
                    idx += 1

                # Stack into (batch_size, chunk_size, n_neurons)
                batch_tensor = torch.stack(batch, dim=0).to(self.device)
                yield batch_tensor

    def __len__(self) -> int:
        """Return arbitrary large number for effectively infinite iteration."""
        return int(1e9)

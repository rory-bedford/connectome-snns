"""
PyTorch DataLoaders for generating input spike trains.

This module provides DataLoader implementations for generating various types
of spike train inputs for spiking neural network simulations, compatible with
PyTorch's training pipeline.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Union


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

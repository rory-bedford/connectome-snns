"""
PyTorch DataLoaders for generating input spike trains.

This module provides DataLoader implementations for generating various types
of spike train inputs for spiking neural network simulations, compatible with
PyTorch's training pipeline.
"""

import torch
import numpy as np
import zarr
from pathlib import Path
from torch.utils.data import Dataset, Sampler
from typing import Union, Tuple, Iterator
from analysis.firing_rate import compute_firing_rates_from_zarr


class PoissonSpikeDataset(Dataset):
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
        device (Union[str, torch.device]): Device to generate spikes on. Default: "cpu".

    Example:
        >>> # Single pattern (homogeneous)
        >>> firing_rates = np.array([10.0, 10.0, 20.0, 20.0])
        >>> dataset = PoissonSpikeDataset(firing_rates, chunk_size=100.0, dt=1.0)
        >>> spikes, pattern_idx = dataset[0]  # pattern_idx always 0
        >>>
        >>> # Multiple patterns (e.g., odours)
        >>> firing_rates = np.random.uniform(5, 20, (10, 100))
        >>> dataset = PoissonSpikeDataset(firing_rates, chunk_size=100.0, dt=1.0)
        >>> spikes, pattern_idx = dataset[15]  # pattern_idx = 15 % 10 = 5
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
            firing_rates = torch.from_numpy(firing_rates).float().to(device)
        elif isinstance(firing_rates, torch.Tensor):
            firing_rates = firing_rates.float().to(device)
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


def generate_odour_firing_rates(
    n_input_neurons: int,
    input_source_indices: Union[np.ndarray, torch.Tensor],
    cell_type_names: list[str],
    odour_configs: dict,
    n_patterns: int,
) -> np.ndarray:
    """
    Generate firing rate patterns for odour-modulated Poisson inputs.

    Creates a firing rate matrix where each pattern (odour) has different modulation
    of cell firing rates. A fraction of cells are modulated up/down from baseline.

    Args:
        n_input_neurons: Total number of input neurons.
        input_source_indices: Array indicating cell type for each input neuron.
        cell_type_names: List of cell type names (e.g., ["mitral"]).
        odour_configs: Dict mapping cell type names to OdourInputConfig objects.
        n_patterns: Number of distinct odour patterns to generate.

    Returns:
        Firing rates array of shape (n_patterns, n_input_neurons) in Hz.

    Example:
        >>> from parameter_loaders.fitting_activity import OdourInputConfig
        >>> odour_config = OdourInputConfig(
        ...     baseline_rate=6.0,
        ...     modulation_rate=1.0,
        ...     modulation_fraction=0.1
        ... )
        >>> firing_rates = generate_odour_firing_rates(
        ...     n_input_neurons=1000,
        ...     input_source_indices=np.zeros(1000, dtype=int),
        ...     cell_type_names=["mitral"],
        ...     odour_configs={"mitral": odour_config},
        ...     n_patterns=10
        ... )
        >>> firing_rates.shape
        (10, 1000)
    """
    # Convert to numpy if needed
    if isinstance(input_source_indices, torch.Tensor):
        input_source_indices = input_source_indices.cpu().numpy()

    # Initialize firing rates: (n_patterns, n_input_neurons)
    firing_rates = np.zeros((n_patterns, n_input_neurons))

    for ct_idx, ct_name in enumerate(cell_type_names):
        # Get mask for this cell type
        mask = input_source_indices == ct_idx

        if ct_name not in odour_configs:
            raise ValueError(f"No odour configuration found for cell type '{ct_name}'")

        odour_config = odour_configs[ct_name]

        # Get modulation parameters
        baseline_rate = odour_config.baseline_rate
        up_rate, down_rate = odour_config.get_modulated_rates()

        # Number of neurons of this cell type
        n_neurons_this_type = np.sum(mask)
        n_modulated = odour_config.get_n_modulated(n_neurons_this_type)

        # Get indices of neurons of this cell type
        neuron_indices = np.where(mask)[0]

        # For each pattern, randomly select which neurons are modulated
        for pattern_idx in range(n_patterns):
            # Shuffle neuron indices for this pattern
            shuffled_indices = np.random.permutation(neuron_indices)

            # Assign modulated up neurons
            up_indices = shuffled_indices[:n_modulated]
            firing_rates[pattern_idx, up_indices] = up_rate

            # Assign modulated down neurons
            down_indices = shuffled_indices[n_modulated : 2 * n_modulated]
            firing_rates[pattern_idx, down_indices] = down_rate

            # Assign baseline to remaining neurons
            baseline_indices = shuffled_indices[2 * n_modulated :]
            firing_rates[pattern_idx, baseline_indices] = baseline_rate

    return firing_rates


def generate_baseline_firing_rates(
    n_input_neurons: int,
    input_source_indices: Union[np.ndarray, torch.Tensor],
    cell_type_names: list[str],
    odour_configs: dict,
) -> np.ndarray:
    """
    Generate constant baseline firing rates (no modulation).

    Creates a single firing rate pattern where all cells fire at their baseline rate.
    This serves as a control condition with no odour-specific modulation.

    Args:
        n_input_neurons: Total number of input neurons.
        input_source_indices: Array indicating cell type for each input neuron.
        cell_type_names: List of cell type names (e.g., ["mitral"]).
        odour_configs: Dict mapping cell type names to OdourInputConfig objects.

    Returns:
        Firing rates array of shape (1, n_input_neurons) in Hz.
        Single pattern with all neurons at baseline rate.

    Example:
        >>> from parameter_loaders.fitting_activity import OdourInputConfig
        >>> odour_config = OdourInputConfig(
        ...     baseline_rate=6.0,
        ...     modulation_rate=1.0,
        ...     modulation_fraction=0.1
        ... )
        >>> firing_rates = generate_baseline_firing_rates(
        ...     n_input_neurons=1000,
        ...     input_source_indices=np.zeros(1000, dtype=int),
        ...     cell_type_names=["mitral"],
        ...     odour_configs={"mitral": odour_config}
        ... )
        >>> firing_rates.shape
        (1, 1000)
    """
    # Convert to numpy if needed
    if isinstance(input_source_indices, torch.Tensor):
        input_source_indices = input_source_indices.cpu().numpy()

    # Initialize firing rates: single pattern (1, n_input_neurons)
    firing_rates = np.zeros((1, n_input_neurons))

    for ct_idx, ct_name in enumerate(cell_type_names):
        # Get mask for this cell type
        mask = input_source_indices == ct_idx

        if ct_name not in odour_configs:
            raise ValueError(f"No odour configuration found for cell type '{ct_name}'")

        odour_config = odour_configs[ct_name]

        # Get baseline rate only (no modulation)
        baseline_rate = odour_config.baseline_rate

        # Set all neurons of this type to baseline
        firing_rates[0, mask] = baseline_rate

    return firing_rates


def collate_pattern_batches(batch):
    """
    Collate function for organizing pattern data with batch dimension.

    The DataLoader fetches batch_size * n_patterns items. This function organizes them
    into (batch_size, n_patterns, n_steps, n_neurons) where batch_size represents
    independent repeats/trials of all patterns.

    Args:
        batch: List of (spikes, pattern_idx) tuples from PoissonSpikeDataset.
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


class PrecomputedSpikeDataset(Dataset):
    """
    Dataset for loading pre-generated spike trains from disk in chunks.

    This dataset loads spike data stored in Zarr format on disk, yielding chunks
    of a specified size. Designed to work with spike data saved in the format:
    (batch_size, n_patterns, total_time, n_neurons).

    Returns both input spikes and target (output) spikes for each chunk.

    At initialization, computes the firing rates for each neuron in each pattern
    and batch using the firing_rate analysis module.

    Args:
        spike_data_path (Path): Path to zarr directory containing spike data.
        chunk_size (int): Number of timesteps per chunk.
        input_dataset_name (str): Name of input spike dataset within zarr group. Default: "input_spikes".
        target_dataset_name (str): Name of target spike dataset within zarr group. Default: "output_spikes".
        device (Union[str, torch.device]): Device to load data to. Default: "cpu".

    Attributes:
        dt (float): Timestep in milliseconds, loaded from zarr attributes.
        input_firing_rates (torch.Tensor): Input firing rates in Hz with shape (batch_size, n_patterns, n_input_neurons).
        target_firing_rates (torch.Tensor): Target firing rates in Hz with shape (batch_size, n_patterns, n_neurons).

    Example:
        >>> dataset = PrecomputedSpikeDataset(
        ...     spike_data_path=Path("results/spike_data.zarr"),
        ...     chunk_size=100,
        ...     device="cuda"
        ... )
        >>> input_spikes, target_spikes = dataset[0]  # First chunk
        >>> input_spikes.shape  # (batch_size, n_patterns, chunk_size, n_input_neurons)
        >>> target_spikes.shape  # (batch_size, n_patterns, chunk_size, n_neurons)
        >>> dataset.dt  # Timestep in ms, loaded from zarr
        >>> dataset.input_firing_rates.shape  # (batch_size, n_patterns, n_input_neurons)
        >>> dataset.target_firing_rates.shape  # (batch_size, n_patterns, n_neurons)
    """

    def __init__(
        self,
        spike_data_path: Union[Path, str],
        chunk_size: int,
        input_dataset_name: str = "input_spikes",
        target_dataset_name: str = "output_spikes",
        device: Union[str, torch.device] = "cpu",
    ):
        self.spike_data_path = Path(spike_data_path)
        self.chunk_size = chunk_size
        self.device = device

        # Open zarr group (read-only)
        root = zarr.open_group(self.spike_data_path, mode="r")

        # Load dt from zarr attributes
        if "dt" not in root.attrs:
            raise ValueError(
                f"Zarr file at {spike_data_path} does not contain 'dt' attribute. "
                "Ensure the data was generated with a version that saves dt to zarr."
            )
        self.dt = float(root.attrs["dt"])

        self.input_spike_data = root[input_dataset_name]
        self.target_spike_data = root[target_dataset_name]

        # Shape: (batch_size, n_patterns, total_time, n_neurons)
        self.batch_size, self.n_patterns, self.total_time, self.n_input_neurons = (
            self.input_spike_data.shape
        )
        _, _, _, self.n_neurons = self.target_spike_data.shape

        self.num_chunks = self.total_time // chunk_size

        if self.total_time % chunk_size != 0:
            print(
                f"Warning: total_time ({self.total_time}) not divisible by chunk_size ({chunk_size})"
            )

        # Compute firing rates for input spikes
        print("Computing input firing rates from spike data...")
        input_firing_rates_np = compute_firing_rates_from_zarr(
            zarr_path=self.spike_data_path,
            dataset_name=input_dataset_name,
            dt=self.dt,
        )
        self.input_firing_rates = (
            torch.from_numpy(input_firing_rates_np).float().to(device)
        )
        print(
            f"✓ Computed input firing rates with shape {self.input_firing_rates.shape}"
        )

        # Compute firing rates for target spikes
        print("Computing target firing rates from spike data...")
        target_firing_rates_np = compute_firing_rates_from_zarr(
            zarr_path=self.spike_data_path,
            dataset_name=target_dataset_name,
            dt=self.dt,
        )
        self.target_firing_rates = (
            torch.from_numpy(target_firing_rates_np).float().to(device)
        )
        print(
            f"✓ Computed target firing rates with shape {self.target_firing_rates.shape}"
        )

    def __len__(self) -> int:
        """Total number of chunks."""
        return self.num_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a full chunk across all batches and patterns.

        Args:
            idx (int): Chunk index (0 to num_chunks-1).

        Returns:
            Tuple of (input_spike_chunk, target_spike_chunk) where:
                - input_spike_chunk: torch.Tensor of shape (batch_size, n_patterns, chunk_size, n_input_neurons)
                - target_spike_chunk: torch.Tensor of shape (batch_size, n_patterns, chunk_size, n_neurons)
        """
        # Extract chunk from zarr - all batches and patterns for this time slice
        start_t = idx * self.chunk_size
        end_t = start_t + self.chunk_size

        input_chunk = self.input_spike_data[:, :, start_t:end_t, :]
        target_chunk = self.target_spike_data[:, :, start_t:end_t, :]

        # Convert to torch tensors as bool (zarr storage format)
        input_tensor = torch.from_numpy(input_chunk).bool().to(self.device)
        target_tensor = torch.from_numpy(target_chunk).bool().to(self.device)

        return input_tensor, target_tensor


class CyclicSampler(Sampler):
    """
    Sampler that cycles through indices infinitely.

    This sampler repeats the dataset indices indefinitely, allowing
    multi-epoch training without exhausting the dataloader iterator.

    Args:
        data_source: Dataset to sample from

    Example:
        >>> dataset = PrecomputedSpikeDataset(...)
        >>> sampler = CyclicSampler(dataset)
        >>> dataloader = DataLoader(dataset, sampler=sampler, batch_size=None)
        >>> # Dataloader will never run out of data
    """

    def __init__(self, data_source: Dataset):
        self.data_source = data_source
        self.num_samples = len(data_source)

    def __iter__(self) -> Iterator[int]:
        """Yield indices cycling through the dataset infinitely."""
        while True:
            yield from range(self.num_samples)

    def __len__(self) -> int:
        """Return a very large number to indicate indefinite iteration."""
        return 2**31 - 1  # Effectively infinite

"""
Supervised learning dataloaders for pre-generated spike trains.

This module provides DataLoader implementations for loading spike data from disk,
typically used for supervised learning scenarios where both input and target (output)
spike trains have been pre-computed and stored.

The datasets handle paired input-target data stored in Zarr format, designed for
training student networks to match teacher network outputs.
"""

import torch
import zarr
from pathlib import Path
from torch.utils.data import Dataset, Sampler
from typing import Union, Tuple, Iterator


class PrecomputedSpikeDataset(Dataset):
    """
    Dataset for loading pre-generated spike trains from disk in chunks.

    This dataset loads spike data stored in Zarr format on disk, yielding chunks
    of a specified size. Designed to work with spike data saved in the format:
    (batch_size, total_time, n_neurons).

    Returns both input spikes and target (output) spikes for each chunk.

    Args:
        spike_data_path (Path): Path to zarr directory containing spike data.
        chunk_size (int): Number of timesteps per chunk.
        input_dataset_name (str): Name of input spike dataset within zarr group. Default: "input_spikes".
        target_dataset_name (str): Name of target spike dataset within zarr group. Default: "output_spikes".
        device (Union[str, torch.device]): Device to load data to. Default: "cpu".

    Attributes:
        dt (float): Timestep in milliseconds, loaded from zarr attributes.
        batch_size (int): Number of samples in the batch dimension.
        total_time (int): Total number of timesteps in the full spike data.
        n_input_neurons (int): Number of input neurons.
        n_neurons (int): Number of target/output neurons.

    Example:
        >>> dataset = PrecomputedSpikeDataset(
        ...     spike_data_path=Path("results/spike_data.zarr"),
        ...     chunk_size=100,
        ...     device="cuda"
        ... )
        >>> input_spikes, target_spikes = dataset[0]  # First chunk
        >>> input_spikes.shape  # (batch_size, chunk_size, n_input_neurons)
        >>> target_spikes.shape  # (batch_size, chunk_size, n_neurons)
        >>> dataset.dt  # Timestep in ms, loaded from zarr
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

        # Shape: (batch_size, total_time, n_neurons)
        self.batch_size, self.total_time, self.n_input_neurons = (
            self.input_spike_data.shape
        )
        _, _, self.n_neurons = self.target_spike_data.shape

        self.num_chunks = self.total_time // chunk_size

        if self.total_time % chunk_size != 0:
            print(
                f"Warning: total_time ({self.total_time}) not divisible by chunk_size ({chunk_size})"
            )

    def __len__(self) -> int:
        """Total number of chunks."""
        return self.num_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a chunk from the dataset.

        Args:
            idx (int): Chunk index (0 to num_chunks-1).

        Returns:
            Tuple of (input_spike_chunk, target_spike_chunk) where:
                - input_spike_chunk: torch.Tensor of shape (batch_size, chunk_size, n_input_neurons)
                - target_spike_chunk: torch.Tensor of shape (batch_size, chunk_size, n_neurons)
        """
        # Extract chunk from zarr - all batches for this time slice
        start_t = idx * self.chunk_size
        end_t = start_t + self.chunk_size

        input_chunk = self.input_spike_data[:, start_t:end_t, :]
        target_chunk = self.target_spike_data[:, start_t:end_t, :]

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

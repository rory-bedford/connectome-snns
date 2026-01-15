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
from collections import namedtuple
from torch.utils.data import Dataset, Sampler
from typing import Union, Iterator

# Named tuple for supervised spike data
SpikeData = namedtuple("SpikeData", ["input_spikes", "target_spikes"])


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

    def __getitem__(self, idx: int) -> SpikeData:
        """
        Get a chunk from the dataset.

        Args:
            idx (int): Chunk index (0 to num_chunks-1).

        Returns:
            SpikeData named tuple with fields:
                - input_spikes: torch.Tensor of shape (batch_size, chunk_size, n_input_neurons)
                - target_spikes: torch.Tensor of shape (batch_size, chunk_size, n_neurons)
        """
        # Extract chunk from zarr - all batches for this time slice
        start_t = idx * self.chunk_size
        end_t = start_t + self.chunk_size

        input_chunk = self.input_spike_data[:, start_t:end_t, :]
        target_chunk = self.target_spike_data[:, start_t:end_t, :]

        # Convert to torch tensors as bool (zarr storage format)
        input_tensor = torch.from_numpy(input_chunk).bool().to(self.device)
        target_tensor = torch.from_numpy(target_chunk).bool().to(self.device)

        return SpikeData(input_spikes=input_tensor, target_spikes=target_tensor)


class PoissonInputSpikeDataset(Dataset):
    """
    Dataset that replaces feedforward inputs with Poisson spikes while preserving recurrent targets.

    This dataset is designed for training feedforward-unraveled networks where:
    - The original feedforward (external) inputs are replaced with Poisson spike trains
    - The recurrent neuron spikes (teacher activity) are loaded exactly from disk
    - Both are concatenated to form the model input
    - The recurrent spikes also serve as the training target

    The Poisson firing rate is computed as the average rate across all feedforward neurons
    and all time points in the dataset. This rate is computed once during initialization
    by scanning through the stored spike data.

    This enables testing whether the network can learn from statistically similar but
    not identical input patterns, separating the role of exact spike timing in the
    feedforward inputs from the learning of recurrent dynamics.

    Args:
        spike_data_path (Path): Path to zarr directory containing spike data.
            Must contain 'input_spikes' (feedforward) and 'output_spikes' (recurrent).
        chunk_size (int): Number of timesteps per chunk.
        device (Union[str, torch.device]): Device to load data to. Default: "cpu".
        firing_rate_override (float, optional): If provided, use this firing rate (Hz)
            instead of computing from the data. Useful for experiments varying the rate.

    Attributes:
        dt (float): Timestep in milliseconds, loaded from zarr attributes.
        batch_size (int): Number of samples in the batch dimension.
        total_time (int): Total number of timesteps in the full spike data.
        n_input_neurons (int): Number of feedforward input neurons.
        n_neurons (int): Number of recurrent/target neurons.
        avg_firing_rate (float): Average firing rate (Hz) used for Poisson generation.
        num_chunks (int): Total number of chunks available.

    Example:
        >>> dataset = PoissonInputSpikeDataset(
        ...     spike_data_path=Path("results/spike_data.zarr"),
        ...     chunk_size=1000,
        ...     device="cuda"
        ... )
        >>> print(f"Using Poisson rate: {dataset.avg_firing_rate:.2f} Hz")
        >>>
        >>> # Each access generates fresh Poisson spikes
        >>> batch = dataset[0]
        >>> batch.input_spikes.shape  # (batch_size, chunk_size, n_ff + n_rec)
        >>> batch.target_spikes.shape  # (batch_size, chunk_size, n_rec)
        >>>
        >>> # Use with DataLoader
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     batch_size=None,
        ...     sampler=CyclicSampler(dataset),
        ... )
    """

    def __init__(
        self,
        spike_data_path: Union[Path, str],
        chunk_size: int,
        device: Union[str, torch.device] = "cpu",
        firing_rate_override: float = None,
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

        self.input_spike_data = root["input_spikes"]
        self.target_spike_data = root["output_spikes"]

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

        # Compute or use provided firing rate
        if firing_rate_override is not None:
            self.avg_firing_rate = firing_rate_override
        else:
            self.avg_firing_rate = self._compute_average_firing_rate()

        # Pre-compute spike probability for Poisson generation
        # p = rate (Hz) * dt (ms) * 1e-3 (ms to s conversion)
        self.spike_prob = self.avg_firing_rate * self.dt * 1e-3

    def _compute_average_firing_rate(self) -> float:
        """
        Compute the average firing rate across all feedforward neurons and time.

        Scans through the entire input spike dataset to compute the mean firing rate.
        This is done once during initialization.

        Returns:
            float: Average firing rate in Hz.
        """
        # Load all input spikes (this may use significant memory for large datasets)
        # For very large datasets, could implement chunked computation
        all_input_spikes = self.input_spike_data[:]  # (batch, time, n_ff)

        # Count total spikes and compute rate
        total_spikes = all_input_spikes.sum()
        total_neuron_time_ms = (
            self.batch_size * self.total_time * self.n_input_neurons * self.dt
        )
        total_neuron_time_s = total_neuron_time_ms / 1000.0

        avg_rate = total_spikes / total_neuron_time_s

        return float(avg_rate)

    def __len__(self) -> int:
        """Total number of chunks."""
        return self.num_chunks

    def __getitem__(self, idx: int) -> SpikeData:
        """
        Get a chunk with Poisson-generated feedforward inputs and exact recurrent spikes.

        For each access, fresh Poisson spikes are generated for the feedforward inputs,
        while the recurrent spikes are loaded exactly from disk. Both are concatenated
        to form the model input.

        Args:
            idx (int): Chunk index (0 to num_chunks-1).

        Returns:
            SpikeData named tuple with fields:
                - input_spikes: torch.Tensor of shape (batch_size, chunk_size, n_ff + n_rec)
                    Concatenation of [Poisson FF spikes, exact recurrent spikes]
                - target_spikes: torch.Tensor of shape (batch_size, chunk_size, n_rec)
                    Exact recurrent spikes (same as second half of input_spikes)
        """
        # Extract recurrent spikes from zarr
        start_t = idx * self.chunk_size
        end_t = start_t + self.chunk_size

        target_chunk = self.target_spike_data[:, start_t:end_t, :]

        # Convert to torch tensor
        target_tensor = torch.from_numpy(target_chunk).bool().to(self.device)

        # Generate fresh Poisson spikes for feedforward inputs
        # Shape: (batch_size, chunk_size, n_input_neurons)
        random_vals = torch.rand(
            self.batch_size, self.chunk_size, self.n_input_neurons, device=self.device
        )
        poisson_spikes = random_vals < self.spike_prob

        # Concatenate: [Poisson FF, exact recurrent] -> model input
        # Shape: (batch_size, chunk_size, n_ff + n_rec)
        concatenated_inputs = torch.cat([poisson_spikes, target_tensor], dim=2)

        return SpikeData(input_spikes=concatenated_inputs, target_spikes=target_tensor)


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


def feedforward_collate_fn(batch: SpikeData) -> SpikeData:
    """
    Custom collate function for feedforward training on all neurons.

    Transforms a recurrent network into a feedforward architecture by concatenating
    feedforward and recurrent spikes as inputs. The recurrent spikes (teacher activity)
    become additional known inputs, allowing the network to be trained with feedforward
    dynamics only.

    This is used when "unraveling" a recurrent network: instead of neurons receiving
    recurrent input from other neurons in the network, they receive the pre-recorded
    teacher spike trains as additional feedforward inputs.

    Args:
        batch: SpikeData named tuple from PrecomputedSpikeDataset
            input_spikes: (batch, time, n_feedforward) - feedforward neuron spikes
            target_spikes: (batch, time, n_neurons) - recurrent neuron spikes (teacher)

    Returns:
        SpikeData named tuple with fields:
            input_spikes: (batch, time, n_feedforward + n_neurons) - concatenated inputs
            target_spikes: (batch, time, n_neurons) - unchanged recurrent targets

    Example:
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     batch_size=None,
        ...     collate_fn=feedforward_collate_fn
        ... )
        >>> for batch_data in dataloader:
        ...     # batch_data.input_spikes.shape: (batch, time, n_ff + n_rec)
        ...     # batch_data.target_spikes.shape: (batch, time, n_rec)
    """
    # Concatenate feedforward and recurrent spikes along neuron dimension
    # Shape: (batch, time, n_feedforward + n_neurons)
    concatenated_inputs = torch.cat([batch.input_spikes, batch.target_spikes], dim=2)

    # Keep all neurons as target
    # Shape: (batch, time, n_neurons)
    return SpikeData(
        input_spikes=concatenated_inputs, target_spikes=batch.target_spikes
    )


def single_neuron_collate_fn(
    batch: SpikeData,
) -> SpikeData:
    """
    Custom collate function for single-neuron training.

    Concatenates feedforward and recurrent spikes as inputs, extracts neuron 0 as target.
    Designed for training a single neuron with feedforward-only dynamics where all recurrent
    neurons become additional inputs alongside feedforward neurons.

    Args:
        batch: SpikeData named tuple from PrecomputedSpikeDataset
            input_spikes: (batch, time, n_feedforward) - feedforward neuron spikes
            target_spikes: (batch, time, n_neurons) - recurrent neuron spikes

    Returns:
        SpikeData named tuple with fields:
            input_spikes: (batch, time, n_feedforward + n_neurons)
            target_spikes: (batch, time, 1)

    Example:
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     batch_size=None,
        ...     collate_fn=single_neuron_collate_fn
        ... )
        >>> for batch_data in dataloader:
        ...     # batch_data.input_spikes.shape: (batch, time, n_total_inputs)
        ...     # batch_data.target_spikes.shape: (batch, time, 1)
    """
    # Concatenate feedforward and recurrent spikes along neuron dimension
    # Shape: (batch, time, n_feedforward + n_neurons)
    concatenated_inputs = torch.cat([batch.input_spikes, batch.target_spikes], dim=2)

    # Extract only neuron 0 as target
    # Shape: (batch, time, 1)
    single_neuron_target = batch.target_spikes[:, :, 0:1]

    return SpikeData(
        input_spikes=concatenated_inputs, target_spikes=single_neuron_target
    )

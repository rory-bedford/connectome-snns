"""Dataset that loads spike probabilities and Bernoulli-samples spikes.

Designed for real data where deconvolved calcium traces have been converted
to spike probabilities (values in [0, 1] per timestep per neuron).

Feedforward inputs are resampled via Bernoulli on every access (stochastic
input). Target spikes are sampled once at initialisation and cached for
stable supervision.

Expected zarr layout (two separate zarr groups)::

    ff_probabilities.zarr/
        attrs: dt (float, ms)
        probabilities: (batch, time, n_ff)   float16/float32

    target_probabilities.zarr/
        attrs: dt (float, ms)
        probabilities: (batch, time, n_rec)  float16/float32
"""

import numpy as np
import torch
import zarr
from pathlib import Path
from torch.utils.data import Dataset
from typing import Union

from ._base import SpikeData


class ProbabilisticSpikeDataset(Dataset):
    """Load spike probabilities from zarr; Bernoulli-sample spikes on access.

    Args:
        ff_zarr_path: Path to zarr group containing feedforward probabilities
            (dataset ``probabilities`` with shape ``(batch, time, n_ff)``).
        target_zarr_path: Path to zarr group containing target probabilities
            (dataset ``probabilities`` with shape ``(batch, time, n_rec)``).
        chunk_size: Number of timesteps per chunk.
        device: Torch device for returned tensors.
        dataset_name: Name of the probability dataset within each zarr group.
    """

    def __init__(
        self,
        ff_zarr_path: Union[Path, str],
        target_zarr_path: Union[Path, str],
        chunk_size: int,
        device: Union[str, torch.device] = "cpu",
        dataset_name: str = "probabilities",
    ):
        self.chunk_size = chunk_size
        self.device = device

        # FF probabilities — kept as zarr, read per-chunk
        ff_root = zarr.open_group(Path(ff_zarr_path), mode="r")
        if "dt" not in ff_root.attrs:
            raise ValueError(f"Zarr at {ff_zarr_path} missing 'dt' attribute.")
        self.dt = float(ff_root.attrs["dt"])

        self._ff_probs = ff_root[dataset_name]
        self.batch_size, self.total_time, self.n_ff = self._ff_probs.shape

        # Target probabilities — sample once, cache as bool tensor
        target_root = zarr.open_group(Path(target_zarr_path), mode="r")
        target_dt = float(target_root.attrs.get("dt", self.dt))
        if target_dt != self.dt:
            raise ValueError(f"FF dt ({self.dt}) != target dt ({target_dt})")

        target_probs_zarr = target_root[dataset_name]
        target_batch, target_time, self.n_neurons = target_probs_zarr.shape
        if target_time != self.total_time:
            raise ValueError(
                f"FF time ({self.total_time}) != target time ({target_time})"
            )
        if target_batch != self.batch_size:
            raise ValueError(
                f"FF batch ({self.batch_size}) != target batch ({target_batch})"
            )

        self.num_chunks = self.total_time // chunk_size
        if self.total_time % chunk_size != 0:
            print(
                f"Warning: total_time ({self.total_time}) not divisible by "
                f"chunk_size ({chunk_size}), last "
                f"{self.total_time % chunk_size} timesteps dropped."
            )

        # Target probabilities — kept as zarr, read per-chunk
        self._target_probs = target_probs_zarr

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> SpikeData:
        start_t = idx * self.chunk_size
        end_t = start_t + self.chunk_size

        # FF: Bernoulli-sample fresh each access
        ff_probs = torch.from_numpy(
            np.array(self._ff_probs[:, start_t:end_t, :], dtype=np.float32)
        )
        ff_spikes = torch.rand_like(ff_probs) < ff_probs

        # Targets: Bernoulli-sample fresh each access
        target_probs = torch.from_numpy(
            np.array(self._target_probs[:, start_t:end_t, :], dtype=np.float32)
        )
        target_spikes = torch.rand_like(target_probs) < target_probs

        return SpikeData(
            input_spikes=ff_spikes,
            target_spikes=target_spikes,
        )

"""Dataset that returns raw OU latent trajectories as feedforward input."""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union

from ._base import TeacherSpikeDataset


class LatentOUDataset(TeacherSpikeDataset):
    """Returns OU latent trajectories as **continuous floats** (not spikes).

    Unlike :class:`OUReconstructedFFDataset` which Poisson-samples binary
    spikes from the OU rates, this dataset returns the raw OU mixing weights
    directly.  The ``SpikeData.input_spikes`` field will therefore contain
    continuous real-valued activations (shape ``batch × time × n_latents``),
    not binary 0/1 spike trains.  The simulator treats these values exactly
    like spike counts in the conductance update (einsum), so they act as
    continuous rate inputs.

    The ``weights`` array in the zarr archive contains the time-varying OU
    mixing weights.  This dataset returns them directly as continuous floats,
    letting the network learn a linear mapping from latents to neurons.

    Args:
        spike_data_path: Path to ``spike_data.zarr`` with ``weights``
            and ``output_spikes``.
        chunk_size: Number of timesteps per chunk.
        device: Torch device for returned tensors.
        recurrent_smoothing_tau: Gaussian sigma (ms) for smoothing recurrent
            input spikes.  ``None`` disables smoothing.
    """

    def __init__(
        self,
        spike_data_path: Union[Path, str],
        chunk_size: int,
        device: Union[str, torch.device] = "cpu",
        recurrent_smoothing_tau: Optional[float] = None,
    ):
        super().__init__(
            spike_data_path=spike_data_path,
            chunk_size=chunk_size,
            device=device,
            recurrent_smoothing_tau=recurrent_smoothing_tau,
        )

        if "weights" not in self._root:
            raise ValueError(
                f"Zarr file at {spike_data_path} is missing 'weights'. "
                "Ensure the teacher activity was generated with OU latents."
            )

        self.weight_data = self._root["weights"]  # (batch, time, n_latents)
        self.n_latents = self.weight_data.shape[2]

        print(
            f"LatentOUDataset: {self.n_latents} OU latents (continuous)  |  "
            f"{self.n_neurons} rec neurons  |  "
            f"{self.num_chunks} chunks  |  dt={self.dt} ms"
        )

    def _get_ff_spikes(self, start_t: int, end_t: int) -> torch.Tensor:
        """Return continuous OU latent activations (NOT binary spikes).

        Despite the inherited method name, this returns float-valued OU mixing
        weights, not 0/1 spike trains.
        """
        w_chunk = self.weight_data[:, start_t:end_t, :]  # (batch, chunk, n_latents)
        return torch.from_numpy(np.array(w_chunk, dtype=np.float32))

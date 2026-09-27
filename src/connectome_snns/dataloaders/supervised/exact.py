"""Dataset that loads exact feedforward spikes from disk."""

import numpy as np
import torch
import zarr
from pathlib import Path
from typing import Optional, Union

from ._base import TeacherSpikeDataset
from ._smoothing import smooth_spike_trains


class ExactFFDataset(TeacherSpikeDataset):
    """Load both feedforward and recurrent spikes exactly from zarr.

    Optionally smooth the feedforward input spikes with a Gaussian kernel and
    Bernoulli-resample them at each chunk access. This degrades the precise
    timing information in the FF input, matching the inferring-inputs trick
    for forcing the student off a pure-FF solution. Target spikes stay exact.

    Args:
        spike_data_path: Path to ``spike_data.zarr``.
        chunk_size: Number of timesteps per chunk.
        input_dataset_name: Zarr dataset name for FF spikes.
        target_dataset_name: Zarr dataset name for recurrent spikes.
        device: Torch device for returned tensors.
        recurrent_smoothing_tau: Gaussian sigma (ms) for smoothing recurrent
            input spikes. ``None`` disables smoothing.
        ff_smoothing_tau: Gaussian sigma (ms) for smoothing FF input spikes.
            ``None`` disables smoothing.
    """

    def __init__(
        self,
        spike_data_path: Union[Path, str],
        chunk_size: int,
        input_dataset_name: str = "input_spikes",
        target_dataset_name: str = "output_spikes",
        device: Union[str, torch.device] = "cpu",
        recurrent_smoothing_tau: Optional[float] = None,
        ff_smoothing_tau: Optional[float] = None,
    ):
        super().__init__(
            spike_data_path=spike_data_path,
            chunk_size=chunk_size,
            device=device,
            recurrent_smoothing_tau=recurrent_smoothing_tau,
            target_dataset_name=target_dataset_name,
        )
        self.input_spike_data = self._root[input_dataset_name]
        _, _, self.n_input_neurons = self.input_spike_data.shape

        self.ff_smoothing_tau = ff_smoothing_tau
        self._ff_smoothed_rate_data: Optional[zarr.Array] = None
        self._input_dataset_name = input_dataset_name
        if self.ff_smoothing_tau:
            self._ensure_ff_smoothed_rates()

    def _ff_smoothed_rates_key(self) -> str:
        return f"smoothed_ff_rates_tau{self.ff_smoothing_tau}"

    def _ensure_ff_smoothed_rates(self) -> None:
        """Compute and cache smoothed FF-input rates in the zarr archive."""
        key = self._ff_smoothed_rates_key()

        try:
            root_rw = zarr.open_group(self.spike_data_path, mode="r+")
        except Exception:
            root_rw = zarr.open_group(self.spike_data_path, mode="a")

        if key in root_rw:
            cached = zarr.open_group(self.spike_data_path, mode="r")[key]
            # Validate: an interrupted write can leave the tail of the array
            # zero-filled, which silently kills downstream gradients. Compare
            # cached sum to the raw spike total (Gaussian kernel is
            # normalised to sum=1, so the totals must match).
            cached_sum = float(np.array(cached[:], dtype=np.float32).sum())
            raw_sum = float(np.array(self.input_spike_data[:], dtype=np.float32).sum())
            if cached_sum < 0.5 * raw_sum:
                print(
                    f"WARNING: cached FF smoothed rates ({key}) look truncated "
                    f"(sum={cached_sum:.2e} vs raw={raw_sum:.2e}); recomputing."
                )
                del root_rw[key]
            else:
                self._ff_smoothed_rate_data = cached
                print(
                    f"Loaded cached FF smoothed rates ({key}) from {self.spike_data_path}"
                )
                return

        print(
            f"Computing FF smoothed rates (tau={self.ff_smoothing_tau} ms) "
            f"for {self.spike_data_path} ..."
        )

        all_spikes = torch.from_numpy(
            np.array(self.input_spike_data[:], dtype=np.float32)
        )
        rates = smooth_spike_trains(all_spikes, self.dt, self.ff_smoothing_tau)

        rates_np = rates.numpy().astype(np.float16)
        root_rw.create_array(
            key,
            shape=rates_np.shape,
            dtype=rates_np.dtype,
            chunks=(self.batch_size, self.chunk_size, self.n_input_neurons),
            overwrite=True,
        )
        root_rw[key][:] = rates_np

        self._ff_smoothed_rate_data = zarr.open_group(self.spike_data_path, mode="r")[
            key
        ]
        print(f"Cached FF smoothed rates to {self.spike_data_path}/{key}")

    def _get_ff_spikes(self, start_t: int, end_t: int) -> torch.Tensor:
        if self._ff_smoothed_rate_data is not None:
            rates = torch.from_numpy(
                np.array(
                    self._ff_smoothed_rate_data[:, start_t:end_t, :],
                    dtype=np.float32,
                )
            )
            return torch.rand_like(rates) < rates
        chunk = self.input_spike_data[:, start_t:end_t, :]
        return torch.from_numpy(np.array(chunk)).bool()

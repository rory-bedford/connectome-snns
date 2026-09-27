"""Feedforward inputs Poisson-sampled from measured rates, with a calcium target."""

import numpy as np
import torch
import zarr
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset

from ._base import SpikeData


class RateCalciumFFDataset(Dataset):
    """Teacher-forced FF inputs from measured rates, fit in dF/F (calcium) space.

    Each chunk is one window of a single continuous recording. The feedforward
    input spikes are inhomogeneous-Poisson (Bernoulli-per-bin) samples of the
    measured per-cell firing rates, drawn fresh on every access; the supervision
    target is the measured dF/F for that window's imaging frames, carried as the
    first-class :attr:`SpikeData.target_calcium` for :class:`CalciumMSELoss`.

    The rates zarr provides ``rates (n_windows, n_cells, n_steps)`` in Hz on a
    ``dt_ms`` grid, with attrs ``dt_ms``, ``frame_rate_hz`` and
    ``batch_duration_s``. Each window is ``n_steps`` simulation steps =
    ``frames_per_chunk`` imaging frames; ``chunk_size = n_steps`` and
    ``num_chunks = n_windows``. There is a single continuous recording, so
    ``batch_size = 1`` and chunks must be consumed in order (use a non-shuffling
    / cyclic sampler) so the calcium model's latent state carries correctly.

    Args:
        rates_path: Path to the rates zarr group.
        calcium_frames: Measured dF/F, shape ``(n_cells, n_windows,
            frames_per_chunk)`` in the rates' canonical cell order. NaN marks
            invalid (masked) frames.
        rates_dataset_name: Name of the rates array inside the zarr group.
        device: Retained for interface parity; returned tensors are CPU (so
            ``num_workers > 0`` works) and the trainer moves them to device.
    """

    def __init__(
        self,
        rates_path: Union[Path, str],
        calcium_frames: np.ndarray,
        rates_dataset_name: str = "rates",
        device: Union[str, torch.device] = "cpu",
    ):
        self.rates_path = Path(rates_path)
        self.device = device

        root = zarr.open_group(self.rates_path, mode="r")
        for key in ("dt_ms", "frame_rate_hz", "batch_duration_s"):
            if key not in root.attrs:
                raise ValueError(f"rates zarr {rates_path} is missing attr '{key}'.")
        self.dt = float(root.attrs["dt_ms"])  # ms — the simulation timestep
        self.frame_rate_hz = float(root.attrs["frame_rate_hz"])
        self.batch_duration_s = float(root.attrs["batch_duration_s"])

        self._rates = root[rates_dataset_name]  # lazy (n_windows, n_cells, n_steps)
        self.num_chunks, self.n_neurons, self.chunk_size = self._rates.shape
        self.frames_per_chunk = int(round(self.batch_duration_s * self.frame_rate_hz))
        self.batch_size = 1

        calcium = torch.as_tensor(np.asarray(calcium_frames), dtype=torch.float32)
        expected = (self.n_neurons, self.num_chunks, self.frames_per_chunk)
        if tuple(calcium.shape) != expected:
            raise ValueError(
                f"calcium_frames shape {tuple(calcium.shape)} != expected {expected} "
                "(n_cells, n_windows, frames_per_chunk)."
            )
        self.calcium = calcium

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> SpikeData:
        # Measured rates for this window: (n_cells, n_steps) Hz -> (n_steps, n_cells).
        rate_hz = np.asarray(self._rates[idx], dtype=np.float32).T
        prob = torch.from_numpy(rate_hz) * (self.dt * 1e-3)  # spikes/bin = Hz * s
        input_spikes = (torch.rand_like(prob) < prob).unsqueeze(0)  # (1, T, N) bool

        # Measured dF/F target for this window: (frames, n_cells) -> (1, frames, N).
        target_calcium = self.calcium[:, idx, :].T.unsqueeze(0).clone()
        return SpikeData(input_spikes=input_spikes, target_calcium=target_calcium)


class ParallelSegmentCalciumDataset(Dataset):
    """Parallel-segment loader for one long recording, fit in calcium (dF/F) space.

    The recording is one 2-D array ``rates (n_cells, total_time)`` in Hz, chunked
    along time at ``chunk_size`` (i.e. storage chunks ``(n_cells, chunk_size)``) so
    reading a ``chunk_size`` time-slice of all cells is exactly one whole storage
    chunk. ``chunk_size`` is read from the array's storage chunk — a data property,
    not a training knob — and ``n_total_chunks = total_time // chunk_size``.

    For training the chunks are cut into ``n_segments`` equal contiguous streams
    stacked on a batch axis and advanced in lock-step: stream ``i`` covers chunks
    ``[i*M, (i+1)*M)`` with ``M = n_total_chunks // n_segments``, and
    ``__getitem__(t)`` returns the ``t``-th chunk of every stream — batch shape
    ``(n_segments, chunk_size, n_cells)``. Each stream carries its own state and is
    reset at the epoch (``M`` chunks) boundary; the ragged tail
    (``n_total_chunks % n_segments`` chunks) is dropped.

    Input spikes are inhomogeneous-Poisson (Bernoulli-per-bin) samples of the
    measured rates, drawn fresh per access; the target is the chunk's measured
    dF/F frames, carried as the first-class :attr:`SpikeData.target_calcium`. A
    per-stream warm-up is left to the trainer's ``burn_in_chunks``.

    :attr:`whole_chunk_reads` reports whether each read maps to exactly one storage
    chunk (cells in one chunk and the read width == the storage time-chunk), i.e.
    whether I/O is optimal.

    Args:
        rates_path: zarr group with ``rates (n_cells, total_time)`` and attrs
            ``dt_ms``, ``frame_rate_hz``.
        calcium: measured dF/F, shape ``(n_cells, n_total_chunks*frames_per_chunk)``
            in the rates' canonical cell order; NaN = invalid.
        n_segments: number of parallel streams (== batch size).
        rates_dataset_name: name of the rates array inside the zarr group.
        device: retained for parity; returned tensors are CPU (worker-safe).
    """

    def __init__(
        self,
        rates_path: Union[Path, str],
        calcium: np.ndarray,
        n_segments: int,
        rates_dataset_name: str = "rates",
        device: Union[str, torch.device] = "cpu",
        output_idx: np.ndarray | None = None,
        target_rate_override: np.ndarray | None = None,
    ):
        self.rates_path = Path(rates_path)
        self.device = device
        # Output (scored) cell subset: input spikes always cover every cell (full
        # presynaptic drive), but the targets are restricted to these columns so the
        # network can have an output layer smaller than the input. None = all cells.
        self.output_idx = None if output_idx is None else np.asarray(output_idx)
        # Optional self-consistent target: a precomputed ``target_rate`` (n_out,
        # total_time) used in place of the measured rate (the input spikes still come
        # from the measured rates). Lets a teacher-generated, definitely-realisable
        # output be the training target. None = use the measured rate.
        self.target_rate_override = (
            None if target_rate_override is None else np.asarray(target_rate_override)
        )

        root = zarr.open_group(self.rates_path, mode="r")
        for key in ("dt_ms", "frame_rate_hz"):
            if key not in root.attrs:
                raise ValueError(f"rates zarr {rates_path} is missing attr '{key}'.")
        self.dt = float(root.attrs["dt_ms"])  # ms — the simulation timestep
        self.frame_rate_hz = float(root.attrs["frame_rate_hz"])

        self._rates = root[rates_dataset_name]  # (n_cells, total_time)
        if self._rates.ndim != 2:
            raise ValueError(
                f"rates must be 2-D (n_cells, total_time), got shape "
                f"{tuple(self._rates.shape)}."
            )
        self.n_neurons, self.total_time = self._rates.shape
        self.storage_chunk = tuple(int(c) for c in self._rates.chunks)
        # chunk_size = the storage time-chunk (the unit of a whole-chunk read).
        self.chunk_size = self.storage_chunk[1]
        self.n_total_chunks = self.total_time // self.chunk_size

        steps_per_frame = int(round(1000.0 / (self.frame_rate_hz * self.dt)))
        if self.chunk_size % steps_per_frame != 0:
            raise ValueError(
                f"chunk_size {self.chunk_size} is not a multiple of the frame length "
                f"{steps_per_frame} steps (1/{self.frame_rate_hz}Hz at {self.dt}ms)."
            )
        self.frames_per_chunk = self.chunk_size // steps_per_frame

        # Optimal I/O iff each read = one storage chunk: all cells in one chunk and
        # the read width equals the storage time-chunk.
        self.whole_chunk_reads = (
            self.storage_chunk[0] == self.n_neurons
            and self.storage_chunk[1] == self.chunk_size
        )

        self.n_segments = int(n_segments)
        if not 1 <= self.n_segments <= self.n_total_chunks:
            raise ValueError(
                f"n_segments must be in [1, {self.n_total_chunks}], got {n_segments}."
            )
        self.stream_len = self.n_total_chunks // self.n_segments  # M (chunks/stream)
        self.n_used_chunks = self.n_segments * self.stream_len
        self.n_dropped = self.n_total_chunks - self.n_used_chunks
        self.num_chunks = self.stream_len  # chunks consumed per epoch
        self.batch_size = self.n_segments
        # Contiguous-block stream starts: stream i begins at chunk i*M.
        self._starts = np.arange(self.n_segments) * self.stream_len

        calcium = torch.as_tensor(np.asarray(calcium), dtype=torch.float32)
        expected = (self.n_neurons, self.n_total_chunks * self.frames_per_chunk)
        if tuple(calcium.shape) != expected:
            raise ValueError(
                f"calcium shape {tuple(calcium.shape)} != expected {expected} "
                "(n_cells, n_total_chunks*frames_per_chunk)."
            )
        self.calcium = calcium  # (n_cells, total_frames)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, t: int) -> SpikeData:
        cs, fpc = self.chunk_size, self.frames_per_chunk
        chunk_ids = self._starts + t  # (n_segments,) one chunk per stream

        # Each chunk is one storage chunk (cells whole, time-chunk == chunk_size),
        # so these are whole-chunk reads. -> (n_segments, chunk_size, n_cells).
        rate = np.stack(
            [np.asarray(self._rates[:, c * cs : (c + 1) * cs]) for c in chunk_ids]
        )
        rate = np.ascontiguousarray(rate.transpose(0, 2, 1)).astype(np.float32)
        prob = torch.from_numpy(rate) * (self.dt * 1e-3)  # spikes/bin = Hz * s
        input_spikes = torch.rand_like(prob) < prob  # (n_seg, chunk_size, N) bool
        # The smooth rate (expected spikes/bin) is also the firing-rate target:
        # E[filter(input_spikes)] = filter(prob), so a van-Rossum loss on the
        # output spikes vs filter(prob) is unbiased.
        target_rate = prob.clone()

        # Calcium target: per stream, frames [c*fpc, (c+1)*fpc) -> (n_seg, fpc, N).
        cal = torch.stack([self.calcium[:, c * fpc : (c + 1) * fpc] for c in chunk_ids])
        target_calcium = cal.permute(0, 2, 1).clone()  # (n_seg, fpc, n_cells)

        # Restrict the targets to the output cells (input spikes stay full).
        if self.output_idx is not None:
            target_rate = target_rate[:, :, self.output_idx]
            target_calcium = target_calcium[:, :, self.output_idx]

        # Self-consistent target: override the measured rate with the precomputed
        # (already output-cell-shaped) teacher rate for this chunk.
        if self.target_rate_override is not None:
            ov = np.stack(
                [self.target_rate_override[:, c * cs : (c + 1) * cs] for c in chunk_ids]
            )  # (n_seg, n_out, cs)
            target_rate = torch.from_numpy(
                np.ascontiguousarray(ov.transpose(0, 2, 1))
            ).float()  # (n_seg, cs, n_out)
        return SpikeData(
            input_spikes=input_spikes,
            target_calcium=target_calcium,
            target_rate=target_rate,
        )

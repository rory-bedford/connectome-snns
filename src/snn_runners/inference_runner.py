"""
SNN Inference Runner for running inference on spiking neural networks.

This module provides a runner for executing inference (forward passes only)
on SNN models with support for streaming large-scale simulations to zarr
or accumulating results in memory.
"""

import torch
import zarr
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, Literal
from tqdm import tqdm


class SNNInference:
    """
    Runner for SNN inference with zarr or memory output modes.

    This runner handles:
    - Iterating through dataloader chunks
    - Running forward passes through the model
    - Collecting dataloader outputs and model outputs
    - Streaming to zarr or accumulating in memory
    - Progress tracking

    Args:
        model (torch.nn.Module): The SNN model to run inference on
        dataloader: Iterator providing input data (must return named tuples)
        device (str): Device string ('cpu' or 'cuda')
        output_mode (Literal['zarr', 'memory']): How to store results
        zarr_path (Optional[Path]): Path to zarr output (required if output_mode='zarr')
        save_tracked_variables (bool): Whether to save model's tracked variables (default: True)
        max_chunks (Optional[int]): Maximum number of chunks to process (None = process until dataloader exhausts)
        progress_bar (bool): Whether to show progress bar (default: True)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader,
        device: str,
        output_mode: Literal["zarr", "memory"] = "memory",
        zarr_path: Optional[Path] = None,
        save_tracked_variables: bool = True,
        max_chunks: Optional[int] = None,
        progress_bar: bool = True,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.output_mode = output_mode
        self.zarr_path = Path(zarr_path) if zarr_path else None
        self.save_tracked_variables = save_tracked_variables
        self.max_chunks = max_chunks
        self.show_progress = progress_bar

        # Validate inputs
        if output_mode == "zarr" and zarr_path is None:
            raise ValueError("zarr_path must be provided when output_mode='zarr'")

        # Storage for memory mode
        self.memory_storage: Dict[str, list] = {}

        # Zarr datasets
        self.zarr_root: Optional[zarr.Group] = None
        self.zarr_datasets: Dict[str, zarr.Array] = {}

    def run(self) -> Dict[str, Any]:
        """
        Run inference on the model.

        Returns:
            Dict with results:
                - If output_mode='zarr': {'zarr_path': Path, 'metadata': dict}
                - If output_mode='memory': dict with all collected data as numpy arrays
        """
        # Initialize storage
        if self.output_mode == "zarr":
            self._init_zarr_storage()

        # Setup progress bar
        pbar = None
        if self.show_progress:
            total = self.max_chunks if self.max_chunks else None
            pbar = tqdm(total=total, desc="Running inference")

        # Run inference loop
        chunk_idx = 0
        chunk_size = None
        batch_size = None

        with torch.inference_mode():
            for batch_data in self.dataloader:
                # Run forward pass
                outputs = self.model.forward(input_spikes=batch_data.input_spikes)

                # Infer shapes on first iteration
                if chunk_idx == 0:
                    batch_size = batch_data.input_spikes.shape[0]
                    chunk_size = batch_data.input_spikes.shape[1]

                    # Initialize zarr datasets now that we know shapes
                    if self.output_mode == "zarr":
                        self._create_zarr_datasets(
                            batch_data, outputs, batch_size, chunk_size
                        )

                # Store data
                if self.output_mode == "zarr":
                    self._write_to_zarr(chunk_idx, batch_data, outputs, chunk_size)
                else:
                    self._accumulate_in_memory(batch_data, outputs)

                # Update progress
                if pbar:
                    pbar.update(1)

                chunk_idx += 1

                # Check stopping condition
                if self.max_chunks and chunk_idx >= self.max_chunks:
                    break

        # Cleanup
        if pbar:
            pbar.close()

        # Return results
        if self.output_mode == "zarr":
            return {
                "zarr_path": self.zarr_path,
                "num_chunks": chunk_idx,
                "metadata": dict(self.zarr_root.attrs) if self.zarr_root else {},
            }
        else:
            return self._finalize_memory_storage()

    def _init_zarr_storage(self):
        """Initialize zarr group."""
        self.zarr_root = zarr.open_group(self.zarr_path, mode="w")

        # Store dt if model has it
        if hasattr(self.model, "dt"):
            self.zarr_root.attrs["dt"] = float(self.model.dt)

    def _create_zarr_datasets(
        self,
        batch_data,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        batch_size: int,
        chunk_size: int,
    ):
        """Create zarr datasets based on first batch shapes."""
        if self.max_chunks is None:
            raise ValueError("max_chunks must be specified when using zarr output mode")

        total_steps = self.max_chunks * chunk_size

        # Create datasets for dataloader outputs
        for field_name in batch_data._fields:
            data = getattr(batch_data, field_name)
            if isinstance(data, torch.Tensor):
                # Shape: (batch_size, chunk_size, ...) -> (batch_size, total_steps, ...)
                shape = (batch_size, total_steps) + data.shape[2:]
                chunks = (batch_size, chunk_size) + data.shape[2:]
                dtype = self._torch_to_numpy_dtype(data.dtype)

                self.zarr_datasets[field_name] = self.zarr_root.create_dataset(
                    field_name,
                    shape=shape,
                    chunks=chunks,
                    dtype=dtype,
                )

        # Create dataset for output spikes
        if isinstance(outputs, dict):
            spikes = outputs["spikes"]
        else:
            spikes = outputs

        n_neurons = spikes.shape[-1]
        self.zarr_datasets["output_spikes"] = self.zarr_root.create_dataset(
            "output_spikes",
            shape=(batch_size, total_steps, n_neurons),
            chunks=(batch_size, chunk_size, n_neurons),
            dtype=np.bool_,
        )

        # Create datasets for tracked variables if requested
        if self.save_tracked_variables and isinstance(outputs, dict):
            for key, value in outputs.items():
                if key == "spikes":
                    continue  # Already handled as output_spikes

                if isinstance(value, torch.Tensor):
                    shape = (batch_size, total_steps) + value.shape[2:]
                    chunks = (batch_size, chunk_size) + value.shape[2:]
                    dtype = self._torch_to_numpy_dtype(value.dtype)

                    self.zarr_datasets[key] = self.zarr_root.create_dataset(
                        key,
                        shape=shape,
                        chunks=chunks,
                        dtype=dtype,
                    )

    def _write_to_zarr(
        self,
        chunk_idx: int,
        batch_data,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        chunk_size: int,
    ):
        """Write chunk data to zarr."""
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size

        # Write dataloader outputs
        for field_name in batch_data._fields:
            data = getattr(batch_data, field_name)
            if isinstance(data, torch.Tensor) and field_name in self.zarr_datasets:
                if self.device == "cuda":
                    numpy_data = data.cpu().numpy()
                else:
                    numpy_data = data.numpy()
                self.zarr_datasets[field_name][:, start_idx:end_idx, ...] = numpy_data

        # Write output spikes
        if isinstance(outputs, dict):
            spikes = outputs["spikes"]
        else:
            spikes = outputs

        if self.device == "cuda":
            numpy_spikes = spikes.bool().cpu().numpy()
        else:
            numpy_spikes = spikes.bool().numpy()
        self.zarr_datasets["output_spikes"][:, start_idx:end_idx, :] = numpy_spikes

        # Write tracked variables if requested
        if self.save_tracked_variables and isinstance(outputs, dict):
            for key, value in outputs.items():
                if key == "spikes":
                    continue

                if isinstance(value, torch.Tensor) and key in self.zarr_datasets:
                    if self.device == "cuda":
                        numpy_data = value.cpu().numpy()
                    else:
                        numpy_data = value.numpy()
                    self.zarr_datasets[key][:, start_idx:end_idx, ...] = numpy_data

    def _accumulate_in_memory(
        self,
        batch_data,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ):
        """Accumulate chunk data in memory."""
        # Store dataloader outputs
        for field_name in batch_data._fields:
            data = getattr(batch_data, field_name)
            if isinstance(data, torch.Tensor):
                if field_name not in self.memory_storage:
                    self.memory_storage[field_name] = []

                if self.device == "cuda":
                    numpy_data = data.cpu().numpy()
                else:
                    numpy_data = data.numpy()
                self.memory_storage[field_name].append(numpy_data)

        # Store output spikes
        if isinstance(outputs, dict):
            spikes = outputs["spikes"]
        else:
            spikes = outputs

        if "output_spikes" not in self.memory_storage:
            self.memory_storage["output_spikes"] = []

        if self.device == "cuda":
            numpy_spikes = spikes.bool().cpu().numpy()
        else:
            numpy_spikes = spikes.bool().numpy()
        self.memory_storage["output_spikes"].append(numpy_spikes)

        # Store tracked variables if requested
        if self.save_tracked_variables and isinstance(outputs, dict):
            for key, value in outputs.items():
                if key == "spikes":
                    continue

                if isinstance(value, torch.Tensor):
                    if key not in self.memory_storage:
                        self.memory_storage[key] = []

                    if self.device == "cuda":
                        numpy_data = value.cpu().numpy()
                    else:
                        numpy_data = value.numpy()
                    self.memory_storage[key].append(numpy_data)

    def _finalize_memory_storage(self) -> Dict[str, np.ndarray]:
        """Concatenate accumulated chunks into final arrays."""
        result = {}
        for key, chunks in self.memory_storage.items():
            if chunks:
                # Concatenate along time dimension (axis=1)
                result[key] = np.concatenate(chunks, axis=1)
        return result

    @staticmethod
    def _torch_to_numpy_dtype(torch_dtype) -> np.dtype:
        """Convert torch dtype to numpy dtype."""
        if torch_dtype == torch.bool:
            return np.bool_
        elif torch_dtype == torch.float32:
            return np.float32
        elif torch_dtype == torch.float64:
            return np.float64
        elif torch_dtype == torch.int32:
            return np.int32
        elif torch_dtype == torch.int64:
            return np.int64
        else:
            return np.float32  # Default fallback

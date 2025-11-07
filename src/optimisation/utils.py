"""
Checkpoint management utilities for saving and loading model training state.

This module provides utilities for creating and restoring checkpoints during
neural network training, enabling training resumption and model recovery.
"""

import csv
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.amp import GradScaler


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    initial_v: np.ndarray,
    initial_g: np.ndarray,
    initial_g_FF: np.ndarray,
    input_spikes: np.ndarray,
    cv_loss: float,
    fr_loss: float,
    total_loss: float,
    best_loss: float,
) -> bool:
    """Save model checkpoint to disk.

    Args:
        output_dir (Path): Directory where checkpoint will be saved
        epoch (int): Current epoch number
        model (torch.nn.Module): Model to checkpoint
        optimiser (torch.optim.Optimizer): Optimizer state to save
        scaler (GradScaler): Mixed precision scaler state to save
        initial_v (np.ndarray): Current membrane potentials
        initial_g (np.ndarray): Current recurrent conductances
        initial_g_FF (np.ndarray): Current feedforward conductances
        input_spikes (np.ndarray): Input spike trains
        cv_loss (float): Current CV loss value
        fr_loss (float): Current firing rate loss value
        total_loss (float): Current total loss value
        best_loss (float): Best loss seen so far

    Returns:
        bool: True if this is the best model so far, False otherwise
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "initial_v": initial_v,
        "initial_g": initial_g,
        "initial_g_FF": initial_g_FF,
        "input_spikes": input_spikes,
        "cv_loss": cv_loss,
        "fr_loss": fr_loss,
        "total_loss": total_loss,
        "best_loss": best_loss,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }

    # Save as latest checkpoint (for resumption)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    # Save as best if this is the best model
    is_best = total_loss <= best_loss
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"  ✓ New best model saved (loss: {total_loss:.6f})")

    return is_best


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
) -> Tuple[
    int, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], float
]:
    """Load model checkpoint from disk.

    Args:
        checkpoint_path (Path): Path to checkpoint file
        model (torch.nn.Module): Model to load state into
        optimiser (torch.optim.Optimizer): Optimizer to load state into
        scaler (GradScaler): Mixed precision scaler to load state into
        device (str): Device to load tensors onto

    Returns:
        tuple: (epoch, initial_v, initial_g, initial_g_FF, best_loss)
            - epoch (int): Epoch number where training was checkpointed
            - initial_v (torch.Tensor | None): Initial membrane potentials
            - initial_g (torch.Tensor | None): Initial recurrent conductances
            - initial_g_FF (torch.Tensor | None): Initial feedforward conductances
            - best_loss (float): Best loss achieved so far
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    # Load scaler state if available (for backward compatibility)
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]

    # Convert numpy arrays to tensors if needed, then move to device
    if checkpoint["initial_v"] is not None:
        initial_v = checkpoint["initial_v"]
        if isinstance(initial_v, np.ndarray):
            initial_v = torch.from_numpy(initial_v).to(device)
        else:
            initial_v = initial_v.to(device)
    else:
        initial_v = None

    if checkpoint["initial_g"] is not None:
        initial_g = checkpoint["initial_g"]
        if isinstance(initial_g, np.ndarray):
            initial_g = torch.from_numpy(initial_g).to(device)
        else:
            initial_g = initial_g.to(device)
    else:
        initial_g = None

    if checkpoint["initial_g_FF"] is not None:
        initial_g_FF = checkpoint["initial_g_FF"]
        if isinstance(initial_g_FF, np.ndarray):
            initial_g_FF = torch.from_numpy(initial_g_FF).to(device)
        else:
            initial_g_FF = initial_g_FF.to(device)
    else:
        initial_g_FF = None

    best_loss = checkpoint.get("best_loss", float("inf"))

    # Restore random states (convert from numpy if needed)
    rng_state = checkpoint["rng_state"]
    if isinstance(rng_state, np.ndarray):
        # Convert numpy array to ByteTensor
        rng_state = torch.ByteTensor(rng_state.tobytes())
    elif not isinstance(rng_state, torch.Tensor):
        # If it's some other type, try to convert
        rng_state = torch.ByteTensor(rng_state)

    # Ensure it's a ByteTensor (uint8) on CPU
    if rng_state.dtype != torch.uint8:
        rng_state = rng_state.to(torch.uint8)
    if rng_state.device.type != "cpu":
        rng_state = rng_state.cpu()

    torch.set_rng_state(rng_state)
    np.random.set_state(checkpoint["numpy_rng_state"])

    print(f"  ✓ Resumed from epoch {epoch}, best loss: {best_loss:.6f}")
    return epoch, initial_v, initial_g, initial_g_FF, best_loss


class AsyncLogger:
    """Asynchronous logger for training metrics.

    This logger uses background threads to handle I/O operations without blocking
    the training loop. Metrics are buffered and periodically flushed to a CSV file.

    Args:
        log_dir (str | Path): Directory where log files will be saved
        flush_interval (float): Time interval in seconds between automatic flushes

    Example:
        >>> logger = AsyncLogger(log_dir='training_logs')
        >>> for step in range(1000):
        ...     loss, accuracy = train_step()
        ...     logger.log(epoch=step, loss=loss, accuracy=accuracy)
        >>> logger.close()
    """

    def __init__(self, log_dir: str | Path = "logs", flush_interval: float = 5.0):
        """Initialize the async logger.

        Args:
            log_dir (str | Path): Directory where log files will be saved
            flush_interval (float): Time interval in seconds between automatic flushes
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.queue: Queue = Queue()
        self.csv_buffer: list[dict[str, Any]] = []
        self.flush_interval = flush_interval
        self._running = True

        # Initialize CSV file with header
        self.csv_path = self.log_dir / "training_metrics.csv"
        self.csv_fieldnames: Optional[list[str]] = None

        # Start worker thread
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

        # Start periodic flush thread
        self.flusher = threading.Thread(target=self._periodic_flush, daemon=True)
        self.flusher.start()

    def _worker(self):
        """Background worker thread that processes the queue."""
        while True:
            item = self.queue.get()
            if item is None:
                self._flush()
                break

            metric_type, data = item

            if metric_type == "csv_row":
                self._log_csv_row(data)
            elif metric_type == "flush":
                self._flush()

            self.queue.task_done()

    def _log_csv_row(self, data: dict[str, Any]):
        """Buffer a row for CSV output.

        Args:
            data (dict): Dictionary containing all fields for the CSV row
        """
        self.csv_buffer.append(data)

    def _flush(self):
        """Write buffered metrics to disk."""
        # Flush CSV rows
        if self.csv_buffer:
            # Determine fieldnames from first row if not set
            if self.csv_fieldnames is None:
                self.csv_fieldnames = list(self.csv_buffer[0].keys())

            # Check if file exists to determine if we need to write header
            file_exists = self.csv_path.exists()

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)

                # Write header if this is a new file
                if not file_exists:
                    writer.writeheader()

                # Write all buffered rows
                writer.writerows(self.csv_buffer)

            self.csv_buffer.clear()

    def _periodic_flush(self):
        """Periodically flush buffered data to avoid data loss."""
        while self._running:
            time.sleep(self.flush_interval)
            if self._running:
                self.queue.put(("flush", None))

    def log(self, epoch: int, **metrics: float):
        """Log metrics for a given epoch (non-blocking).

        Args:
            epoch (int): Epoch/step number
            **metrics: Arbitrary keyword arguments for metric names and values
        """
        data = {"epoch": epoch, **metrics}
        self.queue.put(("csv_row", data))

    def flush(self):
        """Force an immediate flush of all buffered data."""
        self.queue.put(("flush", None))
        self.queue.join()

    def close(self):
        """Shutdown the logger gracefully, flushing all remaining data."""
        self._running = False
        self.queue.put(None)
        self.worker.join(timeout=10)
        if self.flusher.is_alive():
            self.flusher.join(timeout=1)

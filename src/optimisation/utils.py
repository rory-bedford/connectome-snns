"""
Checkpoint management utilities for saving and loading model training state.

This module provides utilities for creating and restoring checkpoints during
neural network training, enabling training resumption and model recovery.
"""

import csv
import threading
import time
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Any, Optional, Tuple, Dict, Callable

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
    best_loss: float,
    **losses: float,
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
        best_loss (float): Best loss seen so far
        **losses: Arbitrary loss values (must include 'total' for comparison)

    Returns:
        bool: True if this is the best model so far, False otherwise
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Convert numpy arrays to torch tensors for safe loading with weights_only=True
    def to_tensor(arr):
        """Convert numpy array to torch tensor, handling empty arrays."""
        if isinstance(arr, np.ndarray):
            if arr.size == 0:
                # Return empty tensor with original dtype
                return torch.empty(0, dtype=torch.float32)
            return torch.from_numpy(arr)
        return arr

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "initial_v": to_tensor(initial_v),
        "initial_g": to_tensor(initial_g),
        "initial_g_FF": to_tensor(initial_g_FF),
        "input_spikes": to_tensor(input_spikes),
        "best_loss": best_loss,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        **losses,  # Include all provided losses
    }

    # Save as latest checkpoint (for resumption)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    # Save as best if this is the best model (requires 'total' loss)
    total_loss = losses.get("total")
    if total_loss is None:
        raise ValueError(
            "save_checkpoint requires 'total' loss for best model comparison"
        )

    is_best = total_loss <= best_loss
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"  ✓ New best model saved (loss: {total_loss:.6f})")

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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    # Load scaler state if available (for backward compatibility)
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]

    # All data should now be tensors, just move to device
    # Handle None or empty tensors gracefully
    def process_tensor(tensor, target_device):
        """Process tensor, handling None and empty cases."""
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            # Check if empty tensor
            if tensor.numel() == 0:
                return None
            return tensor.to(target_device)
        # Backward compatibility: convert numpy if present
        if isinstance(tensor, np.ndarray):
            if tensor.size == 0:
                return None
            return torch.from_numpy(tensor).to(target_device)
        return None

    initial_v = process_tensor(checkpoint["initial_v"], device)
    initial_g = process_tensor(checkpoint["initial_g"], device)
    initial_g_FF = process_tensor(checkpoint["initial_g_FF"], device)

    best_loss = checkpoint.get("best_loss", float("inf"))

    # Restore random states
    torch.set_rng_state(checkpoint["rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])

    print(f"  ✓ Resumed from epoch {epoch}, best loss: {best_loss:.6f}")
    return epoch, initial_v, initial_g, initial_g_FF, best_loss


class AsyncLogger:
    """Asynchronous logger for training metrics.

    This logger uses a background thread to handle CSV I/O operations without blocking
    the training loop. Each metric entry is written to disk immediately when processed.

    Args:
        log_dir (str | Path): Directory where log files will be saved
        max_queue_size (int): Maximum queue size (blocks when full for backpressure)

    Example:
        >>> logger = AsyncLogger(log_dir='training_logs')
        >>> for step in range(1000):
        ...     loss, accuracy = train_step()
        ...     logger.log(epoch=step, loss=loss, accuracy=accuracy)
        >>> logger.close()
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        max_queue_size: int = 1,
    ):
        """Initialize the async logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.queue: Queue = Queue(maxsize=max_queue_size)
        self._running = True

        # CSV file tracking
        self.csv_path = self.log_dir / "training_metrics.csv"
        self.csv_fieldnames: Optional[list[str]] = None

        # Start worker thread
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        """Background worker thread that writes data to disk."""
        while True:
            try:
                item = self.queue.get()
                if item is None:
                    break

                item_type, data = item

                if item_type == "csv_row":
                    self._write_csv_row(data)
                elif item_type == "weights":
                    self._save_weights(data)

                self.queue.task_done()
            except Exception:
                # Daemon threads fail silently, so we need to handle errors gracefully
                import traceback

                traceback.print_exc()
                self.queue.task_done()

    def _write_csv_row(self, data: dict[str, Any]):
        """Write a single row to CSV file immediately."""
        # Update fieldnames if new fields are present
        all_fieldnames = set(data.keys())
        if self.csv_fieldnames:
            all_fieldnames.update(self.csv_fieldnames)

        # Sort fieldnames (epoch first)
        sorted_fieldnames = sorted(all_fieldnames)
        if "epoch" in sorted_fieldnames:
            sorted_fieldnames.remove("epoch")
            sorted_fieldnames = ["epoch"] + sorted_fieldnames

        fieldnames_changed = self.csv_fieldnames != sorted_fieldnames
        file_exists = self.csv_path.exists()

        if fieldnames_changed:
            self.csv_fieldnames = sorted_fieldnames

            if file_exists:
                # Rewrite file with new header
                with open(self.csv_path, "r") as read_f:
                    reader = csv.DictReader(read_f)
                    existing_data = list(reader)

                with open(self.csv_path, "w", newline="") as write_f:
                    writer = csv.DictWriter(write_f, fieldnames=self.csv_fieldnames)
                    writer.writeheader()
                    for row in existing_data:
                        writer.writerow(row)
                    writer.writerow(data)
            else:
                # New file
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                    writer.writeheader()
                    writer.writerow(data)
        else:
            # Append to existing file
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                writer.writerow(data)

    def _save_weights(self, data: dict[str, Any]):
        """Save weights to disk as compressed numpy arrays."""
        epoch = data["epoch"]
        weights_dir = self.log_dir / "weights"
        weights_dir.mkdir(exist_ok=True)

        recurrent_path = weights_dir / f"recurrent_weights_epoch_{epoch:06d}.npz"
        np.savez_compressed(recurrent_path, weights=data["recurrent_weights"])

        if "feedforward_weights" in data:
            ff_path = weights_dir / f"feedforward_weights_epoch_{epoch:06d}.npz"
            np.savez_compressed(ff_path, weights=data["feedforward_weights"])

    def log(self, epoch: int, **metrics: float):
        """Log metrics for a given epoch."""
        data = {"epoch": epoch, **metrics}
        self.queue.put(("csv_row", data), block=True)

    def save_weights(
        self,
        epoch: int,
        recurrent_weights: np.ndarray,
        feedforward_weights: Optional[np.ndarray] = None,
    ):
        """Save network weights to disk asynchronously."""
        data = {"epoch": epoch, "recurrent_weights": recurrent_weights}
        if feedforward_weights is not None:
            data["feedforward_weights"] = feedforward_weights
        self.queue.put(("weights", data), block=True)

    def close(self):
        """Close the logger and wait for all pending writes."""
        self._running = False
        self.queue.put(None, block=True)
        self.worker.join(timeout=10.0)


class AsyncPlotter:
    """Asynchronous plotter for training visualizations.

    This plotter uses background threads to handle CPU-intensive plotting operations
    without blocking the training loop. Plot jobs are queued and processed asynchronously,
    with automatic cleanup and graceful shutdown.

    Args:
        plot_generator (Callable): Function that generates matplotlib figures
        output_dir (str | Path): Base directory where plots will be saved
        wandb_logger (Optional[Any]): Wandb logger for uploading plots (optional)
        max_queue_size (int): Maximum number of pending plot jobs
        blocking_mode (bool): If True, run plots synchronously for debugging (default: False)

    Example:
        >>> plotter = AsyncPlotter(my_plot_function, output_dir='./plots')
        >>> for epoch in range(1000):
        ...     activity_data = copy_from_gpu(train_step())  # Copy to numpy first
        ...     plotter.submit_plot(activity_data, epoch, weights.cpu().numpy())
        >>> plotter.close()
    """

    def __init__(
        self,
        plot_generator: Callable,
        output_dir: str | Path,
        wandb_logger: Optional[Any] = None,
        max_queue_size: int = 3,
        blocking_mode: bool = False,
    ):
        """Initialize the async plotter.

        Args:
            plot_generator (Callable): Function that generates matplotlib figures
            output_dir (str | Path): Base directory where plots will be saved
            wandb_logger (Optional[Any]): Wandb logger for uploading plots
            max_queue_size (int): Maximum number of pending plot jobs
            blocking_mode (bool): If True, run plots synchronously for debugging
        """
        self.plot_generator = plot_generator
        self.output_dir = Path(output_dir)
        self.wandb_logger = wandb_logger
        self.blocking_mode = blocking_mode

        if self.blocking_mode:
            # Don't initialize queue or thread in blocking mode
            return

        # Thread-safe queue for plot jobs
        self.plot_queue: Queue = Queue(maxsize=max_queue_size)
        self.shutdown_event = threading.Event()
        self._running = True

        # Start background plotting thread
        self.plot_thread = threading.Thread(target=self._plot_worker, daemon=True)
        self.plot_thread.start()

    def submit_plot(
        self,
        plot_data: Dict[str, np.ndarray],
        epoch: int,
        block: bool = True,
        timeout: Optional[float] = 90.0,
    ) -> bool:
        """Submit a plotting job to the background thread.

        Args:
            plot_data (Dict[str, np.ndarray]): Dictionary containing activity data and weights as numpy arrays.
                Should include keys: spikes, voltages, currents, etc., plus:
                - weights: Current network weights
                - feedforward_weights: Current feedforward weights
                - connectome_mask (optional): Binary mask for recurrent connections
                - feedforward_mask (optional): Binary mask for feedforward connections
                - scaling_factors (optional): Recurrent scaling factors
                - scaling_factors_FF (optional): Feedforward scaling factors
            epoch (int): Current epoch number
            block (bool): If True, wait for space in queue. If False, skip if queue full.
            timeout (Optional[float]): Maximum time to wait if blocking (None = wait forever)

        Returns:
            bool: True if job was queued, False if queue full and not blocking or timeout
        """
        # Extract weights and masks from plot_data
        weights = plot_data.pop("weights", None)
        weights_ff = plot_data.pop("feedforward_weights", None)
        connectome_mask = plot_data.pop("connectome_mask", None)
        feedforward_mask = plot_data.pop("feedforward_mask", None)
        scaling_factors = plot_data.pop("scaling_factors", None)
        scaling_factors_FF = plot_data.pop("scaling_factors_FF", None)

        # Create plot job with already-copied numpy data
        plot_job = {
            "plot_data": plot_data.copy(),  # Shallow copy of dict
            "epoch": epoch,
            "weights": weights.copy() if weights is not None else None,
            "weights_ff": weights_ff.copy() if weights_ff is not None else None,
            "connectome_mask": connectome_mask.copy()
            if connectome_mask is not None
            else None,
            "feedforward_mask": feedforward_mask.copy()
            if feedforward_mask is not None
            else None,
            "scaling_factors": scaling_factors.copy()
            if scaling_factors is not None
            else None,
            "scaling_factors_FF": scaling_factors_FF.copy()
            if scaling_factors_FF is not None
            else None,
            "timestamp": time.time(),
        }

        # In blocking mode, execute plot job immediately (synchronously)
        if self.blocking_mode:
            try:
                self._process_plot_job(plot_job)
                return True
            except Exception:
                import traceback

                traceback.print_exc()
                return False

        # Normal async mode
        try:
            if block:
                # Block until queue has space (with timeout)
                self.plot_queue.put(plot_job, block=True, timeout=timeout)
                return True
            else:
                # Non-blocking: skip if queue is full
                self.plot_queue.put_nowait(plot_job)
                return True
        except Full:
            # Queue is full - either timed out or non-blocking skip
            return False

    def _plot_worker(self) -> None:
        """Background worker that processes plotting jobs."""
        while not self.shutdown_event.is_set():
            try:
                # Get next job with timeout
                job = self.plot_queue.get(timeout=1.0)

                # Process the plotting job
                self._process_plot_job(job)

                # Mark job as done
                self.plot_queue.task_done()

            except Empty:
                # Timeout - continue waiting
                continue
            except Exception:
                # Catch any plotting errors to prevent thread crash
                import traceback

                traceback.print_exc()
                # Still mark task as done to prevent hanging
                try:
                    self.plot_queue.task_done()
                except ValueError:
                    pass  # task_done called too many times

    def _process_plot_job(self, job: Dict[str, Any]) -> None:
        """Process a single plotting job.

        Args:
            job (Dict[str, Any]): Plot job containing data and metadata
        """
        epoch = job["epoch"]

        # Create output directory for this epoch
        figures_dir = self.output_dir / "figures" / f"chunk_{epoch + 1:06d}"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Prepare kwargs for plot generator
        plot_kwargs = {
            "weights": job["weights"],
            **job["plot_data"],
        }
        if job["weights_ff"] is not None:
            plot_kwargs["feedforward_weights"] = job["weights_ff"]
        if job["connectome_mask"] is not None:
            plot_kwargs["connectome_mask"] = job["connectome_mask"]
        if job["feedforward_mask"] is not None:
            plot_kwargs["feedforward_mask"] = job["feedforward_mask"]
        if job["scaling_factors"] is not None:
            plot_kwargs["scaling_factors"] = job["scaling_factors"]
        if job["scaling_factors_FF"] is not None:
            plot_kwargs["scaling_factors_FF"] = job["scaling_factors_FF"]

        # Generate plots using the provided function
        figures = self.plot_generator(**plot_kwargs)

        # Save plots to disk
        for plot_name, fig in figures.items():
            fig_path = figures_dir / f"{plot_name}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")

        # Upload to wandb if available
        if self.wandb_logger:
            import wandb

            wandb_plots = {
                f"plots/{name}": wandb.Image(fig) for name, fig in figures.items()
            }
            wandb.log(wandb_plots, step=epoch + 1)

        # Close figures to free memory
        from matplotlib import pyplot as plt

        for fig in figures.values():
            plt.close(fig)

    def has_pending(self) -> bool:
        """Check if there are pending plots in the queue.

        Returns:
            bool: True if plots are pending, False otherwise
        """
        if self.blocking_mode:
            return False
        return not self.plot_queue.empty()

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Wait for all pending plots to complete.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds (None = wait forever)

        Returns:
            bool: True if all plots completed, False if timeout
        """
        if self.blocking_mode:
            return True

        try:
            if timeout is None:
                self.plot_queue.join()
                return True
            else:
                # Queue.join() doesn't support timeout, so we poll
                start_time = time.time()
                while not self.plot_queue.empty():
                    if time.time() - start_time > timeout:
                        return False
                    time.sleep(0.1)
                # Brief wait for last task to complete
                time.sleep(0.5)
                return True
        except Exception:
            return False

    def close(self, timeout: float = 120.0) -> None:
        """Gracefully shutdown the async plotter.

        Args:
            timeout (float): Maximum time to wait for shutdown in seconds
        """
        if self.blocking_mode:
            return

        # Signal shutdown
        self.shutdown_event.set()

        # Wait for remaining jobs to complete
        try:
            self.plot_queue.join()
            self.plot_thread.join(timeout=timeout)
        except Exception:
            pass

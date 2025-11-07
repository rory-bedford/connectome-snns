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

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "initial_v": initial_v,
        "initial_g": initial_g,
        "initial_g_FF": initial_g_FF,
        "input_spikes": input_spikes,
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
        raise ValueError("save_checkpoint requires 'total' loss for best model comparison")
    
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
            elif metric_type == "weights":
                self._save_weights(data)
            elif metric_type == "flush":
                self._flush()

            self.queue.task_done()

    def _log_csv_row(self, data: dict[str, Any]):
        """Buffer a row for CSV output.

        Args:
            data (dict): Dictionary containing all fields for the CSV row
        """
        self.csv_buffer.append(data)

    def _save_weights(self, data: dict[str, Any]):
        """Save weights to disk as compressed numpy arrays.

        Args:
            data (dict): Dictionary containing epoch and weight arrays
        """
        epoch = data["epoch"]
        weights_dir = self.log_dir / "weights"
        weights_dir.mkdir(exist_ok=True)

        # Save recurrent weights
        recurrent_path = weights_dir / f"recurrent_weights_epoch_{epoch:06d}.npz"
        np.savez_compressed(recurrent_path, weights=data["recurrent_weights"])

        # Save feedforward weights if present
        if "feedforward_weights" in data:
            ff_path = weights_dir / f"feedforward_weights_epoch_{epoch:06d}.npz"
            np.savez_compressed(ff_path, weights=data["feedforward_weights"])

    def _flush(self):
        """Write buffered metrics to disk."""
        # Flush CSV rows
        if self.csv_buffer:
            # Update fieldnames to include all fields from all rows
            all_fieldnames = set()
            for row in self.csv_buffer:
                all_fieldnames.update(row.keys())
            
            # Sort fieldnames for consistent ordering (epoch first if present)
            sorted_fieldnames = sorted(all_fieldnames)
            if 'epoch' in sorted_fieldnames:
                sorted_fieldnames.remove('epoch')
                sorted_fieldnames = ['epoch'] + sorted_fieldnames
            
            # Check if file exists and if fieldnames have changed
            file_exists = self.csv_path.exists()
            fieldnames_changed = self.csv_fieldnames != sorted_fieldnames
            
            self.csv_fieldnames = sorted_fieldnames

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)

                # Write header if this is a new file or if fieldnames changed
                if not file_exists or (file_exists and fieldnames_changed):
                    # If file exists and fieldnames changed, we need to rewrite the whole file
                    if file_exists and fieldnames_changed:
                        # Read existing data
                        existing_data = []
                        if self.csv_path.exists():
                            with open(self.csv_path, "r") as read_f:
                                reader = csv.DictReader(read_f)
                                existing_data = list(reader)
                        
                        # Rewrite file with new fieldnames
                        with open(self.csv_path, "w", newline="") as write_f:
                            new_writer = csv.DictWriter(write_f, fieldnames=self.csv_fieldnames)
                            new_writer.writeheader()
                            # Write existing data (missing fields will be empty)
                            for row in existing_data:
                                new_writer.writerow(row)
                    else:
                        # New file, just write header
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

    def save_weights(self, epoch: int, recurrent_weights: np.ndarray, feedforward_weights: Optional[np.ndarray] = None):
        """Save network weights to disk asynchronously.

        Args:
            epoch (int): Epoch/step number
            recurrent_weights (np.ndarray): Recurrent weights as numpy array
            feedforward_weights (Optional[np.ndarray]): Feedforward weights as numpy array
        """
        # Create weights data with already-copied numpy arrays
        weights_data = {
            "epoch": epoch,
            "recurrent_weights": recurrent_weights.copy(),  # Copy numpy array
        }
        
        if feedforward_weights is not None:
            weights_data["feedforward_weights"] = feedforward_weights.copy()

        self.queue.put(("weights", weights_data))

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
    ):
        """Initialize the async plotter.

        Args:
            plot_generator (Callable): Function that generates matplotlib figures
            output_dir (str | Path): Base directory where plots will be saved
            wandb_logger (Optional[Any]): Wandb logger for uploading plots
            max_queue_size (int): Maximum number of pending plot jobs
        """
        self.plot_generator = plot_generator
        self.output_dir = Path(output_dir)
        self.wandb_logger = wandb_logger

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
        weights: np.ndarray,
        weights_ff: Optional[np.ndarray] = None,
    ) -> bool:
        """Submit a plotting job to the background thread.

        Args:
            plot_data (Dict[str, np.ndarray]): Dictionary containing activity data as numpy arrays
            epoch (int): Current epoch number
            weights (np.ndarray): Current network weights as numpy array
            weights_ff (Optional[np.ndarray]): Current feedforward weights as numpy array

        Returns:
            bool: True if job was queued, False if queue is full
        """
        try:
            # Create plot job with already-copied numpy data
            plot_job = {
                "plot_data": plot_data.copy(),  # Shallow copy of dict
                "epoch": epoch,
                "weights": weights.copy(),  # Copy numpy array
                "weights_ff": weights_ff.copy() if weights_ff is not None else None,
                "timestamp": time.time(),
            }

            # Submit to queue (non-blocking)
            self.plot_queue.put_nowait(plot_job)
            return True

        except:
            # Queue is full or other error - skip this plot
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

            except:
                continue

    def _process_plot_job(self, job: Dict[str, Any]) -> None:
        """Process a single plotting job.

        Args:
            job (Dict[str, Any]): Plot job containing data and metadata
        """
        epoch = job["epoch"]

        try:
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
            # Import here to avoid backend issues
            from matplotlib import pyplot as plt

            for fig in figures.values():
                plt.close(fig)

            elapsed = time.time() - job["timestamp"]
            print(f"  ✓ Async plots for epoch {epoch + 1} completed in {elapsed:.2f}s")

        except Exception as e:
            print(f"  ✗ Error processing plot job for epoch {epoch + 1}: {e}")

    def close(self, timeout: float = 30.0) -> None:
        """Gracefully shutdown the async plotter.

        Args:
            timeout (float): Maximum time to wait for shutdown in seconds
        """
        # Signal shutdown
        self.shutdown_event.set()

        # Wait for remaining jobs to complete
        try:
            self.plot_queue.join()
            self.plot_thread.join(timeout=timeout)
        except Exception as e:
            print(f"Warning: AsyncPlotter shutdown encountered an issue: {e}")

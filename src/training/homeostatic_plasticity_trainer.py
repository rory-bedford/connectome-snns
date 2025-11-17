"""
Lightweight PyTorch trainer for homeostatic plasticity in conductance-based SNN models.

This trainer handles just the core training loop logic for homeostatic plasticity
experiments, assuming all components (model, optimizer, dataloaders, loss functions,
loggers, etc.) are already initialized.
"""

import torch
from torch.amp import autocast
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from matplotlib import pyplot as plt

from optimisation.utils import save_checkpoint, AsyncPlotter


class HomeostaticPlasticityTrainer:
    """
    Lightweight trainer for homeostatic plasticity training loops.

    All components (model, optimizer, dataloaders, loss functions, loggers)
    should be initialized beforehand and passed to this trainer.

    Args:
        model (torch.nn.Module): The initialized SNN model
        optimizer (torch.optim.Optimizer): The initialized optimizer
        scaler (torch.amp.GradScaler): The initialized gradient scaler
        spike_dataloader: Iterator providing input spike data
        loss_functions (Dict[str, Callable]): Dict with individual loss functions
        loss_weights (Dict[str, float]): Dict with weights for each loss function
        params: Training parameters object
        device (str): Device string ('cpu' or 'cuda')
        metrics_logger: Initialized async logger
        wandb_logger (Optional[Any]): Initialized wandb run object (optional)
        progress_bar (Optional[Any]): Initialized tqdm progress bar (optional)
        plot_generator (Optional[Callable]): Optional function for generating plots
        stats_computer (Optional[Callable]): Optional function for computing network stats
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        spike_dataloader,
        loss_functions: Dict[str, Callable],
        loss_weights: Dict[str, float],
        params: Any,
        device: str,
        metrics_logger: Any,
        wandb_logger: Optional[Any] = None,
        progress_bar: Optional[Any] = None,
        plot_generator: Optional[Callable] = None,
        stats_computer: Optional[Callable] = None,
    ):
        """Initialize the trainer with pre-configured components."""
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.spike_iter = iter(spike_dataloader)
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        self.params = params
        self.device = device
        self.metrics_logger = metrics_logger
        self.wandb_logger = wandb_logger
        self.pbar = progress_bar
        self.plot_generator = plot_generator
        self.stats_computer = stats_computer

        # Extract commonly used parameters
        self.simulation = params.simulation
        self.training = params.training
        self.hyperparameters = params.hyperparameters

        # Initialize async plotter if plot generator is provided
        self.async_plotter = None
        if self.plot_generator:
            # Will be initialized in train() method with output_dir
            pass

        # Training state
        self.initial_states = {"v": None, "g": None, "g_FF": None}
        self.plot_accumulators = self._init_plot_accumulators()

        # Loss history for running averages (detached from computation graph)
        # maxlen automatically handles rolling window
        self.loss_history = deque(maxlen=self.training.log_interval)

        # Spike accumulator for computing statistics over log_interval chunks
        # This ensures CV and other stats are computed on longer recordings
        self.spike_accumulator = deque(maxlen=self.training.log_interval)

        # Track training state
        self.current_epoch = 0
        self.best_loss = float("inf")

    def _init_plot_accumulators(self) -> Dict[str, List]:
        """Initialize plot data accumulators."""
        return {
            "spikes": [],
            "voltages": [],
            "currents": [],
            "currents_FF": [],
            "conductances": [],
            "conductances_FF": [],
            "input_spikes": [],
        }

    def train(
        self,
        output_dir: Optional[Path] = None,
    ) -> float:
        """
        Run the main training loop.

        The trainer will automatically determine start_epoch and best_loss from
        its internal state, which can be set by calling set_checkpoint_state()
        when resuming from a checkpoint.

        Args:
            output_dir (Optional[Path]): Directory for saving checkpoints and plots

        Returns:
            float: Final best loss value
        """

        # Initialize gradient accumulation
        self.optimizer.zero_grad(set_to_none=True)

        # Initialize async plotter if conditions are met
        if self.plot_generator and output_dir and self.async_plotter is None:
            self.async_plotter = AsyncPlotter(
                plot_generator=self.plot_generator,
                output_dir=output_dir,
                wandb_logger=self.wandb_logger,
                max_queue_size=1,
            )

        # Save initial checkpoint at epoch 0 if starting from beginning
        if self.current_epoch == 0 and output_dir:
            self._save_initial_checkpoint(output_dir)

        # Main training loop
        for epoch in range(self.current_epoch, self.simulation.num_chunks):
            self.current_epoch = epoch

            # Update progress bar at start of iteration
            if self.pbar:
                self.pbar.update(1)

            # Forward pass
            chunk_outputs = self._forward_pass()

            # Compute losses and backward pass (do this BEFORE detaching anything)
            losses = self._compute_losses(chunk_outputs)

            # Update states for next chunk (now safe to detach)
            self._update_initial_states(chunk_outputs)

            # Accumulate detached data for plotting
            self._accumulate_data(chunk_outputs, epoch)

            # Add detached losses to history for running averages
            detached_losses = {name: value for name, value in losses.items()}
            self.loss_history.append(detached_losses)

            # Accumulate spikes for statistics computation (only batch 0, as boolean for memory efficiency)
            self.spike_accumulator.append(
                chunk_outputs["spikes"][0:1, ...].detach().cpu().numpy().astype(bool)
            )

            # Update weights if needed
            if self._should_update_weights(epoch):
                self._update_weights()

            # Log metrics (using running average of losses and accumulated spikes)
            if self._should_log(epoch):
                avg_losses = self._compute_average_losses()
                # Concatenate accumulated spikes along time axis (axis=1)
                accumulated_spikes = np.concatenate(
                    list(self.spike_accumulator), axis=1
                )
                self._log_metrics(avg_losses, accumulated_spikes, epoch)

            # Checkpoint and plot
            if self._should_checkpoint(epoch) and output_dir:
                self.best_loss = self._checkpoint_and_plot(
                    epoch, losses, self.best_loss, output_dir
                )

            # Update progress bar postfix with losses
            self._update_progress_bar(losses)

            # Memory cleanup
            self._cleanup_chunk_data(chunk_outputs)

        # Close progress bar
        if self.pbar:
            self.pbar.close()

        # Flush async logger to ensure all data is written
        if hasattr(self.metrics_logger, "close"):
            self.metrics_logger.close()

        # Close async plotter to ensure all plots are completed
        if self.async_plotter:
            print("Shutting down async plotter...")
            self.async_plotter.close()

        return self.best_loss

    def _forward_pass(self) -> Dict[str, torch.Tensor]:
        """Run forward pass through the model."""
        # Get next input spikes
        input_spikes = next(self.spike_iter)

        # Run network simulation with mixed precision
        with autocast(
            device_type="cuda",
            enabled=self.training.mixed_precision and self.device == "cuda",
        ):
            outputs = self.model.forward(
                input_spikes=input_spikes,
                initial_v=self.initial_states["v"],
                initial_g=self.initial_states["g"],
                initial_g_FF=self.initial_states["g_FF"],
            )

        return {
            "spikes": outputs[0],
            "voltages": outputs[1],
            "currents": outputs[2],
            "currents_FF": outputs[3],
            "conductances": outputs[4],
            "conductances_FF": outputs[5],
            "input_spikes": input_spikes,
        }

    def _update_initial_states(self, chunk_outputs: Dict[str, torch.Tensor]) -> None:
        """Update initial states for next chunk."""
        self.initial_states = {
            "v": chunk_outputs["voltages"][:, -1, :].detach(),
            "g": chunk_outputs["conductances"][:, -1, :, :, :].detach(),
            "g_FF": chunk_outputs["conductances_FF"][:, -1, :, :, :].detach(),
        }

    def _accumulate_data(
        self, chunk_outputs: Dict[str, torch.Tensor], epoch: int
    ) -> None:
        """Accumulate detached data for plotting only."""
        # Accumulate for plotting only when approaching checkpoint
        if self._should_store_for_plot(epoch):
            for key in self.plot_accumulators.keys():
                if key == "input_spikes":
                    tensor = chunk_outputs["input_spikes"]
                else:
                    tensor = chunk_outputs[key]

                # Convert spikes to bool for storage efficiency
                if key == "spikes":
                    tensor = tensor.bool()

                # Only store batch 0 to reduce memory usage 100x
                tensor = tensor[0:1, ...]

                if self.device == "cuda":
                    numpy_array = tensor.detach().cpu().pin_memory().numpy()
                else:
                    numpy_array = tensor.detach().numpy()

                self.plot_accumulators[key].append(numpy_array)

    def _compute_losses(
        self, chunk_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute losses and perform backward pass for a single chunk."""
        spikes = chunk_outputs["spikes"]

        # Compute losses with mixed precision
        with autocast(
            device_type="cuda",
            enabled=self.training.mixed_precision and self.device == "cuda",
        ):
            # Compute individual losses
            individual_losses = {}
            for loss_name, loss_fn in self.loss_functions.items():
                individual_losses[loss_name] = loss_fn(spikes)

            # Compute weighted total loss for optimization
            total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for loss_name, loss_value in individual_losses.items():
                weight = self.loss_weights.get(loss_name, 0.0)
                total_loss += weight * loss_value

            # Scale by number of loss computations per update
            total_loss = total_loss / self.training.chunks_per_update

        # Backward pass with weighted total loss (accumulates gradients)
        self.scaler.scale(total_loss).backward()

        # Convert to detached scalars for logging
        losses = {"total": total_loss.detach().cpu().item()}
        # Add individual loss values for logging
        for loss_name, loss_value in individual_losses.items():
            losses[loss_name] = loss_value.detach().cpu().item()

        # Cleanup
        del total_loss
        for loss_value in individual_losses.values():
            del loss_value

        return losses

    def _update_weights(self) -> None:
        """Update model weights."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _log_metrics(
        self, losses: Dict[str, float], spikes: np.ndarray, epoch: int
    ) -> None:
        """Log metrics to CSV and wandb.

        Args:
            losses: Dictionary of loss values
            spikes: Accumulated spike data as numpy array (already on CPU)
            epoch: Current epoch number
        """
        # Compute statistics if function provided
        stats = {}
        if self.stats_computer:
            stats = self.stats_computer(spikes)

        # Prepare CSV data with _loss suffix for loss names
        csv_data = {}
        for loss_name, loss_value in losses.items():
            csv_data[f"{loss_name}_loss"] = loss_value

        # Add stats without modification
        csv_data.update(stats)

        # Log to CSV with modified loss names
        self.metrics_logger.log(epoch=epoch + 1, **csv_data)

        # Save weights to disk asynchronously (copy tensors to numpy first)
        def copy_tensor_optimized(tensor: torch.Tensor) -> np.ndarray:
            if self.device == "cuda":
                return tensor.detach().cpu().pin_memory().numpy()
            else:
                return tensor.detach().cpu().numpy()

        recurrent_weights_np = copy_tensor_optimized(self.model.weights)
        feedforward_weights_np = copy_tensor_optimized(self.model.weights_FF)

        self.metrics_logger.save_weights(
            epoch=epoch + 1,
            recurrent_weights=recurrent_weights_np,
            feedforward_weights=feedforward_weights_np,
        )

        # Log to wandb
        if self.wandb_logger:
            import wandb

            # Create loss dict with generic naming
            wandb_losses = {}
            for loss_name, loss_value in losses.items():
                wandb_losses[f"loss/{loss_name}"] = loss_value

            wandb.log(
                {
                    **wandb_losses,
                    **stats,
                },
                step=epoch + 1,
            )

    def _save_initial_checkpoint(self, output_dir: Path) -> None:
        """Save initial checkpoint before training begins."""
        # Create initial losses with infinity values for all configured loss functions
        initial_losses = {name: float("inf") for name in self.loss_functions.keys()}
        initial_losses["total"] = float("inf")

        # Ensure we have at least the total loss if no loss functions are configured
        if not initial_losses or len(initial_losses) == 1:
            initial_losses = {"total": float("inf")}

        # Save checkpoint with initial state
        # Use empty arrays for initial states since network hasn't run yet
        save_checkpoint(
            output_dir=output_dir,
            epoch=0,
            model=self.model,
            optimiser=self.optimizer,
            scaler=self.scaler,
            initial_v=np.array([]),  # Empty array for initial checkpoint
            initial_g=np.array([]),  # Empty array for initial checkpoint
            initial_g_FF=np.array([]),  # Empty array for initial checkpoint
            input_spikes=np.array([]),  # Empty array for initial checkpoint
            best_loss=float("inf"),
            **initial_losses,
        )

        # Log initial state to wandb if available
        if self.wandb_logger:
            import wandb

            wandb_losses = {
                f"loss/{name}": value for name, value in initial_losses.items()
            }
            wandb.log(wandb_losses, step=0)

        print(f"Initial checkpoint (epoch 0) saved to {output_dir / 'checkpoints'}")

    def _checkpoint_and_plot(
        self,
        epoch: int,
        losses: Dict[str, float],
        best_loss: float,
        output_dir: Path,
    ) -> float:
        """Save checkpoint and generate plots."""
        if self.pbar:
            self.pbar.clear()

        print(f"\n{'=' * 60}")
        print(f"Checkpoint at chunk {epoch + 1}/{self.simulation.num_chunks}")
        print("=" * 60)

        # Concatenate plot data
        plot_data = self._concatenate_plot_data()

        # Save checkpoint
        is_best = save_checkpoint(
            output_dir=output_dir,
            epoch=epoch + 1,
            model=self.model,
            optimiser=self.optimizer,
            scaler=self.scaler,
            initial_v=self.initial_states["v"].cpu().numpy(),
            initial_g=self.initial_states["g"].cpu().numpy(),
            initial_g_FF=self.initial_states["g_FF"].cpu().numpy(),
            input_spikes=plot_data["input_spikes"],
            **losses,
            best_loss=best_loss,
        )

        if is_best:
            best_loss = losses["total"]

        # Generate plots asynchronously if async plotter is available
        if self.async_plotter and plot_data:
            # Convert tensors to numpy with pinned memory optimization for GPU
            def copy_tensor_optimized(tensor: torch.Tensor) -> np.ndarray:
                if self.device == "cuda":
                    return tensor.detach().cpu().pin_memory().numpy()
                else:
                    return tensor.detach().cpu().numpy()

            # Take only first batch (index 0) to reduce data size, keeping batch dimension
            plot_data = {key: arr[0:1, ...] for key, arr in plot_data.items()}

            # Add weights
            plot_data["weights"] = copy_tensor_optimized(self.model.weights)
            plot_data["feedforward_weights"] = copy_tensor_optimized(
                self.model.weights_FF
            )

            success = self.async_plotter.submit_plot(
                plot_data=plot_data,
                epoch=epoch,
            )
            if success:
                print("  ✓ Plot job submitted for async processing")
            else:
                print("  ⚠ Plot queue full, skipping plots for this checkpoint")
        elif self.plot_generator:
            # Fallback to synchronous plotting if no async plotter
            self._generate_and_save_plots(plot_data, epoch, output_dir)

        # Print checkpoint info
        self._print_checkpoint_info(losses)

        # Cleanup and refresh
        self._clear_plot_data()
        if self.pbar:
            self.pbar.refresh()

        return best_loss

    def _generate_and_save_plots(
        self, plot_data: Dict, epoch: int, output_dir: Path
    ) -> None:
        """Generate and save plots."""
        print("  Generating plots...")

        figures_dir = output_dir / "figures" / f"chunk_{epoch + 1:06d}"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots using provided function
        figures = self.plot_generator(
            spikes=plot_data["spikes"],
            voltages=plot_data["voltages"],
            conductances=plot_data["conductances"],
            conductances_FF=plot_data["conductances_FF"],
            currents=plot_data["currents"],
            currents_FF=plot_data["currents_FF"],
            input_spikes=plot_data["input_spikes"],
            weights=self.model.weights.detach().cpu().numpy(),
            feedforward_weights=self.model.weights_FF.detach().cpu().numpy(),
        )

        # Log to wandb if logger provided
        if self.wandb_logger:
            import wandb

            wandb_plots = {
                f"plots/{name}": wandb.Image(fig) for name, fig in figures.items()
            }
            wandb.log(wandb_plots, step=epoch + 1)

        # Save to disk
        for plot_name, fig in figures.items():
            fig_path = figures_dir / f"{plot_name}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        print(f"  ✓ Saved plots to {figures_dir}")

    def _concatenate_plot_data(self) -> Dict[str, np.ndarray]:
        """Concatenate accumulated plot data."""
        return {
            key: np.concatenate(arrays, axis=1) if arrays else np.array([])
            for key, arrays in self.plot_accumulators.items()
        }

    def _clear_plot_data(self) -> None:
        """Clear plot data and force garbage collection."""
        for key in self.plot_accumulators:
            self.plot_accumulators[key].clear()

        import gc

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _cleanup_chunk_data(self, chunk_outputs: Dict[str, torch.Tensor]) -> None:
        """Clean up chunk data to free memory."""
        # Delete non-essential chunk tensors
        for key in [
            "voltages",
            "currents",
            "currents_FF",
            "conductances",
            "conductances_FF",
        ]:
            if key in chunk_outputs:
                del chunk_outputs[key]

    def _update_progress_bar(self, losses: Dict[str, float]) -> None:
        """Update progress bar with current losses."""
        if self.pbar:
            # Create dynamic postfix with all losses, prioritizing total loss
            postfix = {}

            # Add total loss first
            if "total" in losses:
                postfix["Total"] = f"{losses['total']:.4f}"

            # Add individual losses (excluding 'total')
            for loss_name, loss_value in losses.items():
                if loss_name != "total":
                    # Split on underscore and take first letter of each word
                    words = loss_name.split("_")
                    if len(words) > 1:
                        display_name = "".join(word[0].upper() for word in words)
                    else:
                        display_name = loss_name.upper()[:3]

                    postfix[display_name] = f"{loss_value:.4f}"

            # Add weight update indicator
            chunks_since_update = (
                self.current_epoch + 1
            ) % self.training.chunks_per_update
            if chunks_since_update == 0:
                postfix["Status"] = "Updated"
            else:
                postfix["Status"] = (
                    f"{chunks_since_update}/{self.training.chunks_per_update}"
                )

            self.pbar.set_postfix(postfix)

    def _should_update_weights(self, epoch: int) -> bool:
        """Check if weights should be updated."""
        return (epoch + 1) % self.training.chunks_per_update == 0

    def _should_log(self, epoch: int) -> bool:
        """Check if metrics should be logged."""
        return (epoch + 1) % self.training.log_interval == 0

    def _should_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be saved."""
        return (epoch + 1) % self.training.checkpoint_interval == 0

    def _should_store_for_plot(self, epoch: int) -> bool:
        """Check if data should be stored for plotting."""
        epochs_until_checkpoint = self.training.checkpoint_interval - (
            (epoch + 1) % self.training.checkpoint_interval
        )
        if epochs_until_checkpoint == self.training.checkpoint_interval:
            epochs_until_checkpoint = 0
        return epochs_until_checkpoint < self.training.plot_size

    def _compute_average_losses(self) -> Dict[str, float]:
        """Compute running average of losses over the last log_interval epochs."""
        if not self.loss_history:
            return {}

        # Get all loss names from the history
        all_loss_names = set()
        for loss_dict in self.loss_history:
            all_loss_names.update(loss_dict.keys())

        # Compute average for each loss
        avg_losses = {}
        for loss_name in all_loss_names:
            values = [loss_dict.get(loss_name, 0.0) for loss_dict in self.loss_history]
            avg_losses[loss_name] = sum(values) / len(values)

        return avg_losses

    def _print_checkpoint_info(self, losses: Dict[str, float]) -> None:
        """Print checkpoint information."""
        # Print all losses dynamically
        for loss_name, loss_value in losses.items():
            if loss_name == "total":
                print(f"  Total Loss: {loss_value:.6f}")
            else:
                # Format loss name for display
                display_name = loss_name.replace("_", " ").title()
                print(f"  {display_name} Loss: {loss_value:.6f}")
        print("=" * 60)

    def set_checkpoint_state(
        self,
        epoch: int,
        best_loss: float,
        initial_v: torch.Tensor,
        initial_g: torch.Tensor,
        initial_g_FF: torch.Tensor,
    ) -> None:
        """
        Set checkpoint state for resuming training.

        Args:
            epoch (int): The epoch to resume from
            best_loss (float): The best loss achieved so far
            initial_v (torch.Tensor): Initial voltage states
            initial_g (torch.Tensor): Initial conductance states
            initial_g_FF (torch.Tensor): Initial feedforward conductance states
        """
        self.current_epoch = epoch
        self.best_loss = best_loss
        self.initial_states = {"v": initial_v, "g": initial_g, "g_FF": initial_g_FF}
        """
        Set checkpoint state for resuming training.

        Args:
            epoch (int): The epoch to resume from
            best_loss (float): The best loss achieved so far
            initial_v (torch.Tensor): Initial voltage states
            initial_g (torch.Tensor): Initial conductance states
            initial_g_FF (torch.Tensor): Initial feedforward conductance states
        """
        self.current_epoch = epoch
        self.best_loss = best_loss
        self.initial_states = {"v": initial_v, "g": initial_g, "g_FF": initial_g_FF}

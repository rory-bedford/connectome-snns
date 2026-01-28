"""
Lightweight PyTorch trainer for training conductance-based SNN models.

This trainer handles the core training loop logic for SNN experiments,
assuming all components (model, optimizer, dataloaders, loss functions,
loggers, etc.) are already initialized.
"""

import torch
from torch.amp import autocast
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from matplotlib import pyplot as plt

from training_utils import save_checkpoint, AsyncPlotter, AsyncLogger


class SNNTrainer:
    """
    General-purpose trainer for spiking neural network training loops.

    All components (model, optimizer, dataloaders, loss functions, loggers)
    should be initialized beforehand and passed to this trainer.

    AsyncLogger for metrics CSV logging and wandb are automatically initialized
    when output_dir and wandb_config are provided to the train() method.

    Args:
        model (torch.nn.Module): The initialized SNN model
        optimizer (torch.optim.Optimizer): The initialized optimizer
        scaler (torch.amp.GradScaler): The initialized gradient scaler
        dataloader: Iterator providing input spike data (must return named tuples)
        loss_functions (Dict[str, Callable]): Dict with individual loss functions
        loss_weights (Dict[str, float]): Dict with weights for each loss function
        device (str): Device string ('cpu' or 'cuda')
        num_epochs (int): Total number of training iterations
        chunks_per_update (int): Number of chunks to accumulate before updating weights
        log_interval (int): Log metrics every N epochs
        checkpoint_interval (int): Save checkpoint every N epochs
        plot_size (Optional[int]): Number of chunks to accumulate for plotting (default: None)
        mixed_precision (bool): Whether to use mixed precision training (default: True)
        grad_norm_clip (Optional[float]): Maximum gradient norm for clipping (default: None)
        wandb_config (Optional[Dict[str, Any]]): Wandb configuration dict (optional)
        progress_bar (Optional[Any]): Initialized tqdm progress bar (optional)
        plot_generator (Optional[Callable]): Optional function for generating plots
        stats_computer (Optional[Callable]): Optional function for computing network stats
        connectome_mask (Optional[torch.Tensor]): Binary mask for recurrent connectivity constraints
        feedforward_mask (Optional[torch.Tensor]): Binary mask for feedforward connectivity constraints
        chunks_per_data_epoch (Optional[int]): Number of chunks in one full pass through the dataset.
            When using a cyclic dataloader, this enables automatic state reset at data epoch boundaries.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        dataloader,
        loss_functions: Dict[str, Callable],
        loss_weights: Dict[str, float],
        device: str,
        num_epochs: int,
        chunks_per_update: int,
        log_interval: int,
        checkpoint_interval: int,
        plot_size: Optional[int] = None,
        mixed_precision: bool = True,
        grad_norm_clip: Optional[float] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        progress_bar: Optional[Any] = None,
        plot_generator: Optional[Callable] = None,
        stats_computer: Optional[Callable] = None,
        connectome_mask: Optional[torch.Tensor] = None,
        feedforward_mask: Optional[torch.Tensor] = None,
        chunks_per_data_epoch: Optional[int] = None,
    ):
        """Initialize the trainer with pre-configured components."""
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.spike_iter = iter(dataloader)
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        self.device = device

        # Training parameters
        self.num_epochs = num_epochs
        self.chunks_per_update = chunks_per_update
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.plot_size = plot_size
        self.mixed_precision = mixed_precision
        self.grad_norm_clip = grad_norm_clip

        # Optional components
        self.wandb_config = wandb_config
        self.pbar = progress_bar
        self.plot_generator = plot_generator
        self.stats_computer = stats_computer
        self.connectome_mask = connectome_mask
        self.feedforward_mask = feedforward_mask
        self.chunks_per_data_epoch = chunks_per_data_epoch

        # Loggers (initialized in train() method)
        self.metrics_logger = None
        self.wandb_logger = None
        self.async_plotter = None

        # Training state
        self.plot_accumulators = self._init_plot_accumulators()
        self.loss_history = deque(maxlen=self.log_interval)
        self.accumulated_chunks = 0
        self.current_epoch = 0
        self.best_loss = float("inf")

    def _init_plot_accumulators(self) -> Dict[str, List]:
        """Initialize plot data accumulators."""
        return {
            "spikes": [],
            "voltages": [],
            "currents": [],
            "currents_FF": [],
            "currents_leak": [],
            "conductances": [],
            "conductances_FF": [],
            "input_spikes": [],
            "target_spikes": [],
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

        # Initialize wandb if config provided and not already initialized
        if self.wandb_config and self.wandb_logger is None:
            import wandb

            # Build wandb config from training parameters
            trainer_config = {
                "num_epochs": self.num_epochs,
                "chunks_per_update": self.chunks_per_update,
                "log_interval": self.log_interval,
                "checkpoint_interval": self.checkpoint_interval,
                "mixed_precision": self.mixed_precision,
                "grad_norm_clip": self.grad_norm_clip,
                "output_dir": str(output_dir) if output_dir else None,
                "device": self.device,
            }

            # Merge trainer config with user-provided config
            # Extract 'config' from wandb_config if it exists, otherwise use empty dict
            user_config = self.wandb_config.pop("config", {})
            merged_config = {**user_config, **trainer_config}

            # Build init kwargs with only non-None optional parameters
            wandb_init_kwargs = {
                "name": output_dir.name if output_dir else "snn_training",
                "config": merged_config,
                "dir": str(output_dir) if output_dir else None,
                **self.wandb_config,  # project, entity, tags, etc.
            }

            print("\n" + "=" * 60)
            self.wandb_logger = wandb.init(**wandb_init_kwargs)
            wandb.watch(self.model, log="parameters", log_freq=self.log_interval)

            # Use fractional epoch as x-axis for all metrics
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
            print("=" * 60 + "\n")

        # Initialize async logger if output_dir provided and not already initialized
        if output_dir and self.metrics_logger is None:
            self.metrics_logger = AsyncLogger(
                log_dir=output_dir,
                max_queue_size=10,
            )

        # Initialize async plotter if conditions are met
        if self.plot_generator and output_dir and self.async_plotter is None:
            self.async_plotter = AsyncPlotter(
                plot_generator=self.plot_generator,
                output_dir=output_dir,
                wandb_logger=self.wandb_logger,
                max_queue_size=1,
                blocking_mode=False,
            )

        # Save initial checkpoint at epoch 0 if starting from beginning
        if self.current_epoch == 0 and output_dir:
            self._save_initial_checkpoint(output_dir)

        # Main training loop
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Reset states at data epoch boundaries (when cycling back to start of dataset)
            # This ensures model and loss internal states match the conditions when data was generated
            if (
                self.chunks_per_data_epoch is not None
                and epoch % self.chunks_per_data_epoch == 0
            ):
                self._reset_states_for_data_epoch()

            # Update progress bar at start of iteration
            if self.pbar:
                self.pbar.update(1)

            # Forward pass
            chunk_outputs = self._forward_pass()

            # Compute losses and backward pass (do this BEFORE detaching anything)
            losses = self._compute_losses(chunk_outputs)

            # Accumulate detached data for plotting
            self._accumulate_data(chunk_outputs, epoch)

            # Add detached losses to history for running averages
            detached_losses = {name: value for name, value in losses.items()}
            self.loss_history.append(detached_losses)

            # Update weights if needed
            if self._should_update_weights(epoch):
                self._update_weights()

            # Log metrics (using running average of losses and accumulated data)
            if self._should_log(epoch):
                avg_losses = self._compute_average_losses()
                # Use concatenated plot data (spikes) for stats computation
                plot_data = self._concatenate_plot_data()
                self._log_metrics(avg_losses, plot_data["spikes"], epoch)

            # Checkpoint and plot
            if self._should_checkpoint(epoch) and output_dir:
                self.best_loss = self._checkpoint_and_plot(
                    epoch, losses, self.best_loss, output_dir
                )

            # Periodic flush for async plotter (safety check every 10 log intervals)
            if self.async_plotter and (epoch + 1) % (self.log_interval * 10) == 0:
                if self.async_plotter.has_pending():
                    print(f"  ⚠ Flushing pending plots at epoch {epoch + 1}...")
                    self.async_plotter.flush()

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
        # Get next data from dataloader (named tuple)
        batch_data = next(self.spike_iter)

        # Extract input spikes (always present)
        input_spikes = batch_data.input_spikes

        # Extract target spikes if present (supervised learning)
        target_spikes = getattr(batch_data, "target_spikes", None)

        # Determine if we should track variables for logging/checkpointing
        # Tracking is ONLY for visualization - always detached in simulator
        # Losses only use spikes (+ model parameters accessed directly)
        should_log_checkpoint = (
            self._should_log(self.current_epoch)
            or self._should_checkpoint(self.current_epoch)
            or self.accumulated_chunks < self.plot_size
        )
        self.model.track_variables = should_log_checkpoint

        # Always track only first batch element for memory efficiency
        # (batch_size=1 tracked vs batch_size=N spikes)
        self.model.track_batch_idx = 0 if should_log_checkpoint else None

        # Run network simulation with mixed precision
        with autocast(
            device_type="cuda",
            enabled=self.mixed_precision and self.device == "cuda",
        ):
            outputs = self.model.forward(input_spikes=input_spikes)

        # Handle different return types based on tracking mode
        if should_log_checkpoint:
            # outputs is a dict with all variables (all detached except spikes)
            result = {
                "spikes": outputs["spikes"],
                "input_spikes": input_spikes,
                "target_spikes": target_spikes,
            }
            # Add tracked variables if they exist (for visualization only)
            if "voltages" in outputs:
                result["voltages"] = outputs["voltages"]
            if "currents_recurrent" in outputs:
                result["currents"] = outputs["currents_recurrent"]
            if "currents_feedforward" in outputs:
                result["currents_FF"] = outputs["currents_feedforward"]
            if "currents_leak" in outputs:
                result["currents_leak"] = outputs["currents_leak"]
            if "conductances_recurrent" in outputs:
                result["conductances"] = outputs["conductances_recurrent"]
            if "conductances_feedforward" in outputs:
                result["conductances_FF"] = outputs["conductances_feedforward"]
        else:
            # outputs is just the spike tensor
            result = {
                "spikes": outputs,
                "input_spikes": input_spikes,
                "target_spikes": target_spikes,
            }

        return result

    def _accumulate_data(
        self, chunk_outputs: Dict[str, torch.Tensor], epoch: int
    ) -> None:
        """Accumulate detached data for both plotting and stats computation."""
        # Always accumulate data (used for both stats and plotting)
        # Accumulate up to plot_size, then start dropping oldest
        for key in self.plot_accumulators.keys():
            # Skip keys that aren't present (e.g., voltages when not tracking for losses)
            if key == "input_spikes":
                if "input_spikes" not in chunk_outputs:
                    continue
                tensor = chunk_outputs["input_spikes"]
            elif key == "target_spikes":
                if (
                    "target_spikes" not in chunk_outputs
                    or chunk_outputs["target_spikes"] is None
                ):
                    continue
                tensor = chunk_outputs["target_spikes"]
            elif key not in chunk_outputs:
                continue
            else:
                tensor = chunk_outputs[key]

            # Convert spikes to bool for storage efficiency
            if key in ("spikes", "target_spikes"):
                tensor = tensor.bool()

            # Only store batch 0 to reduce memory usage 100x
            tensor = tensor[0:1, ...]

            if self.device == "cuda":
                numpy_array = tensor.detach().cpu().pin_memory().numpy()
            else:
                numpy_array = tensor.detach().numpy()

            self.plot_accumulators[key].append(numpy_array)

            # Maintain rolling window of plot_size chunks
            if len(self.plot_accumulators[key]) > self.plot_size:
                self.plot_accumulators[key].pop(0)

        # Update accumulated chunks counter (saturates at plot_size)
        self.accumulated_chunks = min(self.accumulated_chunks + 1, self.plot_size)

    def _compute_losses(
        self, chunk_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute losses and perform backward pass for a single chunk."""
        spikes = chunk_outputs["spikes"]

        # Check if any loss function needs weights
        needs_weights = any(
            hasattr(loss_fn, "required_inputs")
            and (
                "recurrent_weights" in loss_fn.required_inputs
                or "feedforward_weights" in loss_fn.required_inputs
            )
            for loss_fn in self.loss_functions.values()
        )

        # Get weights OUTSIDE autocast to avoid float16 overflow in exp()
        # Use the @property which handles both log-space and linear-space weights
        if needs_weights:
            weights_for_loss = self.model.weights
            weights_FF_for_loss = self.model.weights_FF
        else:
            weights_for_loss = None
            weights_FF_for_loss = None

        # Compute losses with mixed precision
        with autocast(
            device_type="cuda",
            enabled=self.mixed_precision and self.device == "cuda",
        ):
            # Compute individual losses
            individual_losses = {}
            for loss_name, loss_fn in self.loss_functions.items():
                # Build inputs based on loss function's required_inputs metadata
                inputs = {}
                if hasattr(loss_fn, "required_inputs"):
                    for req_input in loss_fn.required_inputs:
                        if req_input == "output_spikes":
                            inputs["output_spikes"] = spikes
                        elif req_input == "voltages":
                            inputs["voltages"] = chunk_outputs["voltages"]
                        elif req_input == "dt":
                            inputs["dt"] = self.model.dt
                        elif req_input == "recurrent_weights":
                            inputs["recurrent_weights"] = weights_for_loss
                        elif req_input == "feedforward_weights":
                            inputs["feedforward_weights"] = weights_FF_for_loss
                        elif req_input == "cell_type_indices":
                            inputs["cell_type_indices"] = self.model.cell_type_indices
                        elif req_input == "connectome_mask":
                            inputs["connectome_mask"] = self.connectome_mask
                        elif req_input == "feedforward_mask":
                            inputs["feedforward_mask"] = self.feedforward_mask
                        elif req_input == "scaling_factors":
                            inputs["scaling_factors"] = self.model.scaling_factors
                        elif req_input == "scaling_factors_FF":
                            inputs["scaling_factors_FF"] = self.model.scaling_factors_FF
                else:
                    # For losses without metadata, assume they take output_spikes
                    inputs["output_spikes"] = spikes

                # Add target spikes if the loss requires them
                if hasattr(loss_fn, "requires_target") and loss_fn.requires_target:
                    if chunk_outputs["target_spikes"] is not None:
                        inputs["target_spikes"] = chunk_outputs["target_spikes"]
                    else:
                        raise ValueError(
                            f"Loss function '{loss_name}' requires target spikes, "
                            "but none were provided by the dataloader. "
                            "Ensure your dataset returns (input_spikes, target_spikes)."
                        )

                # Compute loss
                loss_value = loss_fn(**inputs)
                individual_losses[loss_name] = loss_value

            # Compute weighted total loss for optimization
            total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for loss_name, loss_value in individual_losses.items():
                weight = self.loss_weights.get(loss_name, 0.0)
                total_loss += weight * loss_value

            # Scale by number of loss computations per update
            total_loss = total_loss / self.chunks_per_update

        # Backward pass
        self.scaler.scale(total_loss).backward()

        # Convert to detached scalars for logging
        losses = {"total": total_loss.detach().cpu().item()}
        for loss_name, loss_value in individual_losses.items():
            losses[loss_name] = loss_value.detach().cpu().item()

        del total_loss
        for loss_value in individual_losses.values():
            del loss_value

        return losses

    def _update_weights(self) -> None:
        """Update model weights with optional connectome-constrained optimization."""
        self.scaler.unscale_(self.optimizer)

        # Apply connectome masks to gradients before optimizer step
        with torch.no_grad():
            # Check if we're optimizing log weights
            if hasattr(self.model, "log_weights"):
                # Mask log-space recurrent weight gradients
                if self.connectome_mask is not None:
                    if self.model.log_weights.grad is not None:
                        self.model.log_weights.grad *= self.connectome_mask

                # Mask log-space feedforward weight gradients
                if self.feedforward_mask is not None:
                    if self.model.log_weights_FF.grad is not None:
                        self.model.log_weights_FF.grad *= self.feedforward_mask
            else:
                # Legacy: Mask linear-space gradients (for non-weight optimization modes)
                if self.connectome_mask is not None and hasattr(self.model, "weights"):
                    if self.model.weights.grad is not None:
                        self.model.weights.grad *= self.connectome_mask

                if self.feedforward_mask is not None and hasattr(
                    self.model, "weights_FF"
                ):
                    if self.model.weights_FF.grad is not None:
                        self.model.weights_FF.grad *= self.feedforward_mask

        # Clip gradients if configured
        if self.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_norm_clip
            )

        # Store gradient snapshot before zero_grad clears them
        self._last_gradient_snapshot = self._extract_current_gradients()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Remove weight clamping - no longer needed with log parameterization
        # Weights are automatically positive through exp(log_weights)

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

        # Compute stats if stats_computer is provided
        stats = {}
        if self.stats_computer:
            # Copy model parameters to numpy (on CPU) for stats computation
            model_snapshot = {}

            # Add scaling factors if they exist (handle both recurrent and feedforward models)
            if hasattr(self.model, "scaling_factors"):
                if self.device == "cuda":
                    model_snapshot["scaling_factors"] = (
                        self.model.scaling_factors.detach().cpu().numpy()
                    )
                else:
                    model_snapshot["scaling_factors"] = (
                        self.model.scaling_factors.detach().numpy()
                    )

            if hasattr(self.model, "scaling_factors_FF"):
                if self.device == "cuda":
                    model_snapshot["scaling_factors_FF"] = (
                        self.model.scaling_factors_FF.detach().cpu().numpy()
                    )
                else:
                    model_snapshot["scaling_factors_FF"] = (
                        self.model.scaling_factors_FF.detach().numpy()
                    )

            # Compute stats synchronously
            stats = self.stats_computer(spikes, model_snapshot)

        # Extract current gradient values
        gradient_stats = self._last_gradient_snapshot

        # Prepare CSV data
        csv_data = {
            f"{loss_name}_loss": loss_value for loss_name, loss_value in losses.items()
        }
        csv_data.update(stats)
        csv_data.update(gradient_stats)

        # Log to CSV asynchronously
        if self.metrics_logger:
            self.metrics_logger.log(epoch=epoch + 1, **csv_data)

        # Log to wandb directly (synchronous but fast)
        if self.wandb_logger:
            import wandb

            # Create loss dict with wandb naming convention
            wandb_losses = {
                f"loss/{loss_name}": loss_value
                for loss_name, loss_value in losses.items()
            }

            # Compute fractional epoch (uses chunks_per_data_epoch if available)
            if self.chunks_per_data_epoch and self.chunks_per_data_epoch > 0:
                fractional_epoch = (epoch + 1) / self.chunks_per_data_epoch
            else:
                fractional_epoch = epoch + 1

            wandb.log(
                {
                    "epoch": fractional_epoch,
                    **wandb_losses,
                    **stats,
                    **gradient_stats,
                },
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
            wandb.log({"epoch": 0.0, **wandb_losses})

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
        print(f"Checkpoint at chunk {epoch + 1}/{self.num_epochs}")
        print("=" * 60)

        # Concatenate plot data
        plot_data = self._concatenate_plot_data()

        # Capture current model state for checkpoint
        # Handle both recurrent (v, g, g_FF) and feedforward-only (v, g_FF) models
        initial_states = {}
        if hasattr(self.model, "v"):
            initial_states["v"] = self.model.v.cpu().numpy()
        if hasattr(self.model, "g"):
            initial_states["g"] = self.model.g.cpu().numpy()
        else:
            # Feedforward-only model doesn't have recurrent conductances
            initial_states["g"] = np.array([])
        if hasattr(self.model, "g_FF"):
            initial_states["g_FF"] = self.model.g_FF.cpu().numpy()

        # Save checkpoint
        is_best = save_checkpoint(
            output_dir=output_dir,
            epoch=epoch + 1,
            model=self.model,
            optimiser=self.optimizer,
            scaler=self.scaler,
            initial_v=initial_states["v"],
            initial_g=initial_states["g"],
            initial_g_FF=initial_states["g_FF"],
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
            # Only slice if array has data and is multi-dimensional
            plot_data = {
                key: arr[0:1, ...] if arr.size > 0 and arr.ndim > 1 else arr
                for key, arr in plot_data.items()
            }

            # Get weights as separate parameters
            # Handle both recurrent (weights + weights_FF) and feedforward-only (weights_FF) models
            if hasattr(self.model, "weights"):
                weights = copy_tensor_optimized(self.model.weights)
            else:
                weights = None
            if hasattr(self.model, "weights_FF"):
                weights_ff = copy_tensor_optimized(self.model.weights_FF)
            else:
                weights_ff = None

            # Add weights and masks to plot_data
            plot_data["weights"] = weights
            plot_data["feedforward_weights"] = weights_ff
            plot_data["connectome_mask"] = (
                self.connectome_mask.detach().cpu().numpy()
                if self.connectome_mask is not None
                else None
            )
            plot_data["feedforward_mask"] = (
                self.feedforward_mask.detach().cpu().numpy()
                if self.feedforward_mask is not None
                else None
            )

            # Submit plot with blocking to ensure it doesn't get skipped
            print("  📊 Submitting plot (will wait if queue is full)...")
            success = self.async_plotter.submit_plot(
                plot_data=plot_data,
                epoch=epoch,
                block=True,
                timeout=90.0,
            )
            if success:
                print("  ✓ Plot job submitted")
            else:
                print("  ⚠ Plot submission timed out after 90s")
        elif self.plot_generator:
            # Fallback to synchronous plotting if no async plotter
            self._generate_and_save_plots(plot_data, epoch, output_dir)

        # Print checkpoint info
        self._print_checkpoint_info(losses)

        # Refresh progress bar
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
            currents_leak=plot_data["currents_leak"],
            input_spikes=plot_data["input_spikes"],
            target_spikes=plot_data.get("target_spikes"),
            weights=self.model.weights.detach().cpu().numpy(),
            feedforward_weights=self.model.weights_FF.detach().cpu().numpy(),
            connectome_mask=self.connectome_mask.detach().cpu().numpy()
            if self.connectome_mask is not None
            else None,
            feedforward_mask=self.feedforward_mask.detach().cpu().numpy()
            if self.feedforward_mask is not None
            else None,
        )

        # Log to wandb if logger provided
        if self.wandb_logger:
            import wandb

            wandb_plots = {
                f"plots/{name}": wandb.Image(fig) for name, fig in figures.items()
            }
            # Compute fractional epoch for x-axis consistency
            if self.chunks_per_data_epoch and self.chunks_per_data_epoch > 0:
                fractional_epoch = (epoch + 1) / self.chunks_per_data_epoch
            else:
                fractional_epoch = epoch + 1
            wandb.log({"epoch": fractional_epoch, **wandb_plots})

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

        # Don't reset accumulated_chunks - it tracks total progress, not queue size
        # The plot_accumulators themselves maintain a rolling window via pop(0)

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
            "currents_leak",
            "conductances",
            "conductances_FF",
        ]:
            if key in chunk_outputs:
                del chunk_outputs[key]

    def _extract_current_gradients(self) -> Dict[str, float]:
        """Extract current gradient norms for logging."""
        grad_stats = {}

        # Extract gradient norms for main parameters
        if (
            hasattr(self.model, "log_scaling_factors")
            and self.model.log_scaling_factors.grad is not None
        ):
            grad_norm = self.model.log_scaling_factors.grad.norm().item()
            grad_stats["gradients/log_scaling_factors_norm"] = grad_norm

        if (
            hasattr(self.model, "log_scaling_factors_FF")
            and self.model.log_scaling_factors_FF.grad is not None
        ):
            grad_norm = self.model.log_scaling_factors_FF.grad.norm().item()
            grad_stats["gradients/log_scaling_factors_FF_norm"] = grad_norm

        # Also track weight gradients if they exist
        if (
            hasattr(self.model, "log_weights_flat")
            and self.model.log_weights_flat.grad is not None
        ):
            grad_norm = self.model.log_weights_flat.grad.norm().item()
            grad_stats["gradients/log_weights_flat_norm"] = grad_norm

        if (
            hasattr(self.model, "log_weights_FF_flat")
            and self.model.log_weights_FF_flat.grad is not None
        ):
            grad_norm = self.model.log_weights_FF_flat.grad.norm().item()
            grad_stats["gradients/log_weights_FF_flat_norm"] = grad_norm

        return grad_stats

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
            chunks_since_update = (self.current_epoch + 1) % self.chunks_per_update
            if chunks_since_update == 0:
                postfix["Status"] = "Updated"
            else:
                postfix["Status"] = f"{chunks_since_update}/{self.chunks_per_update}"

            self.pbar.set_postfix(postfix)

    def _should_update_weights(self, epoch: int) -> bool:
        """Check if weights should be updated."""
        return (epoch + 1) % self.chunks_per_update == 0

    def _should_log(self, epoch: int) -> bool:
        """Check if metrics should be logged."""
        return (epoch + 1) % self.log_interval == 0

    def _should_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be saved."""
        return (epoch + 1) % self.checkpoint_interval == 0

    def _should_store_for_plot(self, epoch: int) -> bool:
        """Check if data should be stored for plotting."""
        # Only store if plot generator exists and plot_size is configured
        if not self.plot_generator or self.plot_size is None:
            return False

        epochs_until_checkpoint = self.checkpoint_interval - (
            (epoch + 1) % self.checkpoint_interval
        )
        if epochs_until_checkpoint == self.checkpoint_interval:
            epochs_until_checkpoint = 0
        return epochs_until_checkpoint < self.plot_size

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

    def set_checkpoint_state(self, epoch: int, best_loss: float) -> None:
        """
        Set checkpoint state for resuming training.

        Note: Model state (v, g, g_FF) is managed internally by the model via reset_state().
        This method only sets the training loop state (epoch counter and best loss).

        Args:
            epoch (int): The epoch to resume from
            best_loss (float): The best loss achieved so far
        """
        self.current_epoch = epoch
        self.best_loss = best_loss

    def _reset_states_for_data_epoch(self) -> None:
        """
        Reset model and loss function states at data epoch boundaries.

        When using a cyclic dataloader, the original data was generated with the model
        starting from zero internal state at chunk 0. To maintain consistency during
        training, we reset states when cycling back to the start of the dataset.
        """
        # Reset model internal state (voltages, conductances)
        if hasattr(self.model, "reset_state"):
            # Get batch size from model if available
            batch_size = getattr(self.model, "batch_size", 1)
            self.model.reset_state(batch_size=batch_size)

        # Reset loss function internal states
        for loss_fn in self.loss_functions.values():
            if hasattr(loss_fn, "reset_state"):
                loss_fn.reset_state()

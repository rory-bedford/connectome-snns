"""
Generating spike trains from a synthetic connectome

This script generates spiketrains from a predefined synthetic connectome
to be used as teacher activity for fitting recurrent networks.
"""

# ruff: noqa

import numpy as np
import torch
import toml
from inputs.dataloaders import (
    PrecomputedSpikeDataset,
)
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from torch.utils.data import DataLoader
from parameter_loaders import StudentTrainingParams
from optimisation.utils import AsyncLogger
from torch.amp import GradScaler
from optimisation.loss_functions import FiringRateLoss, VanRossumLoss
import wandb
from tqdm import tqdm
from training.snn_trainer import SNNTrainer
from training.weight_perturbers import perturb_weights_scaling_factor


def main(input_dir, output_dir, params_file, wandb_config=None):
    """Main execution function for Dp network simulation.

    Args:
        input_dir (Path, optional): Directory containing input data files (may be None)
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
    """

    # Device Selection and Parameter Loading
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all network parameters from TOML file
    with open(params_file, "r") as f:
        data = toml.load(f)
    params = StudentTrainingParams(**data)

    # Extract commonly used parameter groups
    simulation = params.simulation
    recurrent = params.recurrent
    feedforward = params.feedforward
    training = params.training
    hyperparameters = params.hyperparameters

    # Set random seed if provided
    if simulation.seed is not None:
        np.random.seed(simulation.seed)
        torch.manual_seed(simulation.seed)

    # Load Network Structure from Disk
    network_structure = np.load(input_dir / "network_structure.npz")

    # Extract network components
    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    feedforward_cell_type_indices = network_structure["feedforward_cell_type_indices"]

    connectivity_graph = weights != 0.0
    feedforward_connectivity_graph = feedforward_weights != 0.0

    # Apply Weight Perturbations
    (
        weights,
        feedforward_weights,
        target_scaling_factors,
        target_feedforward_scaling_factors,
    ) = perturb_weights_scaling_factor(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        feedforward_cell_type_indices=feedforward_cell_type_indices,
        variance=training.weight_perturbation_variance,
    )

    # Create targets directory if it doesn't exist
    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    # Save perturbed network structure
    np.savez(
        targets_dir / "perturbed_network_structure.npz",
        recurrent_weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        feedforward_cell_type_indices=feedforward_cell_type_indices,
    )

    # Save target scaling factors
    np.savez(
        targets_dir / "target_scaling_factors.npz",
        recurrent_scaling_factors=target_scaling_factors,
        feedforward_scaling_factors=target_feedforward_scaling_factors,
    )

    # Load precomputed spike data from zarr
    spike_dataset = PrecomputedSpikeDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=simulation.chunk_size,
        device=device,
    )

    n_patterns = spike_dataset.n_patterns
    batch_size = spike_dataset.batch_size

    print(
        f"✓ Loaded {spike_dataset.num_chunks} chunks × {n_patterns} patterns × {batch_size} batches"
    )

    # DataLoader without additional batching (data is already batched)
    # Result shape per iteration: (batch_size, n_patterns, chunk_size, n_neurons)
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,  # No additional batching
        shuffle=False,
        num_workers=0,
    )

    # ======================
    # Initialize LIF Network
    # ======================

    section = "INITIALIZING LIF NETWORK"
    print("\n" + "=" * len(section))
    print(section)
    print("=" * len(section))

    # Initialize conductance-based LIF network model
    model = ConductanceLIFNetwork(
        dt=spike_dataset.dt,
        weights=weights,
        cell_type_indices=cell_type_indices,
        cell_params=recurrent.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        surrgrad_scale=1.0,  # Not used for inference, but required parameter
        weights_FF=feedforward_weights,
        cell_type_indices_FF=feedforward_cell_type_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        optimisable="scaling_factors",
        use_tqdm=False,  # Disable model's internal progress bar
    )

    # Move model to device for GPU acceleration
    model.to(device)
    print(f"Model moved to device: {device}")

    # Jit compile the model for faster execution
    model.compile_step()
    print("✓ Model JIT compiled for faster execution")

    # Print initial scaling factors and target values
    section = "INITIAL SCALING FACTORS AND TARGETS"
    print("\n" + "=" * len(section))
    print(section)
    print("=" * len(section))
    print(
        f"Initial recurrent scaling factors:\n{model.scaling_factors.detach().cpu().numpy()}"
    )
    print(
        f"\nInitial feedforward scaling factors:\n{model.scaling_factors_FF.detach().cpu().numpy()}"
    )
    print(f"\nTarget recurrent scaling factors:\n{target_scaling_factors}")
    print(
        f"\nTarget feedforward scaling factors:\n{target_feedforward_scaling_factors}"
    )
    print("=" * len(section) + "\n")

    # ===============
    # Setup Optimiser
    # ===============

    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    # Mixed precision training scaler (only used if enabled)
    scaler = GradScaler("cuda", enabled=training.mixed_precision and device == "cuda")

    # Initialize all target tensors
    target_rate_tensor = spike_dataset.target_firing_rates

    # Initialize loss functions
    firing_rate_loss_fn = FiringRateLoss(
        target_rate=target_rate_tensor, dt=spike_dataset.dt
    )
    van_rossum_loss_fn = VanRossumLoss(
        tau=hyperparameters.van_rossum_tau,
        dt=spike_dataset.dt,
        window_size=simulation.chunk_size,
        device=device,
    )

    # Define loss weights from config
    loss_weights = {
        "firing_rate": hyperparameters.loss_weight.firing_rate,
        "van_rossum": hyperparameters.loss_weight.van_rossum,
    }

    # =============
    # Setup Loggers
    # =============

    # Initialize async logger for non-blocking metric logging
    metrics_logger = AsyncLogger(log_dir=output_dir, flush_interval=120.0)

    # Setup wandb if config provided (enabled flag already filtered by experiment_runners)
    wandb_run = None
    if wandb_config:
        # Build wandb config from network parameters
        wandb_config_dict = {
            **params.model_dump(),  # Convert all params to dict
            "output_dir": str(output_dir),
            "device": device,
        }

        # Build init kwargs with only non-None optional parameters
        wandb_init_kwargs = {
            "name": output_dir.name,
            "config": wandb_config_dict,
            "dir": str(output_dir),
            **wandb_config,
        }

        print("\n" + "=" * 60)
        wandb_run = wandb.init(**wandb_init_kwargs)
        wandb.watch(model, log="parameters", log_freq=training.log_interval)
        print("=" * 60 + "\n")

    # ===================
    # Setup Training Loop
    # ===================

    section = "SETTING UP TRAINING LOOP"
    print("\n" + "=" * len(section))
    print(section)
    print("=" * len(section))

    # Initialize simulation config with dataset and training parameters
    simulation.setup(
        dt=spike_dataset.dt,
        chunks_per_epoch=spike_dataset.num_chunks,
        epochs=training.epochs,
    )

    # Initialize progress bar for training loop
    pbar = tqdm(
        range(simulation.num_chunks),
        desc="Training",
        unit="chunk",
        total=simulation.num_chunks,
    )

    # Create trainer with all initialized components
    trainer = SNNTrainer(
        model=model,
        optimizer=optimiser,
        scaler=scaler,
        spike_dataloader=spike_dataloader,
        loss_functions={
            "firing_rate": firing_rate_loss_fn,
            "van_rossum": van_rossum_loss_fn,
        },
        loss_weights=loss_weights,
        params=params,
        device=device,
        metrics_logger=metrics_logger,
        wandb_logger=wandb_run,
        progress_bar=pbar,
        connectome_mask=torch.from_numpy(connectivity_graph.astype(np.float32)).to(
            device
        ),
        feedforward_mask=torch.from_numpy(
            feedforward_connectivity_graph.astype(np.float32)
        ).to(device),
    )

    # =================
    # Run Training Loop
    # =================

    section = "STARTING TRAINING"
    print("\n" + "=" * len(section))
    print(section)
    print("=" * len(section))

    # Print training configuration
    print(f"Starting training from chunk {trainer.current_epoch}...")
    print(
        f"Chunk size: {simulation.chunk_size} timesteps ({simulation.chunk_duration_s:.1f}s)"
    )
    print(
        f"Total chunks: {simulation.num_chunks} ({simulation.total_duration_s:.1f}s total)"
    )
    print(f"Epochs: {training.epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Patterns: {n_patterns}")
    print(
        f"Log interval: {training.log_interval} chunks ({training.log_interval_s(simulation.chunk_duration_s):.1f}s)"
    )
    print(
        f"Checkpoint interval: {training.checkpoint_interval} chunks ({training.checkpoint_interval_s(simulation.chunk_duration_s):.1f}s)"
    )

    # Run training with the trainer
    best_loss = trainer.train(output_dir=output_dir)

    # ========
    # Clean Up
    # ========

    section = "TRAINING COMPLETE"
    print("\n" + "=" * len(section))
    print(section)
    print("=" * len(section))
    print(f"Best loss achieved: {best_loss:.6f}")

    # ===================================
    # Save Final State and Run Inference
    # ===================================

    section = "SAVING FINAL STATE AND RUNNING INFERENCE"
    print("\n" + "=" * len(section))
    print(section)
    print("=" * len(section))

    # Create final_state directory
    final_state_dir = output_dir / "final_state"
    final_state_dir.mkdir(parents=True, exist_ok=True)

    # Save final network structure as single npz file
    print("Saving final network structure...")
    np.savez(
        final_state_dir / "network_structure.npz",
        recurrent_weights=model.weights.detach().cpu().numpy(),
        feedforward_weights=model.weights_FF.detach().cpu().numpy(),
        recurrent_connectivity=connectivity_graph,
        feedforward_connectivity=feedforward_connectivity_graph,
        cell_type_indices=cell_type_indices,
        feedforward_cell_type_indices=feedforward_cell_type_indices,
    )
    print(
        f"✓ Final network structure saved to {final_state_dir / 'network_structure.npz'}"
    )

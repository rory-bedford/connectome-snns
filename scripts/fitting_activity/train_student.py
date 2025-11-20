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


def main(input_dir, output_dir, params_file, wandb_config=None):
    """Main execution function for Dp network simulation.

    Args:
        input_dir (Path, optional): Directory containing input data files (may be None)
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
    """

    # ======================================
    # Device Selection and Parameter Loading
    # ======================================

    # Select device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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
        print(f"Using seed: {simulation.seed}")
    else:
        print("No seed specified - using random initialization")

    # ================================
    # Load Network Structure from Disk
    # ================================

    print("Loading network structure from disk...")
    network_structure = np.load(input_dir / "network_structure.npz")

    # Extract network components
    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    input_source_indices = network_structure["feedforward_cell_type_indices"]

    print(f"✓ Loaded network with {len(cell_type_indices)} neurons")
    print(f"✓ Loaded recurrent weights: {weights.shape}")
    print(f"✓ Loaded feedforward weights: {feedforward_weights.shape}")

    # ==============================
    # Setup Optimiser and DataLoader
    # ==============================

    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    # Mixed precision training scaler (only used if enabled)
    scaler = GradScaler("cuda", enabled=training.mixed_precision and device == "cuda")

    print("Loading precomputed spike trains from disk...")

    # Load precomputed spike data from zarr
    spike_dataset = PrecomputedSpikeDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=simulation.chunk_size,
        dataset_name="input_spikes",
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

    # Initialize conductance-based LIF network model
    model = ConductanceLIFNetwork(
        dt=simulation.dt,
        weights=weights,
        cell_type_indices=cell_type_indices,
        cell_params=recurrent.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        surrgrad_scale=1.0,  # Not used for inference, but required parameter
        weights_FF=feedforward_weights,
        cell_type_indices_FF=input_source_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        optimisable=None,  # No optimization for inference
        use_tqdm=False,  # Disable model's internal progress bar
    )

    # Move model to device for GPU acceleration
    model.to(device)
    print(f"Model moved to device: {device}")

    # Jit compile the model for faster execution
    model.compile_step()
    print("✓ Model JIT compiled for faster execution")

    # =====================
    # Define Loss Functions
    # =====================

    # Initialize all target tensors
    target_rate_tensor = spike_dataset.firing_rates

    # Initialize loss functions
    firing_rate_loss_fn = FiringRateLoss(
        target_rate=target_rate_tensor, dt=simulation.dt
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

"""
Training all neurons with feedforward dynamics using constant rate inputs.

This script trains all neurons from a connectome-constrained network using
feedforward-only dynamics, similar to train_feedforward.py, but with one key
difference: the feedforward (external) inputs are replaced with constant fractional
values representing the average firing rate of the original inputs.

The recurrent neuron spikes (teacher activity) are still loaded exactly from disk
and used as additional feedforward inputs. This tests whether the network can learn
the correct dynamics when the external input timing is replaced with constant rate
estimates.

Architecture:
    - Constant fractional rates replace original feedforward inputs
      (value = firing_rate_hz * dt / 1000 at each timestep)
    - Exact teacher recurrent spikes become additional feedforward inputs
    - Combined: [constant rate FF, exact recurrent] -> feedforward network -> output
    - Loss: Van Rossum distance on all neurons vs teacher recurrent spikes
"""

import numpy as np
from dataloaders.supervised import (
    PrecomputedSpikeDataset,
    CyclicSampler,
    SpikeData,
)
from network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm
from training_utils.losses import (
    VanRossumLoss,
)
from configs import (
    StudentSimulationConfig,
    StudentTrainingConfig,
    StudentHyperparameters,
)
from configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig
from snn_runners import SNNTrainer
import toml
import wandb
from visualization.neuronal_dynamics import plot_spike_trains


def make_constant_rate_collate_fn(
    n_feedforward: int,
    feedforward_fractional_rate: float,
):
    """
    Create a collate function that replaces feedforward inputs with constant rates.

    For feedforward neurons: instead of binary spikes, use constant fractional values
    representing the probability of spiking per timestep.

    For recurrent neurons: keep exact teacher spikes as additional inputs.

    Args:
        n_feedforward: Number of feedforward input neurons
        feedforward_fractional_rate: Constant fractional rate for feedforward inputs
                                     (firing_rate_hz * dt / 1000)

    Returns:
        Collate function for DataLoader
    """

    def collate_fn(batch: SpikeData) -> SpikeData:
        """Replace feedforward inputs with constant rates."""
        # batch.input_spikes: (batch, time, n_feedforward) - teacher feedforward spikes
        # batch.target_spikes: (batch, time, n_neurons) - teacher recurrent spikes

        device = batch.input_spikes.device
        batch_size, time_steps, _ = batch.input_spikes.shape

        # Create constant rate tensor for feedforward inputs
        # Shape: (batch, time, n_feedforward)
        constant_ff_inputs = torch.full(
            (batch_size, time_steps, n_feedforward),
            feedforward_fractional_rate,
            dtype=torch.float32,
            device=device,
        )

        # Concatenate: [constant rate FF, exact recurrent spikes]
        # Shape: (batch, time, n_feedforward + n_neurons)
        combined_inputs = torch.cat([constant_ff_inputs, batch.target_spikes], dim=2)

        return SpikeData(
            input_spikes=combined_inputs,
            target_spikes=batch.target_spikes,
        )

    return collate_fn


def main(
    input_dir,
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
):
    """Train all neurons with constant rate inputs to match target spike train activity.

    This function trains a feedforward-unraveled network where:
    - Original feedforward inputs are replaced with constant fractional rates
    - Teacher recurrent spikes are used as exact additional inputs
    - The network learns to reproduce teacher activity using Van Rossum loss

    Args:
        input_dir (Path): Directory containing teacher data (network_structure.npz, spike_data.zarr)
        output_dir (Path): Directory where training outputs will be saved
        params_file (Path): Path to the file containing training parameters
        wandb_config (dict, optional): W&B configuration from experiment.toml
        resume_from (Path, optional): Path to checkpoint to resume from
    """

    # ======================================
    # Device Selection and Parameter Loading
    # ======================================

    # Select device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load network parameters from TOML file
    with open(params_file, "r") as f:
        data = toml.load(f)

    # Load configuration sections
    simulation = StudentSimulationConfig(**data["simulation"])
    training = StudentTrainingConfig(**data["training"])
    hyperparameters = StudentHyperparameters(**data["hyperparameters"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])
    scaling_factors = data.get("scaling_factors", {})

    # Extract parameters into plain Python variables
    chunk_size = simulation.chunk_size
    seed = simulation.seed
    epochs = training.epochs
    chunks_per_update = training.chunks_per_update
    log_interval = training.log_interval
    checkpoint_interval = training.checkpoint_interval
    plot_size = training.plot_size
    mixed_precision = training.mixed_precision
    grad_norm_clip = training.grad_norm_clip
    weight_perturbation_variance = training.weight_perturbation_variance
    optimisable = training.optimisable
    learning_rate = hyperparameters.learning_rate
    surrgrad_scale = hyperparameters.surrgrad_scale
    van_rossum_tau_rise = hyperparameters.van_rossum_tau_rise
    van_rossum_tau_decay = hyperparameters.van_rossum_tau_decay
    loss_weight_van_rossum = hyperparameters.loss_weight.van_rossum

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        print("No seed specified - using random initialization")

    # ================================
    # Load Network Structure from Disk
    # ================================

    network_structure = np.load(input_dir / "network_structure.npz")

    # Extract network components
    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    feedforward_cell_type_indices = network_structure["feedforward_cell_type_indices"]
    recurrent_mask = network_structure["recurrent_connectivity"]
    feedforward_mask = network_structure["feedforward_connectivity"]

    n_neurons = weights.shape[0]
    n_feedforward = feedforward_weights.shape[0]
    n_total_inputs = n_feedforward + n_neurons

    # ===============================================
    # Construct Feedforward Weights from Full Network
    # ===============================================

    # Concatenate feedforward weights (top) and recurrent weights (bottom)
    # Feedforward weights: (n_feedforward, n_neurons) - inputs from FF to all neurons
    # Recurrent weights: (n_neurons, n_neurons) - inputs from recurrent to all neurons
    # Combined: (n_feedforward + n_neurons, n_neurons)
    concatenated_weights = np.concatenate([feedforward_weights, weights], axis=0)

    concatenated_mask = np.concatenate([feedforward_mask, recurrent_mask], axis=0)

    print("\n✓ Feedforward network setup (constant rate inputs):")
    print(f"  - Output neurons: {n_neurons}")
    print(
        f"  - Total inputs per neuron: {n_total_inputs} ({n_feedforward} constant rate FF + {n_neurons} exact recurrent)"
    )
    print(f"  - Weight matrix shape: {concatenated_weights.shape}")
    print(f"  - Active connections: {concatenated_mask.sum()}")

    # ==================================================
    # Construct Combined Input Cell Types and Parameters
    # ==================================================

    # Get cell parameters for all types
    recurrent_cell_params = recurrent.get_cell_params()
    feedforward_cell_params = feedforward.get_cell_params()

    n_ff_cell_types = len(feedforward_cell_params)
    n_rec_cell_types = len(recurrent_cell_params)

    # Concatenate cell type indices: feedforward + recurrent (with offset)
    # Recurrent indices need to be offset by number of feedforward types
    concatenated_cell_type_indices = np.concatenate(
        [feedforward_cell_type_indices, cell_type_indices + n_ff_cell_types]
    )

    # Concatenate cell params: feedforward + recurrent (with offset cell_ids)
    combined_cell_params_FF = feedforward_cell_params.copy()
    for cell_params in recurrent_cell_params:
        # Create new dict with offset cell_id
        offset_cell_params = cell_params.copy()
        offset_cell_params["cell_id"] = cell_params["cell_id"] + n_ff_cell_types
        combined_cell_params_FF.append(offset_cell_params)

    # Get synapse parameters
    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()

    n_ff_synapse_types = len(feedforward_synapse_params)

    # Concatenate synapse params with offset indices
    combined_synapse_params_FF = feedforward_synapse_params.copy()
    for syn_params in recurrent_synapse_params:
        # Create new dict with offset indices
        offset_syn_params = syn_params.copy()
        offset_syn_params["cell_id"] = syn_params["cell_id"] + n_ff_cell_types
        offset_syn_params["synapse_id"] = syn_params["synapse_id"] + n_ff_synapse_types
        combined_synapse_params_FF.append(offset_syn_params)

    print("\n✓ Combined parameters:")
    print(
        f"  - Cell types: {n_ff_cell_types} FF + {n_rec_cell_types} rec = {len(combined_cell_params_FF)}"
    )
    print(
        f"  - Synapse types: {n_ff_synapse_types} FF + {len(recurrent_synapse_params)} rec = {len(combined_synapse_params_FF)}"
    )

    # ============================================
    # Concatenate and Perturb Scaling Factors
    # ============================================

    # Get scaling factors from config
    # Shape: (n_ff_types, n_rec_types) and (n_rec_types, n_rec_types)
    sf_feedforward = np.array(scaling_factors["feedforward"])
    sf_recurrent = np.array(scaling_factors["recurrent"])

    # Concatenate vertically: feedforward on top, recurrent on bottom
    # Shape: (n_ff_types + n_rec_types, n_rec_types)
    concatenated_scaling_factors = np.concatenate(
        [sf_feedforward, sf_recurrent], axis=0
    )

    # Apply weight perturbation to create target scaling factors
    # Sample scaling factors from log-normal distribution
    sigma = np.sqrt(weight_perturbation_variance)
    mu = -(sigma**2) / 2.0

    # Generate perturbation factors matching scaling factor shape
    perturbation_factors = np.random.lognormal(
        mean=mu, sigma=sigma, size=concatenated_scaling_factors.shape
    )

    # Apply perturbation to weights for all neurons
    perturbed_weights = concatenated_weights.copy()
    for input_idx in range(n_total_inputs):
        input_type = concatenated_cell_type_indices[input_idx]
        for output_idx in range(n_neurons):
            output_type = cell_type_indices[output_idx]
            perturbed_weights[input_idx, output_idx] *= perturbation_factors[
                input_type, output_type
            ]

    # Target scaling factors are reciprocals of perturbation
    target_scaling_factors_FF = 1.0 / perturbation_factors

    # Save targets
    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        targets_dir / "target_scaling_factors.npz",
        feedforward_scaling_factors=target_scaling_factors_FF,
    )

    # =============================================
    # Load Dataset and Compute Average Firing Rate
    # =============================================

    # Load spike dataset (raw spikes from disk)
    spike_dataset = PrecomputedSpikeDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
    )

    batch_size = spike_dataset.batch_size

    print(f"\n✓ Loaded {spike_dataset.num_chunks} chunks × {batch_size} batch size")

    # Compute average firing rate from feedforward spikes
    # Sample 10 chunks to estimate the average firing rate
    print("Computing average firing rate from feedforward spikes...")
    n_sample_chunks = min(10, spike_dataset.num_chunks)
    total_ff_spikes = 0
    total_ff_time_steps = 0
    for i in range(n_sample_chunks):
        batch = spike_dataset[i]
        # input_spikes: (batch, time, n_feedforward)
        total_ff_spikes += batch.input_spikes.sum().item()
        total_ff_time_steps += (
            batch.input_spikes.shape[0]
            * batch.input_spikes.shape[1]
            * batch.input_spikes.shape[2]
        )

    # Compute average firing rate in Hz
    avg_spikes_per_neuron_per_timestep = total_ff_spikes / total_ff_time_steps
    avg_firing_rate_hz = avg_spikes_per_neuron_per_timestep * 1000.0 / spike_dataset.dt

    # Compute fractional rate (probability of spike per timestep)
    feedforward_fractional_rate = avg_firing_rate_hz * spike_dataset.dt / 1000.0

    print(f"✓ Average feedforward firing rate: {avg_firing_rate_hz:.2f} Hz")
    print(f"✓ Fractional rate per timestep: {feedforward_fractional_rate:.4f}")
    print("  (constant value used for all feedforward inputs)")

    # Create collate function that replaces FF inputs with constant rates
    collate_fn = make_constant_rate_collate_fn(
        n_feedforward=n_feedforward,
        feedforward_fractional_rate=feedforward_fractional_rate,
    )

    # DataLoader with collate function that replaces FF inputs with constant rates
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ==============================================
    # Initialize Feedforward Network (All Neurons)
    # ==============================================

    model = FeedforwardConductanceLIFNetwork(
        dt=spike_dataset.dt,
        weights_FF=perturbed_weights,
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=concatenated_cell_type_indices,
        cell_params=recurrent_cell_params,  # Full list, model uses cell_type_indices to select
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        scaling_factors_FF=concatenated_scaling_factors,
        optimisable=optimisable,
        feedforward_mask=concatenated_mask,
        track_variables=False,
        use_tqdm=False,
    )

    # Move model to device
    model.to(device)

    print("\n✓ Model initialized:")
    print(f"  - Output neurons: {n_neurons}")
    print(f"  - {n_total_inputs} feedforward inputs per neuron")
    print(f"  - Scaling factors shape: {model.scaling_factors_FF.shape}")

    # ==============================
    # Setup Optimizer and Loss
    # ==============================

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler("cuda", enabled=training.mixed_precision and device == "cuda")

    van_rossum_loss_fn = VanRossumLoss(
        tau_rise=van_rossum_tau_rise,
        tau_decay=van_rossum_tau_decay,
        dt=spike_dataset.dt,
        window_size=chunk_size,
        device=device,
    )

    loss_weights = {
        "van_rossum": loss_weight_van_rossum,
    }

    # ================================================
    # Create Plotting Function for Spike Comparison
    # ================================================

    def plot_generator(
        spikes,
        input_spikes,
        target_spikes,
        **kwargs,
    ):
        """Generate spike train comparison plot for subset of neurons."""
        # spikes: (batch, time, n_neurons) - trained network
        # target_spikes: (batch, time, n_neurons) - target spikes

        # Take first batch, plot first few neurons
        n_plot = min(10, spikes.shape[2])

        # Interleave target and trained for comparison
        # Shape: (1, time, 2*n_plot) alternating [target0, trained0, target1, trained1, ...]
        interleaved = np.zeros((1, spikes.shape[1], 2 * n_plot))
        for i in range(n_plot):
            interleaved[0, :, 2 * i] = target_spikes[0, :, i]
            interleaved[0, :, 2 * i + 1] = spikes[0, :, i]

        # Create cell type indices for coloring (0=target, 1=trained)
        cell_type_indices_plot = np.array([0, 1] * n_plot)
        cell_type_names_plot = ["Target", "Trained"]

        fig = plot_spike_trains(
            spikes=interleaved,
            dt=spike_dataset.dt,
            cell_type_indices=cell_type_indices_plot,
            cell_type_names=cell_type_names_plot,
            n_neurons_plot=2 * n_plot,
            fraction=1.0,
            random_seed=None,
            title=f"Constant Rate Inputs: Target vs Trained (first {n_plot} neurons)",
            ylabel="Neuron",
            figsize=(14, 8),
        )

        return {"spike_comparison": fig}

    # ================================================
    # Create Stats Computer Function
    # ================================================

    def stats_computer(spikes, model_snapshot):
        """Compute summary statistics for all neurons."""
        # spikes shape: (batch, time, n_neurons)
        spikes_np = spikes[0, :, :]  # (time, n_neurons)

        # Compute mean firing rate across all neurons
        spike_counts = spikes_np.sum(axis=0)  # (n_neurons,)
        duration_s = spikes_np.shape[0] * spike_dataset.dt / 1000.0
        firing_rates = spike_counts / duration_s

        stats = {
            "firing_rate/mean": float(firing_rates.mean()),
            "firing_rate/std": float(firing_rates.std()),
            "firing_rate/min": float(firing_rates.min()),
            "firing_rate/max": float(firing_rates.max()),
            "constant_rate_input_rate": avg_firing_rate_hz,
        }

        # Add scaling factor tracking with proper cell type names
        current_sf = model_snapshot["scaling_factors_FF"]
        target_sf = target_scaling_factors_FF

        # Get cell type names for proper labeling
        # Combined input types: feedforward names + recurrent names
        input_cell_type_names = (
            feedforward.cell_types.names + recurrent.cell_types.names
        )
        output_cell_type_names = recurrent.cell_types.names

        # Log all scaling factor elements with descriptive names
        for source_idx in range(current_sf.shape[0]):
            source_type_name = input_cell_type_names[source_idx]
            for target_idx in range(current_sf.shape[1]):
                target_type_name = output_cell_type_names[target_idx]
                synapse_name = f"{source_type_name}_to_{target_type_name}"
                stats[f"scaling_factors/{synapse_name}/value"] = float(
                    current_sf[source_idx, target_idx]
                )
                stats[f"scaling_factors/{synapse_name}/target"] = float(
                    target_sf[source_idx, target_idx]
                )

        return stats

    # ===================
    # Setup Training Loop
    # ===================

    # Update params to reflect concatenated cell types for trainer
    # The trainer reads from feedforward.cell_types.names for labeling
    # Since we concatenated recurrent types as additional inputs, update the config
    feedforward.cell_types.names = (
        feedforward.cell_types.names + recurrent.cell_types.names
    )

    # Calculate total number of training iterations
    num_epochs = epochs * spike_dataset.num_chunks

    pbar = tqdm(
        range(num_epochs),
        desc="Training",
        unit="chunk",
        total=num_epochs,
    )

    # Build wandb config dict if wandb is enabled
    wandb_config_dict = None
    if wandb_config:
        # Separate wandb init kwargs (project, entity, tags, etc.) from config parameters
        wandb_config_dict = {
            **wandb_config,  # W&B initialization kwargs (project, entity, tags, etc.)
            "config": {  # Nest all experiment parameters under 'config'
                "simulation": simulation.model_dump(),
                "training": training.model_dump(),
                "hyperparameters": hyperparameters.model_dump(),
                "recurrent": recurrent.model_dump(),
                "feedforward": feedforward.model_dump(),
                "scaling_factors": scaling_factors,
                "output_dir": str(output_dir),
                "device": device,
                "constant_rate_input_rate": avg_firing_rate_hz,
            },
        }

    # Create trainer
    trainer = SNNTrainer(
        model=model,
        optimizer=optimiser,
        scaler=scaler,
        dataloader=spike_dataloader,
        loss_functions={
            "van_rossum": van_rossum_loss_fn,
        },
        loss_weights=loss_weights,
        device=device,
        num_epochs=num_epochs,
        chunks_per_update=chunks_per_update,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        plot_size=plot_size,
        mixed_precision=mixed_precision,
        grad_norm_clip=grad_norm_clip,
        wandb_config=wandb_config_dict,
        progress_bar=pbar,
        plot_generator=plot_generator,
        stats_computer=stats_computer,
        connectome_mask=None,  # Not used for feedforward-only
        feedforward_mask=torch.from_numpy(concatenated_mask.astype(np.float32)).to(
            device
        ),
        chunks_per_data_epoch=spike_dataset.num_chunks,  # Reset states at data epoch boundaries
    )

    # =================
    # Run Training Loop
    # =================

    print("\nStarting training...")
    chunk_duration_s = chunk_size * spike_dataset.dt / 1000.0
    total_duration_s = num_epochs * chunk_duration_s
    print(f"Chunk size: {chunk_size} timesteps ({chunk_duration_s:.1f}s)")
    print(f"Total chunks: {num_epochs} ({total_duration_s:.1f}s total)")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Training all {n_neurons} neurons with Van Rossum loss")
    print(f"Feedforward inputs: constant rate @ {avg_firing_rate_hz:.2f} Hz")

    model.reset_state(batch_size=batch_size)
    model.track_variables = False
    model.use_tqdm = False

    best_loss = trainer.train(output_dir=output_dir)

    # ========
    # Clean Up
    # ========

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best loss achieved: {best_loss:.6f}")
    print("=" * 60)

    # Save final state
    final_state_dir = output_dir / "final_state"
    final_state_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving final network structure...")
    np.savez(
        final_state_dir / "network_structure.npz",
        feedforward_weights=model.weights_FF.detach().cpu().numpy(),
        feedforward_connectivity=concatenated_mask,
        cell_type_indices=cell_type_indices,
        feedforward_cell_type_indices=concatenated_cell_type_indices,
        scaling_factors_FF=model.scaling_factors_FF.detach().cpu().numpy(),
        constant_rate_input_rate=avg_firing_rate_hz,
    )

    # Close logger
    if trainer.metrics_logger:
        trainer.metrics_logger.close()

    if trainer.wandb_logger is not None:
        wandb.finish()

    print(f"\n✓ Checkpoints: {output_dir / 'checkpoints'}")
    print(f"✓ Figures: {output_dir / 'figures'}")
    print(f"✓ Metrics: {output_dir / 'training_metrics.csv'}")
    print(f"✓ Final state: {final_state_dir / 'network_structure.npz'}")
    print(f"✓ Constant rate input used: {avg_firing_rate_hz:.2f} Hz")

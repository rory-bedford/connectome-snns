"""
Training networks to match spiking outputs in a teacher-student framework.

This script sets up and runs a conductance-based leaky integrate-and-fire
network model of the zebrafish dorsal pallium (Dp) with backpropagation-based
mechanisms to match target activity patterns. The network's connectivity
structure is constrained by a synthetic connectome, while synaptic weights
are optimized to reproduce target firing.
"""

# ruff: noqa

import numpy as np
from synthetic_connectome import (
    topology_generators,
    weight_assigners,
    cell_types,
)
from inputs.dataloaders import PoissonSpikeDataset
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from optimisation.loss_functions import (
    FiringRateLoss,
    VanRossumLoss,
)
from optimisation.utils import load_checkpoint, AsyncLogger
from parameter_loaders import HomeostaticPlasticityParams
from training import SNNTrainer
import toml
import wandb
from visualization.dashboards import (
    create_connectivity_dashboard,
    create_activity_dashboard,
)


def main(
    input_dir,
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
):
    """Main execution function for Dp network homeostatic training.

    Args:
        input_dir (Path, optional): Directory containing input data files (may be None)
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
        wandb_config (dict, optional): W&B configuration from experiment.toml
        resume_from (Path, optional): Path to checkpoint to resume from
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
    params = HomeostaticPlasticityParams(**data)

    # Extract commonly used parameter groups
    simulation = params.simulation
    training = params.training
    targets = params.targets
    hyperparameters = params.hyperparameters
    recurrent = params.recurrent
    feedforward = params.feedforward

    # Set random seed if provided
    if simulation.seed is not None:
        np.random.seed(simulation.seed)
        torch.manual_seed(simulation.seed)
        print(f"Using seed: {simulation.seed}")
    else:
        print("No seed specified - using random initialization")

    # =================================
    # Load Network Structure from Input
    # =================================

    print(f"\nLoading network structure from {input_dir / 'network_structure.npz'}...")
    network_structure = np.load(input_dir / "network_structure.npz")

    # Load weights, connectivity, and cell type indices from saved structure
    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    connectivity_graph = network_structure["recurrent_connectivity"]
    feedforward_connectivity_graph = network_structure["feedforward_connectivity"]
    cell_type_indices = network_structure["cell_type_indices"]
    input_source_indices = network_structure["feedforward_cell_type_indices"]

    print(f"✓ Loaded network structure:")
    print(f"  - Recurrent weights shape: {weights.shape}")
    print(f"  - Feedforward weights shape: {feedforward_weights.shape}")
    print(f"  - Recurrent connectivity: {connectivity_graph.sum():.0f} connections")
    print(
        f"  - Feedforward connectivity: {feedforward_connectivity_graph.sum():.0f} connections"
    )
    print(
        f"  - Cell types: {len(np.unique(cell_type_indices))} recurrent, {len(np.unique(input_source_indices))} input"
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
        surrgrad_scale=hyperparameters.surrgrad_scale,
        weights_FF=feedforward_weights,
        cell_type_indices_FF=input_source_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        optimisable="weights",
        use_tqdm=False,  # Disable tqdm progress bar for training loop
    )

    # Move model to device for GPU acceleration
    model.to(device)

    # ==============================
    # Setup Optimiser and DataLoader
    # ==============================

    # Create firing rates array for input neurons
    input_firing_rates = np.zeros(feedforward.topology.num_neurons)
    for cell_type_name in feedforward.cell_types.names:
        cell_type_idx = feedforward.cell_types.names.index(cell_type_name)
        mask = input_source_indices == cell_type_idx
        input_firing_rates[mask] = feedforward.activity[cell_type_name].firing_rate

    # Initialize Poisson spike generator dataset
    spike_dataset = PoissonSpikeDataset(
        firing_rates=input_firing_rates,
        chunk_size=simulation.chunk_size,
        dt=simulation.dt,
        device=device,
    )

    # Create DataLoader with batch_size from parameters
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=params.training.batch_size,
        shuffle=False,
        num_workers=0,  # Keep 0 for GPU generation
    )

    # =====================
    # Define Loss Functions
    # =====================

    # Initialize target firing rate tensor
    target_rate_tensor = torch.zeros(recurrent.topology.num_neurons, device=device)

    # Populate target firing rates for each cell type
    for cell_type_name in recurrent.cell_types.names:
        cell_type_idx = recurrent.cell_types.names.index(cell_type_name)
        mask = cell_type_indices == cell_type_idx
        target_rate_tensor[mask] = targets.firing_rate[cell_type_name]

    # Initialize loss functions
    firing_rate_loss_fn = FiringRateLoss(
        target_rate=target_rate_tensor, dt=simulation.dt
    )

    # Create VanRossum loss with target spikes bound
    vanrossum_base_fn = VanRossumLoss(
        tau=hyperparameters.vanrossum_tau, dt=simulation.dt
    )

    # Wrapper to bind target_spikes to the loss function
    def vanrossum_loss_fn(output_spikes):
        return vanrossum_base_fn(output_spikes, target_spikes_tensor)

    # Define loss weights from config
    loss_weights = {
        "firing_rate": hyperparameters.loss_weight.firing_rate,
        "vanrossum": hyperparameters.loss_weight.vanrossum,
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

    # ================================================
    # Create Functions for Plotting and Tracking Stats
    # ================================================

    # Pre-compute static data for plotting functions

    # Neuron parameters for plotting
    neuron_params = params.recurrent.get_neuron_params_for_plotting()

    # Synapse names for plotting
    recurrent_synapse_names = params.recurrent.get_synapse_names()
    feedforward_synapse_names = params.feedforward.get_synapse_names()

    # Compute g_bar values for synaptic input histogram
    recurrent_g_bar_by_type = params.recurrent.get_g_bar_by_type()
    feedforward_g_bar_by_type = params.feedforward.get_g_bar_by_type()

    # Define plot generator function that creates both dashboards
    def plot_generator(
        spikes,
        voltages,
        conductances,
        conductances_FF,
        currents,
        currents_FF,
        input_spikes,
        weights,
        feedforward_weights,
    ):
        """Generate connectivity and activity dashboards."""
        # Generate connectivity dashboard
        connectivity_fig = create_connectivity_dashboard(
            connectivity_graph=connectivity_graph,
            weights=weights,
            feedforward_weights=feedforward_weights,
            cell_type_indices=cell_type_indices,
            input_cell_type_indices=input_source_indices,
            cell_type_names=params.recurrent.cell_types.names,
            input_cell_type_names=params.feedforward.cell_types.names,
            num_assemblies=params.recurrent.topology.num_assemblies,
            recurrent_g_bar_by_type=recurrent_g_bar_by_type,
            feedforward_g_bar_by_type=feedforward_g_bar_by_type,
        )

        # Generate activity dashboard
        activity_fig = create_activity_dashboard(
            output_spikes=spikes,
            input_spikes=input_spikes,
            cell_type_indices=cell_type_indices,
            cell_type_names=params.recurrent.cell_types.names,
            dt=params.simulation.dt,
            voltages=voltages,
            neuron_types=cell_type_indices,
            neuron_params=neuron_params,
            recurrent_currents=currents,
            feedforward_currents=currents_FF,
            recurrent_conductances=conductances,
            feedforward_conductances=conductances_FF,
            input_cell_type_names=params.feedforward.cell_types.names,
            recurrent_synapse_names=recurrent_synapse_names,
            feedforward_synapse_names=feedforward_synapse_names,
            window_size=50.0,
            n_neurons_plot=20,
            fraction=1.0,
            random_seed=42,
        )

        return {
            "connectivity_dashboard": connectivity_fig,
            "activity_dashboard": activity_fig,
        }

    # Define stats computer function
    def stats_computer(spikes):
        """Compute summary statistics from network activity."""
        # Compute firing rates per neuron (Hz), averaged over batch
        spike_counts = spikes.sum(axis=1)  # Sum over time: (batch, neurons)
        spike_counts_avg = spike_counts.mean(axis=0)  # Average over batch: (neurons,)
        duration_s = spikes.shape[1] * params.simulation.dt / 1000.0  # Convert ms to s
        firing_rates = spike_counts_avg / duration_s

        # Vectorized CV computation
        from analysis.firing_statistics import compute_spike_train_cv

        cv_values = compute_spike_train_cv(
            spikes, dt=params.simulation.dt
        )  # Shape: (batch, neurons)

        # Suppress warning for neurons with no spikes (expected early in training)
        with np.errstate(invalid="ignore"):
            cv_per_neuron = np.nanmean(cv_values, axis=0)  # Average over batches

        # Compute statistics by cell type
        stats = {}
        for cell_type in np.unique(cell_type_indices):
            mask = cell_type_indices == cell_type
            cell_type_name = params.recurrent.cell_types.names[int(cell_type)]

            # Firing rate statistics
            stats[f"firing_rate/{cell_type_name}/mean"] = float(
                firing_rates[mask].mean()
            )
            stats[f"firing_rate/{cell_type_name}/std"] = float(firing_rates[mask].std())

            # CV statistics (only for neurons with valid CVs)
            cell_cvs = cv_per_neuron[mask]
            valid_cvs = cell_cvs[~np.isnan(cell_cvs)]
            stats[f"cv/{cell_type_name}/mean"] = (
                float(np.mean(valid_cvs)) if len(valid_cvs) > 0 else float("nan")
            )
            stats[f"cv/{cell_type_name}/std"] = (
                float(np.std(valid_cvs)) if len(valid_cvs) > 0 else float("nan")
            )

            # Fraction active
            stats[f"fraction_active/{cell_type_name}"] = float(
                (firing_rates[mask] > 0).mean()
            )

        return stats

    # ==========================
    # Setup Optimiser and Scaler
    # ==========================
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    # Mixed precision training scaler (only used if enabled)
    scaler = GradScaler("cuda", enabled=training.mixed_precision and device == "cuda")

    # ===================
    # Setup Training Loop
    # ===================

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
            "vanrossum": vanrossum_loss_fn,
        },
        loss_weights=loss_weights,
        params=params,
        device=device,
        metrics_logger=metrics_logger,
        wandb_logger=wandb_run,
        progress_bar=pbar,
        plot_generator=plot_generator,
        stats_computer=stats_computer,
        connectome_mask=torch.from_numpy(connectivity_graph.astype(np.float32)).to(
            device
        ),
        feedforward_mask=torch.from_numpy(
            feedforward_connectivity_graph.astype(np.float32)
        ).to(device),
        target_spikes=target_spikes_tensor,
    )

    # Handle checkpoint resuming
    if resume_from is not None:
        start_epoch, initial_v, initial_g, initial_g_FF, best_loss = load_checkpoint(
            checkpoint_path=resume_from,
            model=model,
            optimiser=optimiser,
            scaler=scaler,
            device=device,
        )
        trainer.set_checkpoint_state(
            start_epoch, best_loss, initial_v, initial_g, initial_g_FF
        )
        # Update progress bar to reflect resume point
        pbar.initial = start_epoch
        pbar.refresh()

    # =================
    # Run Training Loop
    # =================

    # Print training configuration
    print(f"Starting training from chunk {trainer.current_epoch}...")
    print(
        f"Chunk size: {simulation.chunk_size} timesteps ({simulation.chunk_duration_s:.1f}s)"
    )
    print(
        f"Total chunks: {simulation.num_chunks} ({simulation.total_duration_s:.1f}s total)"
    )
    print(f"Batch size: {training.batch_size}")
    print(
        f"Log interval: {training.log_interval} chunks ({params.log_interval_s:.1f}s)"
    )
    print(
        f"Checkpoint interval: {training.checkpoint_interval} chunks ({params.checkpoint_interval_s:.1f}s)"
    )

    # Run training with the trainer
    best_loss = trainer.train(output_dir=output_dir)

    # ========
    # Clean Up
    # ========

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best loss achieved: {best_loss:.6f}")
    print("=" * 60)

    # Close async logger and flush all remaining data
    print("Flushing metrics to disk...")
    metrics_logger.close()

    # Finish wandb run
    if wandb_run is not None:
        wandb.finish()

    print(f"\n✓ Checkpoints: {output_dir / 'checkpoints'}")
    print(f"✓ Figures: {output_dir / 'figures'}")
    print(f"✓ Metrics: {metrics_logger.log_dir / 'training_metrics.csv'}")

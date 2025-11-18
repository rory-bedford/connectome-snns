"""
Simulating Dp with homeostatic plasticity to achieve target activity regime.

This script sets up and runs a conductance-based leaky integrate-and-fire
network model of the zebrafish dorsal pallium (Dp) with homeostatic plasticity
mechanisms to regulate neuron firing rates and spike train statistics.

Overview:
1. First we generate a biologically plausible recurrent weight matrix with a
   Dp-inspired assembly structure.
2. Next we generate excitatory mitral cell inputs from the OB with Poisson
   statistics and sparse projections.
3. Then we initialise our network with parameters adapted from
   Meissner-Bernard et al. (2025) https://doi.org/10.1016/j.celrep.2025.115330
4. We then run our network simulation in a training loop, applying updates to
   the connectome-constrained weights every so often to optimise the activity
   towards target firing rates and spike train CVs.
"""

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
from optimisation.loss_functions import (
    FiringRateLoss,
    CVLoss,
    SilentNeuronPenalty,
    SubthresholdVarianceLoss,
)
from optimisation.utils import load_checkpoint, AsyncLogger
from network_simulators.conductance_based.parameter_loader import (
    HomeostaticPlasticityParams,
)
from training import HomeostaticPlasticityTrainer
import toml
import wandb
from visualization.dashboards import (
    create_connectivity_dashboard,
    create_activity_dashboard,
)


def main(
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
    resumed_output_dir=None,
):
    """Main execution function for Dp network homeostatic training.

    Args:
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
        wandb_config (dict, optional): W&B configuration from experiment.toml
        resume_from (Path, optional): Path to checkpoint to resume from
        resumed_output_dir (Path, optional): Separate directory for plots when resuming training
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

    # ===================================
    # Create Recurrent Connectivity Graph
    # ===================================

    # Assign cell types to recurrent layer
    cell_type_indices = cell_types.assign_cell_types(
        num_neurons=recurrent.topology.num_neurons,
        cell_type_proportions=recurrent.cell_types.proportion,
    )

    # Generate assembly-based connectivity graph
    connectivity_graph = topology_generators.assembly_generator(
        source_cell_types=cell_type_indices,
        target_cell_types=cell_type_indices,  # Same for recurrent connections
        num_assemblies=recurrent.topology.num_assemblies,
        conn_within=recurrent.topology.conn_within,
        conn_between=recurrent.topology.conn_between,
        method="configuration",  # Ensures exact in-degree/out-degree distributions
    )

    # =======================
    # Assign Synaptic Weights
    # =======================

    # Assign log-normal weights to connectivity graph (no signs for conductance-based)
    weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=connectivity_graph,
        source_cell_indices=cell_type_indices,
        target_cell_indices=cell_type_indices,  # Same for recurrent connections
        cell_type_signs=None,  # No signs for conductance-based model
        w_mu_matrix=recurrent.weights.w_mu,
        w_sigma_matrix=recurrent.weights.w_sigma,
        parameter_space="linear",
    )

    # ==========================================
    # Create Feedforward Connections and Weights
    # ==========================================

    # Assign cell types to input layer
    input_source_indices = cell_types.assign_cell_types(
        num_neurons=feedforward.topology.num_neurons,
        cell_type_proportions=feedforward.cell_types.proportion,
    )

    # Generate feedforward connectivity graph
    feedforward_connectivity_graph = topology_generators.sparse_graph_generator(
        source_cell_types=input_source_indices,
        target_cell_types=cell_type_indices,  # Connectome layer cell assignments
        conn_matrix=feedforward.topology.conn_inputs,  # N_input_types x N_recurrent_types matrix
        allow_self_loops=True,  # Allow for feedforward connections
        method="configuration",  # Ensures exact in-degree/out-degree distributions
    )

    # Assign log-normal weights to feedforward connectivity (no signs for conductance-based)
    feedforward_weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=feedforward_connectivity_graph,
        source_cell_indices=input_source_indices,
        target_cell_indices=cell_type_indices,
        cell_type_signs=None,  # No signs for conductance-based model
        w_mu_matrix=feedforward.weights.w_mu,
        w_sigma_matrix=feedforward.weights.w_sigma,
        parameter_space="linear",
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

    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    # Mixed precision training scaler (only used if enabled)
    scaler = GradScaler("cuda", enabled=training.mixed_precision and device == "cuda")

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

    # Initialize all target tensors
    target_rate_tensor = torch.zeros(recurrent.topology.num_neurons, device=device)
    target_cv_tensor = torch.ones(recurrent.topology.num_neurons, device=device)
    alpha_tensor = torch.zeros(recurrent.topology.num_neurons, device=device)
    v_threshold_tensor = torch.zeros(recurrent.topology.num_neurons, device=device)
    threshold_ratio_tensor = torch.zeros(recurrent.topology.num_neurons, device=device)

    # Populate all tensors in a single loop over cell types
    for cell_type_name in recurrent.cell_types.names:
        cell_type_idx = recurrent.cell_types.names.index(cell_type_name)
        mask = cell_type_indices == cell_type_idx

        # Set cell-type-specific values
        target_rate_tensor[mask] = targets.firing_rate[cell_type_name]
        alpha_tensor[mask] = targets.alpha[cell_type_name]
        threshold_ratio_tensor[mask] = targets.threshold_ratio[cell_type_name]
        v_threshold_tensor[mask] = recurrent.physiology[cell_type_name].theta

    # Initialize loss functions
    firing_rate_loss_fn = FiringRateLoss(
        target_rate=target_rate_tensor, dt=simulation.dt
    )
    cv_loss_fn = CVLoss(target_cv=target_cv_tensor)
    silent_penalty_fn = SilentNeuronPenalty(alpha=alpha_tensor, dt=simulation.dt)
    membrane_variance_loss_fn = SubthresholdVarianceLoss(
        v_threshold=v_threshold_tensor, target_ratio=threshold_ratio_tensor
    )

    # Define loss weights from config
    loss_weights = {
        "firing_rate": hyperparameters.loss_weight.firing_rate,
        "cv": hyperparameters.loss_weight.cv,
        "silent_penalty": hyperparameters.loss_weight.silent_penalty,
        "membrane_variance": hyperparameters.loss_weight.membrane_variance,
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
    trainer = HomeostaticPlasticityTrainer(
        model=model,
        optimizer=optimiser,
        scaler=scaler,
        spike_dataloader=spike_dataloader,
        loss_functions={
            "firing_rate": firing_rate_loss_fn,
            "cv": cv_loss_fn,
            "silent_penalty": silent_penalty_fn,
            "membrane_variance": membrane_variance_loss_fn,
        },
        loss_weights=loss_weights,
        params=params,
        device=device,
        metrics_logger=metrics_logger,
        wandb_logger=wandb_run,
        progress_bar=pbar,
        plot_generator=plot_generator,
        stats_computer=stats_computer,
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

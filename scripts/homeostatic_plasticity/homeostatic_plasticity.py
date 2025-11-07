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
from functools import partial
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
import sys
from pathlib import Path
from tqdm import tqdm
from optimisation.loss_functions import CVLoss, FiringRateLoss
from optimisation.utils import load_checkpoint, AsyncLogger
from network_simulators.conductance_based.parameter_loader import (
    HomeostaticPlasticityParams,
)
from training import HomeostaticPlasticityTrainer
import toml
import wandb


# Import local scripts
sys.path.append(str(Path(__file__).resolve().parent))
import homeostatic_plots


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
    for ct_idx, ct_name in enumerate(feedforward.cell_types.names):
        mask = input_source_indices == ct_idx
        input_firing_rates[mask] = feedforward.activity.firing_rates[ct_name]

    # Initialize Poisson spike generator dataset
    spike_dataset = PoissonSpikeDataset(
        firing_rates=input_firing_rates,
        chunk_size=simulation.chunk_size,
        dt=simulation.dt,
        device=device,
    )

    # Create DataLoader (batch_size=1 for single chunks, no shuffle for continuous time)
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Keep 0 for GPU generation
    )

    # Define loss functions
    target_cv_tensor = (
        torch.ones(recurrent.topology.num_neurons, device=device) * targets.cv
    )
    target_rate_tensor = (
        torch.ones(recurrent.topology.num_neurons, device=device) * targets.firing_rate
    )
    cv_loss_fn = CVLoss(
        target_cv=target_cv_tensor, penalty_value=hyperparameters.cv_high_loss
    )
    firing_rate_loss_fn = FiringRateLoss(
        target_rate=target_rate_tensor, dt=simulation.dt
    )

    # Define loss weights for optimization
    loss_weights = {
        "cv": 1 - hyperparameters.loss_ratio,
        "firing_rate": hyperparameters.loss_ratio,
    }

    # =============
    # Setup Loggers
    # =============

    # Initialize async logger for non-blocking metric logging
    metrics_logger = AsyncLogger(log_dir=output_dir / "metrics", flush_interval=120.0)

    # Setup wandb if enabled
    wandb_run = None
    if wandb_config and wandb_config.get("enabled", False):
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

        wandb_run = wandb.init(**wandb_init_kwargs)
        wandb.watch(model, log="all", log_freq=training.log_interval)

    # ===================
    # Setup Training Loop
    # ===================

    # Pre-compute static data for plotting functions

    # Neuron parameters for plotting
    neuron_params = {}
    for idx, cell_name in enumerate(params.recurrent.cell_types.names):
        cell_params = params.recurrent.physiology[cell_name]
        neuron_params[idx] = {
            "threshold": cell_params["theta"],
            "rest": cell_params["U_reset"],
            "name": cell_name,
            "sign": 1 if "excit" in cell_name.lower() else -1,
        }

    # Synapse names for plotting
    recurrent_synapse_names = {}
    for cell_type in params.recurrent.cell_types.names:
        recurrent_synapse_names[cell_type] = params.recurrent.synapses[cell_type][
            "names"
        ]

    feedforward_synapse_names = {}
    for cell_type in params.feedforward.cell_types.names:
        feedforward_synapse_names[cell_type] = params.feedforward.synapses[cell_type][
            "names"
        ]

    # Compute g_bar values for synaptic input histogram
    recurrent_g_bar_by_type = {}
    for cell_type in params.recurrent.cell_types.names:
        g_bar_values = params.recurrent.synapses[cell_type]["g_bar"]
        recurrent_g_bar_by_type[cell_type] = sum(g_bar_values)

    feedforward_g_bar_by_type = {}
    for cell_type in params.feedforward.cell_types.names:
        g_bar_values = params.feedforward.synapses[cell_type]["g_bar"]
        feedforward_g_bar_by_type[cell_type] = sum(g_bar_values)

    # Create pre-configured plot generator function
    plot_generator = partial(
        homeostatic_plots.generate_training_plots,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_source_indices,
        cell_type_names=params.recurrent.cell_types.names,
        input_cell_type_names=params.feedforward.cell_types.names,
        connectivity_graph=connectivity_graph,
        num_assemblies=params.recurrent.topology.num_assemblies,
        dt=params.simulation.dt,
        neuron_params=neuron_params,
        recurrent_synapse_names=recurrent_synapse_names,
        feedforward_synapse_names=feedforward_synapse_names,
        recurrent_g_bar_by_type=recurrent_g_bar_by_type,
        feedforward_g_bar_by_type=feedforward_g_bar_by_type,
    )

    # Create pre-configured stats computer function
    stats_computer = partial(
        homeostatic_plots.compute_network_statistics,
        cell_type_indices=cell_type_indices,
        cell_type_names=params.recurrent.cell_types.names,
        dt=params.simulation.dt,
    )

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
        loss_functions={"cv": cv_loss_fn, "firing_rate": firing_rate_loss_fn},
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
    print(f"Total chunks: {simulation.num_chunks}")
    print(f"Log interval: {training.log_interval} chunks")
    print(f"Checkpoint interval: {training.checkpoint_interval} chunks")

    # Run training with the trainer
    best_loss = trainer.train(output_dir=output_dir)

    # ========
    # Clean Up
    # ========

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final best loss: {best_loss:.6f}")
    print("=" * 60)

    # Finish wandb run
    if wandb_run is not None:
        wandb.finish()

    # Close async logger and flush all remaining data
    print("  Flushing metrics to disk...")
    metrics_logger.close()

    print(f"✓ Checkpoints saved to {output_dir / 'checkpoints'}")
    print(f"✓ Metrics saved to {output_dir / 'metrics'}")
    print(f"✓ Best loss achieved: {best_loss:.6f}")

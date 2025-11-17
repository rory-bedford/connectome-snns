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
from optimisation.loss_functions import FiringRateLoss
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
from analysis.firing_statistics import (
    compute_firing_rate_by_cell_type,
    compute_cv_by_cell_type,
)
from analysis.voltage_statistics import (
    compute_membrane_potential_by_cell_type,
)
import pandas as pd
import matplotlib.pyplot as plt


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
    for ct_idx, ct_name in enumerate(feedforward.cell_types.names):
        mask = input_source_indices == ct_idx
        input_firing_rates[mask] = feedforward.activity[ct_name].firing_rate

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

    # Define loss functions - create target tensor with cell-type-specific targets
    target_rate_tensor = torch.zeros(recurrent.topology.num_neurons, device=device)
    for cell_type_name, target_firing_rate in targets.firing_rate.items():
        cell_type_idx = recurrent.cell_types.names.index(cell_type_name)
        mask = cell_type_indices == cell_type_idx
        target_rate_tensor[mask] = target_firing_rate

    firing_rate_loss_fn = FiringRateLoss(
        target_rate=target_rate_tensor, dt=simulation.dt
    )

    # Define loss weights for optimization
    loss_weights = {
        "firing_rate": 1.0,
    }

    # =============
    # Setup Loggers
    # =============

    # Initialize async logger for non-blocking metric logging
    metrics_logger = AsyncLogger(log_dir=output_dir, flush_interval=120.0)

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
                float(np.mean(valid_cvs)) if len(valid_cvs) > 0 else 0.0
            )
            stats[f"cv/{cell_type_name}/std"] = (
                float(np.std(valid_cvs)) if len(valid_cvs) > 0 else 0.0
            )

            # Fraction active
            stats[f"fraction_active/{cell_type_name}"] = float(
                (firing_rates[mask] > 0).mean()
            )

        return stats

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
        loss_functions={"firing_rate": firing_rate_loss_fn},
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
    print(f"Batch size: {training.batch_size}")
    print(f"Log interval: {training.log_interval} chunks")
    print(f"Checkpoint interval: {training.checkpoint_interval} chunks")

    # Run training with the trainer
    best_loss = trainer.train(output_dir=output_dir)

    # ============================================
    # Generate Final Analysis and Save Statistics
    # ============================================

    print("\n" + "=" * 60)
    print("Generating final analysis...")
    print("=" * 60)

    # Run one final forward pass to get fresh data for analysis
    print("Running final simulation...")
    with torch.inference_mode():
        # Reset initial states
        initial_v = None
        initial_g = None
        initial_g_FF = None

        # Collect one batch of simulation data
        spike_dataset_analysis = PoissonSpikeDataset(
            firing_rates=input_firing_rates,
            chunk_size=simulation.chunk_size,
            dt=simulation.dt,
            device=device,
        )

        input_spikes_final = spike_dataset_analysis[0].unsqueeze(0)

        (
            output_spikes_final,
            output_voltages_final,
            output_currents_final,
            output_currents_FF_final,
            output_conductances_final,
            output_conductances_FF_final,
        ) = model.forward(
            input_spikes=input_spikes_final,
            initial_v=initial_v,
            initial_g=initial_g,
            initial_g_FF=initial_g_FF,
        )

        # Move to CPU and convert to numpy
        if device == "cuda":
            output_spikes_final = output_spikes_final.cpu().numpy()
            output_voltages_final = output_voltages_final.cpu().numpy()
            input_spikes_final = input_spikes_final.cpu().numpy()
        else:
            output_spikes_final = output_spikes_final.numpy()
            output_voltages_final = output_voltages_final.numpy()
            input_spikes_final = input_spikes_final.numpy()

    # Create analysis directory
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing statistics and saving to {analysis_dir}...")

    # Compute firing rate statistics
    firing_rate_stats = compute_firing_rate_by_cell_type(
        spike_trains=output_spikes_final,
        cell_type_indices=cell_type_indices,
        duration=params.simulation.chunk_size,
    )

    # Convert to DataFrame and save
    firing_rate_df = pd.DataFrame(
        [
            {
                "cell_type": cell_type_idx,
                "cell_type_name": params.recurrent.cell_types.names[cell_type_idx],
                **stats,
            }
            for cell_type_idx, stats in firing_rate_stats.items()
        ]
    )
    firing_rate_csv_path = analysis_dir / "firing_rate_statistics.csv"
    firing_rate_df.to_csv(firing_rate_csv_path, index=False)
    print(f"  Saved firing rate statistics to {firing_rate_csv_path}")

    # Compute CV statistics
    cv_stats = compute_cv_by_cell_type(
        spike_trains=output_spikes_final,
        cell_type_indices=cell_type_indices,
        dt=params.simulation.dt,
    )

    # Convert to DataFrame and save
    cv_df = pd.DataFrame(
        [
            {
                "cell_type": cell_type_idx,
                "cell_type_name": params.recurrent.cell_types.names[cell_type_idx],
                **stats,
            }
            for cell_type_idx, stats in cv_stats.items()
        ]
    )
    cv_csv_path = analysis_dir / "cv_statistics.csv"
    cv_df.to_csv(cv_csv_path, index=False)
    print(f"  Saved CV statistics to {cv_csv_path}")

    # Compute membrane potential statistics
    voltage_stats = compute_membrane_potential_by_cell_type(
        voltages=output_voltages_final,
        cell_type_indices=cell_type_indices,
    )

    # Convert to DataFrame and save
    voltage_df = pd.DataFrame(
        [
            {
                "cell_type": cell_type_idx,
                "cell_type_name": params.recurrent.cell_types.names[cell_type_idx],
                **stats,
            }
            for cell_type_idx, stats in voltage_stats.items()
        ]
    )
    voltage_csv_path = analysis_dir / "voltage_statistics.csv"
    voltage_df.to_csv(voltage_csv_path, index=False)
    print(f"  Saved voltage statistics to {voltage_csv_path}")

    print("All statistics saved successfully!")

    # ============================================
    # Generate Final Dashboards
    # ============================================

    print("Generating final dashboards...")

    # Create final figures directory
    final_figures_dir = output_dir / "final_figures"
    final_figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate final connectivity dashboard
    final_connectivity_fig = create_connectivity_dashboard(
        connectivity_graph=connectivity_graph,
        weights=model.weights.detach().cpu().numpy(),
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_source_indices,
        cell_type_names=params.recurrent.cell_types.names,
        input_cell_type_names=params.feedforward.cell_types.names,
        num_assemblies=params.recurrent.topology.num_assemblies,
        recurrent_g_bar_by_type=recurrent_g_bar_by_type,
        feedforward_g_bar_by_type=feedforward_g_bar_by_type,
    )

    # For final activity dashboard, run a longer simulation
    print("Running longer simulation for final activity dashboard...")

    # Collect multiple chunks for better statistics
    all_output_spikes_final = []
    all_input_spikes_final = []
    all_output_voltages_final = []
    all_output_currents_final = []
    all_output_currents_FF_final = []
    all_output_conductances_final = []
    all_output_conductances_FF_final = []

    # Reset states
    initial_v = None
    initial_g = None
    initial_g_FF = None

    # Run a few chunks for better visualization
    n_chunks_final = min(5, simulation.num_chunks)

    with torch.inference_mode():
        for chunk_idx in range(n_chunks_final):
            input_spikes_chunk = spike_dataset_analysis[chunk_idx].unsqueeze(0)

            (
                output_spikes_chunk,
                output_voltages_chunk,
                output_currents_chunk,
                output_currents_FF_chunk,
                output_conductances_chunk,
                output_conductances_FF_chunk,
            ) = model.forward(
                input_spikes=input_spikes_chunk,
                initial_v=initial_v,
                initial_g=initial_g,
                initial_g_FF=initial_g_FF,
            )

            # Store final states for next chunk
            initial_v = output_voltages_chunk[:, -1, :].clone()
            initial_g = output_conductances_chunk[:, -1, :, :, :].clone()
            initial_g_FF = output_conductances_FF_chunk[:, -1, :, :, :].clone()

            # Move to CPU and accumulate
            if device == "cuda":
                all_output_spikes_final.append(output_spikes_chunk.cpu())
                all_input_spikes_final.append(input_spikes_chunk.cpu())
                all_output_voltages_final.append(output_voltages_chunk.cpu())
                all_output_currents_final.append(output_currents_chunk.cpu())
                all_output_currents_FF_final.append(output_currents_FF_chunk.cpu())
                all_output_conductances_final.append(output_conductances_chunk.cpu())
                all_output_conductances_FF_final.append(
                    output_conductances_FF_chunk.cpu()
                )
            else:
                all_output_spikes_final.append(output_spikes_chunk)
                all_input_spikes_final.append(input_spikes_chunk)
                all_output_voltages_final.append(output_voltages_chunk)
                all_output_currents_final.append(output_currents_chunk)
                all_output_currents_FF_final.append(output_currents_FF_chunk)
                all_output_conductances_final.append(output_conductances_chunk)
                all_output_conductances_FF_final.append(output_conductances_FF_chunk)

    # Concatenate all chunks
    output_spikes_dashboard = torch.cat(all_output_spikes_final, dim=1).numpy()
    input_spikes_dashboard = torch.cat(all_input_spikes_final, dim=1).numpy()
    output_voltages_dashboard = torch.cat(all_output_voltages_final, dim=1).numpy()
    output_currents_dashboard = torch.cat(all_output_currents_final, dim=1).numpy()
    output_currents_FF_dashboard = torch.cat(
        all_output_currents_FF_final, dim=1
    ).numpy()
    output_conductances_dashboard = torch.cat(
        all_output_conductances_final, dim=1
    ).numpy()
    output_conductances_FF_dashboard = torch.cat(
        all_output_conductances_FF_final, dim=1
    ).numpy()

    # Generate final activity dashboard
    final_activity_fig = create_activity_dashboard(
        output_spikes=output_spikes_dashboard,
        input_spikes=input_spikes_dashboard,
        cell_type_indices=cell_type_indices,
        cell_type_names=params.recurrent.cell_types.names,
        dt=params.simulation.dt,
        voltages=output_voltages_dashboard,
        neuron_types=cell_type_indices,
        neuron_params=neuron_params,
        recurrent_currents=output_currents_dashboard,
        feedforward_currents=output_currents_FF_dashboard,
        recurrent_conductances=output_conductances_dashboard,
        feedforward_conductances=output_conductances_FF_dashboard,
        input_cell_type_names=params.feedforward.cell_types.names,
        recurrent_synapse_names=recurrent_synapse_names,
        feedforward_synapse_names=feedforward_synapse_names,
        window_size=50.0,
        n_neurons_plot=20,
        fraction=1.0,
        random_seed=42,
    )

    # Save final dashboards
    final_connectivity_fig.savefig(
        final_figures_dir / "connectivity_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close(final_connectivity_fig)

    final_activity_fig.savefig(
        final_figures_dir / "activity_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close(final_activity_fig)

    print(f"✓ Saved final dashboard plots to {final_figures_dir}")

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
    print(f"✓ Figures saved to {output_dir / 'figures'}")
    print(f"✓ Metrics saved to {output_dir / 'metrics'}")
    print(f"✓ Analysis saved to {output_dir / 'analysis'}")
    print(f"✓ Best loss achieved: {best_loss:.6f}")

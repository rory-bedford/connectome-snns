"""
Homeostatic plasticity using gradient-based weight optimization.

This script trains a conductance-based LIF network with assembly structure to
achieve target firing rates and spike train statistics through gradient-based
backpropagation. The network learns through multiple loss functions including
firing rate loss, CV loss, silent neuron penalty, subthreshold variance loss,
and recurrent-feedforward balance loss.

This is typically run after network_inference (which can be used for grid search
to find good initial connectivity parameters). The optimized weights from this
stage can then be used as the teacher network for student training.

Overview:
1. Generate initial connectome (or use parameters identified via grid search).
2. Generate Poisson input spike trains.
3. Train network using gradient-based optimization with multiple loss functions
   (firing rate, CV, silent penalty, membrane variance, recurrent-feedforward balance).
4. Save optimized weights and activity patterns for downstream use.
"""

import numpy as np
from synthetic_connectome import (
    topology_generators,
    weight_assigners,
    cell_types,
)
from src.network_inputs.unsupervised import HomogeneousPoissonSpikeDataLoader
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
import torch
from torch.amp import GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from optimisation.loss_functions import (
    FiringRateLoss,
    CVLoss,
    SilentNeuronPenalty,
    SubthresholdVarianceLoss,
    ScalingFactorBalanceLoss,
)
from optimisation.utils import load_checkpoint
from parameter_loaders import HomeostaticPlasticityParams
from training import SNNTrainer
from analysis.firing_statistics import compute_spike_train_cv
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

    # ===================================
    # Create Recurrent Connectivity Graph
    # ===================================

    # Assign cell types to recurrent layer
    cell_type_indices = cell_types.assign_cell_types(
        num_neurons=recurrent.topology.num_neurons,
        cell_type_proportions=recurrent.cell_types.proportion,
        num_assemblies=recurrent.topology.num_assemblies,
    )

    # Generate assembly-based connectivity graph
    connectivity_graph, assembly_ids = topology_generators.assembly_generator(
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
        optimisable="scaling_factors",
        connectome_mask=connectivity_graph,
        feedforward_mask=feedforward_connectivity_graph,
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

    # Create DataLoader with batch_size from parameters
    spike_dataloader = HomogeneousPoissonSpikeDataLoader(
        firing_rates=input_firing_rates,
        chunk_size=simulation.chunk_size,
        dt=simulation.dt,
        batch_size=params.training.batch_size,
        device=device,
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
    weight_ratio_loss_fn = ScalingFactorBalanceLoss(
        target_ratio=targets.weight_ratio,
        excitatory_cell_type=0,
    )

    # Define loss weights from config
    loss_weights = {
        "firing_rate": hyperparameters.loss_weight.firing_rate,
        "cv": hyperparameters.loss_weight.cv,
        "silent_penalty": hyperparameters.loss_weight.silent_penalty,
        "membrane_variance": hyperparameters.loss_weight.membrane_variance,
        "weight_ratio": hyperparameters.loss_weight.weight_ratio,
    }

    # ================================================
    # Create Functions for Plotting and Tracking Stats
    # ================================================

    # Pre-compute static data for plotting functions

    # Neuron parameters for plotting
    neuron_params = params.recurrent.get_neuron_params_for_plotting()

    # Synapse names for plotting
    recurrent_synapse_names = params.recurrent.get_synapse_names()
    feedforward_synapse_names = params.feedforward.get_synapse_names()

    # Define plot generator function that creates both dashboards
    def plot_generator(
        spikes,
        voltages,
        conductances,
        conductances_FF,
        currents,
        currents_FF,
        currents_leak,
        input_spikes,
        weights,
        feedforward_weights,
        connectome_mask,
        feedforward_mask,
        scaling_factors,
        scaling_factors_FF,
    ):
        """Generate connectivity and activity dashboards."""

        # Generate connectivity dashboard
        connectivity_fig = create_connectivity_dashboard(
            weights=weights,
            feedforward_weights=feedforward_weights,
            cell_type_indices=cell_type_indices,
            input_cell_type_indices=input_source_indices,
            cell_type_names=params.recurrent.cell_types.names,
            input_cell_type_names=params.feedforward.cell_types.names,
            connectome_mask=connectome_mask,
            feedforward_mask=feedforward_mask,
            num_assemblies=params.recurrent.topology.num_assemblies,
            scaling_factors=scaling_factors,
            scaling_factors_FF=scaling_factors_FF,
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
            leak_currents=currents_leak,
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
    def stats_computer(spikes, model):
        """Compute summary statistics from network activity.

        Args:
            spikes: Spike array of shape (1, time, neurons) - single batch accumulated over plot_size chunks
            model: The network model

        Returns:
            Dictionary with keys: metric/cell_type/stat
        """
        # Remove batch dimension since we only have 1 batch
        # spikes shape: (1, time, neurons) -> (time, neurons)
        spikes = spikes[0]

        # Compute firing rates per neuron (Hz)
        spike_counts = spikes.sum(axis=0)  # Sum over time: (neurons,)
        duration_s = spikes.shape[0] * params.simulation.dt / 1000.0  # Convert ms to s
        firing_rates = spike_counts / duration_s

        # CV computation on single batch
        cv_values = compute_spike_train_cv(
            spikes[np.newaxis, :, :], dt=params.simulation.dt
        )  # Shape: (1, neurons)
        cv_per_neuron = cv_values[0]  # (neurons,)

        # Compute statistics by cell type
        stats = {}
        for cell_type in np.unique(cell_type_indices):
            mask = cell_type_indices == cell_type
            cell_type_name = params.recurrent.cell_types.names[int(cell_type)]

            # Firing rate statistics (excluding silent neurons)
            cell_firing_rates = firing_rates[mask]
            active_neurons = cell_firing_rates > 0
            active_firing_rates = cell_firing_rates[active_neurons]

            if len(active_firing_rates) > 0:
                stats[f"firing_rate/{cell_type_name}/mean"] = float(
                    active_firing_rates.mean()
                )
                stats[f"firing_rate/{cell_type_name}/std"] = float(
                    active_firing_rates.std()
                )
            else:
                stats[f"firing_rate/{cell_type_name}/mean"] = float("nan")
                stats[f"firing_rate/{cell_type_name}/std"] = float("nan")

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

        # Add scaling factor tracking
        current_recurrent_sf = model["scaling_factors"]
        current_feedforward_sf = model["scaling_factors_FF"]

        # Get cell type names
        target_cell_types = params.recurrent.cell_types.names
        source_ff_cell_types = params.feedforward.cell_types.names

        # Log recurrent scaling factors: source -> target
        for target_idx, target_type in enumerate(target_cell_types):
            for source_idx, source_type in enumerate(target_cell_types):
                synapse_name = f"{source_type}_to_{target_type}"
                stats[f"scaling_factors/recurrent/{synapse_name}"] = float(
                    current_recurrent_sf[source_idx, target_idx]
                )

        # Log feedforward scaling factors: source_ff -> target
        for target_idx, target_type in enumerate(target_cell_types):
            for source_idx, source_type in enumerate(source_ff_cell_types):
                synapse_name = f"{source_type}_to_{target_type}"
                stats[f"scaling_factors/feedforward/{synapse_name}"] = float(
                    current_feedforward_sf[source_idx, target_idx]
                )

        return stats

    # ===================================
    # Save Initial State and Run Inference
    # ===================================

    # Only save initial state and run inference if starting from scratch
    if resume_from is None:
        print("\n" + "=" * 60)
        print("SAVING INITIAL STATE AND RUNNING INFERENCE")
        print("=" * 60)

        # Create initial_state directory
        initial_state_dir = output_dir / "initial_state"
        initial_state_dir.mkdir(parents=True, exist_ok=True)

        # Save initial network structure as single npz file
        print("Saving initial network structure...")
        np.savez(
            initial_state_dir / "network_structure.npz",
            recurrent_weights=model.weights.detach().cpu().numpy(),
            feedforward_weights=model.weights_FF.detach().cpu().numpy(),
            recurrent_connectivity=connectivity_graph,
            feedforward_connectivity=feedforward_connectivity_graph,
            cell_type_indices=cell_type_indices,
            feedforward_cell_type_indices=input_source_indices,
            assembly_ids=assembly_ids,
        )
        print(
            f"✓ Initial network structure saved to {initial_state_dir / 'network_structure.npz'}"
        )

        # =====================================
        # Initial Inference and Visualization
        # =====================================

        # Run 10s inference on single batch (batch_size=1)
        print("\nRunning 10s inference with initial weights...")
        inference_duration_ms = 10000.0  # 10 seconds
        inference_timesteps = int(inference_duration_ms / simulation.dt)

        # Create a new dataloader for 10s inference with batch_size=1
        inference_dataloader = HomogeneousPoissonSpikeDataLoader(
            firing_rates=input_firing_rates,
            chunk_size=inference_timesteps,  # Single chunk of 10s
            dt=simulation.dt,
            batch_size=1,
            device=device,
            shuffle=False,
            num_workers=0,
        )
        inference_input_spikes, _ = next(iter(inference_dataloader))

        # Run inference with tqdm progress bar
        model.use_tqdm = True
        with torch.inference_mode():
            (
                inf_spikes,
                inf_voltages,
                inf_currents,
                inf_currents_FF,
                inf_currents_leak,
                inf_conductances,
                inf_conductances_FF,
            ) = model.forward(
                input_spikes=inference_input_spikes,
                initial_v=None,
                initial_g=None,
                initial_g_FF=None,
            )
        model.use_tqdm = False

        print(f"✓ Inference completed ({inference_duration_ms / 1000:.1f}s simulated)")

        # Generate plots for initial state
        if plot_generator:
            print("Generating initial state plots...")
            figures_dir = initial_state_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            # Convert to numpy and take only first batch
            plot_data = {
                "spikes": inf_spikes[0:1, ...].detach().cpu().numpy(),
                "voltages": inf_voltages[0:1, ...].detach().cpu().numpy(),
                "conductances": inf_conductances[0:1, ...].detach().cpu().numpy(),
                "conductances_FF": inf_conductances_FF[0:1, ...].detach().cpu().numpy(),
                "currents": inf_currents[0:1, ...].detach().cpu().numpy(),
                "currents_FF": inf_currents_FF[0:1, ...].detach().cpu().numpy(),
                "currents_leak": inf_currents_leak[0:1, ...].detach().cpu().numpy(),
                "input_spikes": inference_input_spikes[0:1, ...].detach().cpu().numpy(),
                "weights": model.weights.detach().cpu().numpy(),
                "feedforward_weights": model.weights_FF.detach().cpu().numpy(),
                "connectome_mask": connectivity_graph.astype(np.bool_),
                "feedforward_mask": feedforward_connectivity_graph.astype(np.bool_),
            }

            # Generate plots
            figures = plot_generator(**plot_data)

            # Save plots to disk
            for plot_name, fig in figures.items():
                fig_path = figures_dir / f"{plot_name}.png"
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

            print(f"✓ Initial state plots saved to {figures_dir}")

        # Clean up inference data
        del (
            inference_input_spikes,
            inf_spikes,
            inf_voltages,
            inf_currents,
            inf_currents_FF,
            inf_currents_leak,
            inf_conductances,
            inf_conductances_FF,
        )
        if device == "cuda":
            torch.cuda.empty_cache()

        print("=" * 60 + "\n")

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
            "cv": cv_loss_fn,
            "silent_penalty": silent_penalty_fn,
            "membrane_variance": membrane_variance_loss_fn,
            "weight_ratio": weight_ratio_loss_fn,
        },
        loss_weights=loss_weights,
        params=params,
        device=device,
        wandb_config=wandb_config,
        progress_bar=pbar,
        plot_generator=plot_generator,
        stats_computer=stats_computer,
        connectome_mask=torch.from_numpy(connectivity_graph.astype(np.float32)).to(
            device
        ),
        feedforward_mask=torch.from_numpy(
            feedforward_connectivity_graph.astype(np.float32)
        ).to(device),
    )

    trainer.debug_gradients = False

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

    # ===================================
    # Save Final State and Run Inference
    # ===================================

    print("\n" + "=" * 60)
    print("SAVING FINAL STATE AND RUNNING INFERENCE")
    print("=" * 60)

    # Create final_state directory
    final_state_dir = output_dir / "final_state"
    final_state_dir.mkdir(parents=True, exist_ok=True)

    # Save final network structure as single npz file
    print("Saving final network structure...")

    # Scale weights by scaling factors
    # Scaling factors shape: (n_source_types, n_target_types)
    # Recurrent weights shape: (n_neurons, n_neurons, n_syn_types)
    # We need to apply scaling_factors[source_type, target_type] to weights

    scaled_recurrent_weights = model.weights.detach().cpu().numpy().copy()
    scaled_feedforward_weights = model.weights_FF.detach().cpu().numpy().copy()

    scaling_factors_np = model.scaling_factors.detach().cpu().numpy()
    scaling_factors_FF_np = model.scaling_factors_FF.detach().cpu().numpy()

    # Apply recurrent scaling factors
    for source_idx in range(len(params.recurrent.cell_types.names)):
        source_mask = cell_type_indices == source_idx
        for target_idx in range(len(params.recurrent.cell_types.names)):
            target_mask = cell_type_indices == target_idx
            scale = scaling_factors_np[source_idx, target_idx]
            scaled_recurrent_weights[np.ix_(target_mask, source_mask)] *= scale

    # Apply feedforward scaling factors
    for source_idx in range(len(params.feedforward.cell_types.names)):
        source_mask = input_source_indices == source_idx
        for target_idx in range(len(params.recurrent.cell_types.names)):
            target_mask = cell_type_indices == target_idx
            scale = scaling_factors_FF_np[source_idx, target_idx]
            scaled_feedforward_weights[np.ix_(target_mask, source_mask)] *= scale

    np.savez(
        final_state_dir / "network_structure.npz",
        recurrent_weights=scaled_recurrent_weights,
        feedforward_weights=scaled_feedforward_weights,
        recurrent_connectivity=connectivity_graph,
        feedforward_connectivity=feedforward_connectivity_graph,
        cell_type_indices=cell_type_indices,
        feedforward_cell_type_indices=input_source_indices,
        assembly_ids=assembly_ids,
    )
    print(
        f"✓ Final network structure saved to {final_state_dir / 'network_structure.npz'}"
    )

    # Run 10s inference on single batch (batch_size=1)
    print("\nRunning 10s inference with final weights...")
    inference_duration_ms = 10000.0  # 10 seconds
    inference_timesteps = int(inference_duration_ms / simulation.dt)

    # Create a new dataloader for 10s inference with batch_size=1
    final_inference_dataloader = HomogeneousPoissonSpikeDataLoader(
        firing_rates=input_firing_rates,
        chunk_size=inference_timesteps,  # Single chunk of 10s
        dt=simulation.dt,
        batch_size=1,
        device=device,
        shuffle=False,
        num_workers=0,
    )
    final_inference_input_spikes, _ = next(iter(final_inference_dataloader))

    # Run inference with tqdm progress bar
    model.use_tqdm = True
    with torch.inference_mode():
        (
            final_inf_spikes,
            final_inf_voltages,
            final_inf_currents,
            final_inf_currents_FF,
            final_inf_currents_leak,
            final_inf_conductances,
            final_inf_conductances_FF,
        ) = model.forward(
            input_spikes=final_inference_input_spikes,
            initial_v=None,
            initial_g=None,
            initial_g_FF=None,
        )
    model.use_tqdm = False

    print(f"✓ Inference completed ({inference_duration_ms / 1000:.1f}s simulated)")

    # Generate plots for final state
    if plot_generator:
        print("Generating final state plots...")
        figures_dir = final_state_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Convert to numpy and take only first batch
        plot_data = {
            "spikes": final_inf_spikes[0:1, ...].detach().cpu().numpy(),
            "voltages": final_inf_voltages[0:1, ...].detach().cpu().numpy(),
            "conductances": final_inf_conductances[0:1, ...].detach().cpu().numpy(),
            "conductances_FF": final_inf_conductances_FF[0:1, ...]
            .detach()
            .cpu()
            .numpy(),
            "currents": final_inf_currents[0:1, ...].detach().cpu().numpy(),
            "currents_FF": final_inf_currents_FF[0:1, ...].detach().cpu().numpy(),
            "currents_leak": final_inf_currents_leak[0:1, ...].detach().cpu().numpy(),
            "input_spikes": final_inference_input_spikes[0:1, ...]
            .detach()
            .cpu()
            .numpy(),
            "weights": model.weights.detach().cpu().numpy(),
            "feedforward_weights": model.weights_FF.detach().cpu().numpy(),
        }

        # Generate plots
        figures = plot_generator(**plot_data)

        # Save plots to disk
        for plot_name, fig in figures.items():
            fig_path = figures_dir / f"{plot_name}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        print(f"✓ Final state plots saved to {figures_dir}")

    # Clean up inference data
    del (
        final_inference_input_spikes,
        final_inf_spikes,
        final_inf_voltages,
        final_inf_currents,
        final_inf_currents_FF,
        final_inf_currents_leak,
        final_inf_conductances,
        final_inf_conductances_FF,
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    print("=" * 60 + "\n")

    # Close async logger and flush all remaining data
    if trainer.metrics_logger:
        print("Flushing metrics to disk...")
        trainer.metrics_logger.close()

    # Finish wandb run
    if trainer.wandb_logger is not None:
        wandb.finish()

    print(f"\n✓ Checkpoints: {output_dir / 'checkpoints'}")
    print(f"✓ Figures: {output_dir / 'figures'}")
    print(f"✓ Metrics: {output_dir / 'training_metrics.csv'}")
    # print(f"✓ Initial state: {initial_state_dir / 'network_structure.npz'}")
    print(f"✓ Final state: {final_state_dir / 'network_structure.npz'}")

"""
Training a student network to match teacher activity patterns.

This script trains a connectome-constrained conductance-based LIF network to
reproduce target spike train activity from a pre-generated teacher network.
The student network's connectivity structure is loaded from disk and kept fixed,
while synaptic weights are optimized using gradient-based learning with multiple
loss functions (firing rate matching, van Rossum distance, silent neuron penalty).

Workflow position: Stage 4 (final stage after generate_teacher_activity)

The script loads pre-computed teacher spike trains from zarr storage and trains
the student network to match these target patterns through backpropagation with
surrogate gradients.
"""

import numpy as np
from src.network_inputs.supervised import (
    PrecomputedSpikeDataset,
    CyclicSampler,
)
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from optimisation.loss_functions import (
    VanRossumLoss,
)
from optimisation.utils import load_checkpoint
from parameter_loaders import StudentTrainingParams
from training import SNNTrainer
import toml
import wandb
from visualization.dashboards import (
    create_connectivity_dashboard,
    create_activity_dashboard,
)
from training.weight_perturbers import perturb_weights_scaling_factor
from analysis.firing_statistics import compute_spike_train_cv


def main(
    input_dir,
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
):
    """Train a student network to match teacher spike train activity.

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

    # Load all network parameters from TOML file
    with open(params_file, "r") as f:
        data = toml.load(f)
    params = StudentTrainingParams(**data)

    # Extract commonly used parameter groups
    simulation = params.simulation
    training = params.training
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

    # ================================
    # Load Network Structure from Disk
    # ================================

    network_structure = np.load(input_dir / "network_structure.npz")

    # Extract network components
    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    feedforward_cell_type_indices = network_structure["feedforward_cell_type_indices"]
    connectivity_graph = network_structure["recurrent_connectivity"]
    feedforward_connectivity_graph = network_structure["feedforward_connectivity"]

    # Apply Weight Perturbations to get target scaling factors
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
        optimisable=training.optimisable,
    )

    # Create targets directory if it doesn't exist
    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    # Save target scaling factors (only save non-None targets)
    target_data = {}
    if target_scaling_factors is not None:
        target_data["recurrent_scaling_factors"] = target_scaling_factors
    if target_feedforward_scaling_factors is not None:
        target_data["feedforward_scaling_factors"] = target_feedforward_scaling_factors

    np.savez(targets_dir / "target_scaling_factors.npz", **target_data)

    # ======================
    # Load Dataset from Disk
    # ======================

    # Load precomputed spike data from zarr
    spike_dataset = PrecomputedSpikeDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=simulation.chunk_size,
        device=device,
    )

    batch_size = spike_dataset.batch_size

    print(f"✓ Loaded {spike_dataset.num_chunks} chunks × {batch_size} batch size")
    # DataLoader without additional batching (data is already batched)

    # Result shape per iteration: (batch_size, chunk_size, n_neurons)
    # Use CyclicSampler to enable infinite iteration for multi-epoch training
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,  # No additional batching
        sampler=CyclicSampler(spike_dataset),
        num_workers=0,
    )

    # ======================
    # Initialize LIF Network
    # ======================

    # Initialize conductance-based LIF network model with initial scaling factors from parameters
    model = ConductanceLIFNetwork(
        dt=spike_dataset.dt,
        weights=weights,
        cell_type_indices=cell_type_indices,
        cell_params=recurrent.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        surrgrad_scale=hyperparameters.surrgrad_scale,
        weights_FF=feedforward_weights,
        cell_type_indices_FF=feedforward_cell_type_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        scaling_factors=np.array(params.scaling_factors["recurrent"]),
        scaling_factors_FF=np.array(params.scaling_factors["feedforward"]),
        optimisable=training.optimisable,
        use_tqdm=False,  # Disable tqdm progress bar for training loop
    )

    # Move model to device for GPU acceleration
    model.to(device)

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
    if target_scaling_factors is not None:
        print(f"\nTarget recurrent scaling factors:\n{target_scaling_factors}")
    if target_feedforward_scaling_factors is not None:
        print(
            f"\nTarget feedforward scaling factors:\n{target_feedforward_scaling_factors}"
        )
    print("=" * len(section) + "\n")

    # ==============================
    # Setup Optimiser and DataLoader
    # ==============================

    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    # Mixed precision training scaler (only used if enabled)
    scaler = GradScaler("cuda", enabled=training.mixed_precision and device == "cuda")

    # =====================
    # Define Loss Functions
    # =====================

    van_rossum_loss_fn = VanRossumLoss(
        tau=hyperparameters.van_rossum_tau,
        dt=spike_dataset.dt,
        window_size=simulation.chunk_size,
        device=device,
    )

    # Define loss weights from config
    loss_weights = {
        "van_rossum": hyperparameters.loss_weight.van_rossum,
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
    ):
        """Generate connectivity and activity dashboards."""
        # Calculate mean membrane potential by cell type from voltage traces
        recurrent_V_mem_by_type = {}
        for i, cell_type_name in enumerate(params.recurrent.cell_types.names):
            cell_mask = cell_type_indices == i
            if cell_mask.sum() > 0:
                # Average over batch, time, and neurons of this type
                recurrent_V_mem_by_type[cell_type_name] = float(
                    voltages[:, :, cell_mask].mean()
                )

        # Generate connectivity dashboard
        connectivity_fig = create_connectivity_dashboard(
            weights=weights,
            feedforward_weights=feedforward_weights,
            cell_type_indices=cell_type_indices,
            input_cell_type_indices=feedforward_cell_type_indices,
            cell_type_names=params.recurrent.cell_types.names,
            input_cell_type_names=params.feedforward.cell_types.names,
            connectome_mask=connectome_mask,
            feedforward_mask=feedforward_mask,
            plot_fraction_recurrent=0.1,
            plot_fraction_feedforward=0.1,
        )

        # Generate activity dashboard
        activity_fig = create_activity_dashboard(
            output_spikes=spikes,
            input_spikes=input_spikes,
            cell_type_indices=cell_type_indices,
            cell_type_names=params.recurrent.cell_types.names,
            dt=spike_dataset.dt,
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

    # Define stats computer function (captures model and target scaling factors in closure)
    def stats_computer(spikes, model_snapshot):
        """Compute summary statistics from network activity.

        Args:
            spikes: Spike array of shape (1, time, neurons) - single batch accumulated over plot_size chunks
            model_snapshot: Dictionary containing model parameters as numpy arrays
                           (keys: "scaling_factors", "scaling_factors_FF")

        Returns:
            Dictionary with keys: metric/stat/cell_type and scaling_factors/layer/metric/connection
        """
        # Remove batch dimension since we only have 1 batch
        # spikes shape: (1, time, neurons) -> (time, neurons)
        spikes = spikes[0]

        # Compute firing rates per neuron (Hz)
        spike_counts = spikes.sum(axis=0)  # Sum over time: (neurons,)
        duration_s = spikes.shape[0] * spike_dataset.dt / 1000.0  # Convert ms to s
        firing_rates = spike_counts / duration_s

        # CV computation on single batch
        cv_values = compute_spike_train_cv(
            spikes[np.newaxis, :, :], dt=spike_dataset.dt
        )  # Shape: (1, neurons)
        cv_per_neuron = cv_values[0, :]  # (neurons,)

        # Compute statistics by cell type
        stats = {}
        for cell_type in np.unique(cell_type_indices):
            mask = cell_type_indices == cell_type
            cell_type_name = params.recurrent.cell_types.names[int(cell_type)]

            # Firing rate statistics
            stats[f"firing_rate/mean/{cell_type_name}"] = float(
                firing_rates[mask].mean()
            )
            stats[f"firing_rate/std/{cell_type_name}"] = float(firing_rates[mask].std())

            # CV statistics (only for neurons with valid CVs)
            cell_cvs = cv_per_neuron[mask]
            valid_cvs = cell_cvs[~np.isnan(cell_cvs)]
            stats[f"cv/mean/{cell_type_name}"] = (
                float(np.mean(valid_cvs)) if len(valid_cvs) > 0 else float("nan")
            )
            stats[f"cv/std/{cell_type_name}"] = (
                float(np.std(valid_cvs)) if len(valid_cvs) > 0 else float("nan")
            )

            # Fraction active
            stats[f"fraction_active/{cell_type_name}"] = float(
                (firing_rates[mask] > 0).mean()
            )

        # Add scaling factor tracking
        # Scaling factors are 2D: (n_source_types, n_target_types)
        current_recurrent_sf = model_snapshot["scaling_factors"]
        current_feedforward_sf = model_snapshot["scaling_factors_FF"]

        # Get cell type names
        target_cell_types = params.recurrent.cell_types.names
        source_ff_cell_types = params.feedforward.cell_types.names

        # Log recurrent scaling factors: source -> target (only if being optimized)
        if target_scaling_factors is not None:
            for target_idx, target_type in enumerate(target_cell_types):
                for source_idx, source_type in enumerate(target_cell_types):
                    synapse_name = f"{source_type}_to_{target_type}"
                    stats[f"scaling_factors/recurrent/{synapse_name}/value"] = float(
                        current_recurrent_sf[source_idx, target_idx]
                    )
                    stats[f"scaling_factors/recurrent/{synapse_name}/target"] = float(
                        target_scaling_factors[source_idx, target_idx]
                    )

        # Log feedforward scaling factors: source_ff -> target (only if being optimized)
        if target_feedforward_scaling_factors is not None:
            for target_idx, target_type in enumerate(target_cell_types):
                for source_idx, source_type in enumerate(source_ff_cell_types):
                    synapse_name = f"{source_type}_to_{target_type}"
                    stats[f"scaling_factors/feedforward/{synapse_name}/value"] = float(
                        current_feedforward_sf[source_idx, target_idx]
                    )
                    stats[f"scaling_factors/feedforward/{synapse_name}/target"] = float(
                        target_feedforward_scaling_factors[source_idx, target_idx]
                    )

        return stats

    # ====================================
    # Save Initial State and Run Inference
    # ====================================

    # Only save initial state and run inference if starting from scratch
    if resume_from is None:
        section = "Running 10s inference with initial scaling factors..."
        print("\n" + "=" * len(section))
        print(section)
        print("=" * len(section))

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
            feedforward_cell_type_indices=feedforward_cell_type_indices,
        )
        print(
            f"✓ Initial network structure saved to {initial_state_dir / 'network_structure.npz'}"
        )

        # Run inference for plot_size chunks
        print(
            f"\nRunning inference with initial weights ({training.plot_size} chunks)..."
        )
        inference_duration_ms = (
            training.plot_size * simulation.chunk_size * spike_dataset.dt
        )
        inference_timesteps = int(inference_duration_ms / spike_dataset.dt)

        # Calculate number of chunks needed
        chunks_needed = int(np.ceil(inference_timesteps / spike_dataset.chunk_size))

        # Collect input and target spikes from precomputed dataset (batch=0)
        inference_input_spikes_list = []
        for chunk_idx in range(chunks_needed):
            input_chunk, target_chunk = spike_dataset[chunk_idx]
            # Dataset returns (batch, time, neurons)
            # Extract batch 0: (time, neurons)
            inference_input_spikes_list.append(input_chunk[0, :, :])

        # Concatenate chunks and truncate to exact duration
        inference_input_spikes = torch.cat(inference_input_spikes_list, dim=0)[
            :inference_timesteps, :
        ]
        # Add batch dimension for model: (1, time, n_neurons)
        inference_input_spikes = inference_input_spikes.unsqueeze(0)

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

            # Convert to numpy and take only first batch (keep batch dimension)
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
                "connectome_mask": connectivity_graph,
                "feedforward_mask": feedforward_connectivity_graph,
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

    resume_from = None

    # ===================
    # Setup Training Loop
    # ===================

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
            "van_rossum": van_rossum_loss_fn,
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

    # Set debug flag from config
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
    print(f"Epochs: {training.epochs}")
    print(f"Batch size: {batch_size}")
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

    # Run 10s inference on single batch
    print("\nRunning 10s inference with final weights...")
    inference_duration_ms = 10000.0  # 10 seconds
    inference_timesteps = int(inference_duration_ms / spike_dataset.dt)

    # Calculate number of chunks needed for 10s
    chunks_needed = int(np.ceil(inference_timesteps / spike_dataset.chunk_size))

    # Collect input spikes from precomputed dataset (batch 0, pattern 0)
    final_inference_input_spikes_list = []
    for chunk_idx in range(chunks_needed):
        input_chunk, target_chunk = spike_dataset[chunk_idx]
        # Dataset returns (batch, patterns, time, neurons)
        # Extract batch 0, pattern 0: (time, neurons)
        final_inference_input_spikes_list.append(input_chunk[0, 0, :, :])

    # Concatenate chunks and truncate to exact duration
    final_inference_input_spikes = torch.cat(final_inference_input_spikes_list, dim=0)[
        :inference_timesteps, :
    ]
    # Add batch dimension for model: (1, time, n_neurons)
    final_inference_input_spikes = final_inference_input_spikes.unsqueeze(0)

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

        # Convert to numpy and take only first batch (keep batch dimension)
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
    print(f"✓ Initial state: {initial_state_dir / 'network_structure.npz'}")
    print(f"✓ Final state: {final_state_dir / 'network_structure.npz'}")

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
    CyclicSampler,
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
import matplotlib.pyplot as plt
from training.weight_perturbers import perturb_weights_scaling_factor
from visualization.dashboards import (
    create_connectivity_dashboard,
    create_activity_dashboard,
)


def main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None):
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
    # Use CyclicSampler to enable infinite iteration for multi-epoch training
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,  # No additional batching
        sampler=CyclicSampler(spike_dataset),
        num_workers=0,
    )

    # =======================
    # Initialize LIF Network
    # =======================

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

    # ================
    # Setup Optimiser
    # ================

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

    # ==============
    # Setup Loggers
    # ==============

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

    # ==================================================
    # Create Functions for Plotting and Tracking Stats
    # ==================================================

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
        currents_leak,
        input_spikes,
        weights,
        feedforward_weights,
    ):
        """Generate connectivity and activity dashboards."""
        # Generate connectivity dashboard
        connectivity_fig = create_connectivity_dashboard(
            weights=weights,
            feedforward_weights=feedforward_weights,
            cell_type_indices=cell_type_indices,
            input_cell_type_indices=feedforward_cell_type_indices,
            cell_type_names=params.recurrent.cell_types.names,
            input_cell_type_names=params.feedforward.cell_types.names,
            recurrent_g_bar_by_type=recurrent_g_bar_by_type,
            feedforward_g_bar_by_type=feedforward_g_bar_by_type,
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

        # Add scaling factor tracking
        current_recurrent_sf = model.scaling_factors.detach().cpu().numpy()
        current_feedforward_sf = model.scaling_factors_FF.detach().cpu().numpy()

        # Log recurrent scaling factors
        for i, synapse_name in enumerate(recurrent_synapse_names):
            stats[f"scaling_factors/recurrent/{synapse_name}/current"] = float(
                current_recurrent_sf[i]
            )
            stats[f"scaling_factors/recurrent/{synapse_name}/target"] = float(
                target_scaling_factors[i]
            )

        # Log feedforward scaling factors
        for i, synapse_name in enumerate(feedforward_synapse_names):
            stats[f"scaling_factors/feedforward/{synapse_name}/current"] = float(
                current_feedforward_sf[i]
            )
            stats[f"scaling_factors/feedforward/{synapse_name}/target"] = float(
                target_feedforward_scaling_factors[i]
            )

        return stats

    # Only run initial inference if starting from scratch
    if resume_from is None:
        section = "Running 10s inference with initial scaling factors..."
        print("\n" + "=" * len(section))
        print(section)
        print("=" * len(section))

        # Create initial_state directory
        initial_state_dir = output_dir / "initial_state"
        initial_state_dir.mkdir(parents=True, exist_ok=True)

        # Run 10s inference using precomputed data (batch 1, pattern 1)
        inference_duration_ms = 10000.0  # 10 seconds
        inference_timesteps = int(inference_duration_ms / spike_dataset.dt)

        # Calculate number of chunks needed for 10s
        chunks_needed = int(np.ceil(inference_timesteps / spike_dataset.chunk_size))

        # Collect input and target spikes from precomputed dataset (batch=0, pattern=0)
        inference_input_spikes_list = []
        inference_target_spikes_list = []
        for chunk_idx in range(chunks_needed):
            input_chunk, target_chunk = spike_dataset[
                chunk_idx
            ]  # Returns tuple of tensors
            # Each has shape: (batch_size, n_patterns, chunk_size, n_neurons)
            # Extract batch 0, pattern 0 (first indices)
            inference_input_spikes_list.append(
                input_chunk[0, 0, :, :]
            )  # Shape: (chunk_size, n_input_neurons)
            inference_target_spikes_list.append(
                target_chunk[0, 0, :, :]
            )  # Shape: (chunk_size, n_neurons)

        # Concatenate chunks and truncate to exact duration
        inference_input_spikes = torch.cat(inference_input_spikes_list, dim=0)[
            :inference_timesteps, :
        ]
        inference_target_spikes = torch.cat(inference_target_spikes_list, dim=0)[
            :inference_timesteps, :
        ]

        # Add batch dimension: (1, time, n_neurons)
        inference_input_spikes = inference_input_spikes.unsqueeze(0)
        inference_target_spikes = inference_target_spikes.unsqueeze(0)

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
            inference_target_spikes,
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

        print("=" * len(section) + "\n")

    # ====================
    # Setup Training Loop
    # ====================

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
        plot_generator=plot_generator,
        stats_computer=stats_computer,
        connectome_mask=torch.from_numpy(connectivity_graph.astype(np.float32)).to(
            device
        ),
        feedforward_mask=torch.from_numpy(
            feedforward_connectivity_graph.astype(np.float32)
        ).to(device),
    )

    # ==================
    # Run Training Loop
    # ==================

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

    # =========
    # Clean Up
    # =========

    section = "TRAINING COMPLETE"
    print("\n" + "=" * len(section))
    print(section)
    print("=" * len(section))
    print(f"Best loss achieved: {best_loss:.6f}")

    # ====================================
    # Save Final State and Run Inference
    # ====================================

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

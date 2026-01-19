"""
EM-like training of hidden units in a partially observable spiking neural network.

This script implements an EM-like algorithm for training hidden units:
1. E-step (Inference): Run the recurrent network with current scaling factors to infer hidden unit activities
2. M-step (Training): Train the feedforward network to match visible unit teacher activity using inferred hidden unit inputs

Key insight: Loss is computed on visible neurons only - we're matching visible units to teacher activity
while iteratively refining hidden unit estimates through the EM iterations.
"""

import numpy as np
from pathlib import Path
from dataloaders.supervised import (
    PrecomputedSpikeDataset,
    CyclicSampler,
    SpikeData,
)
from network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from network_simulators.conductance_based.simulator import (
    ConductanceLIFNetwork,
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
    EMTrainingConfig,
    StudentHyperparameters,
)
from configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig
from snn_runners import SNNTrainer
from snn_runners.inference_runner import SNNInference
import toml
import wandb
import zarr
from visualization.neuronal_dynamics import plot_spike_trains


def make_em_collate_fn(
    visible_indices: np.ndarray,
    hidden_indices: np.ndarray,
    inferred_spikes_zarr_path: Path,
    chunk_size: int,
    n_neurons_full: int,
):
    """
    Collate function for EM training that reads inferred spikes from zarr.

    For each chunk:
    - Inputs: FF spikes + visible teacher spikes + inferred hidden spikes (from zarr)
    - Targets: visible teacher spikes ONLY (loss computed on visible neurons)

    Args:
        visible_indices: Array of indices for visible neurons
        hidden_indices: Array of indices for hidden neurons
        inferred_spikes_zarr_path: Path to zarr file with inferred spikes
        chunk_size: Number of timesteps per chunk
        n_neurons_full: Total number of neurons in the full network
    """
    visible_tensor = torch.from_numpy(visible_indices).long()
    hidden_tensor = torch.from_numpy(hidden_indices).long()
    chunk_counter = [0]  # Mutable to track chunk index

    # Open zarr file for reading
    zarr_root = zarr.open_group(inferred_spikes_zarr_path, mode="r")
    inferred_spikes_zarr = zarr_root["output_spikes"]

    def em_collate_fn(batch: SpikeData) -> SpikeData:
        batch_size, time_steps, _ = batch.target_spikes.shape
        device = batch.target_spikes.device

        # Build full recurrent input
        recurrent_inputs = torch.zeros(
            batch_size,
            time_steps,
            n_neurons_full,
            device=device,
            dtype=torch.float32,
        )

        # Visible: exact teacher spikes
        recurrent_inputs[:, :, visible_tensor] = batch.target_spikes[
            :, :, visible_tensor
        ].float()

        # Hidden: inferred spikes from zarr file
        start_t = chunk_counter[0] * time_steps
        end_t = start_t + time_steps
        hidden_spikes_chunk = (
            torch.from_numpy(
                np.array(inferred_spikes_zarr[:, start_t:end_t, hidden_indices])
            )
            .float()
            .to(device)
        )
        recurrent_inputs[:, :, hidden_tensor] = hidden_spikes_chunk

        chunk_counter[0] += 1

        # Concatenate FF + recurrent as inputs
        concatenated_inputs = torch.cat([batch.input_spikes, recurrent_inputs], dim=2)

        # Target: VISIBLE neurons only (from teacher)
        visible_targets = batch.target_spikes[:, :, visible_tensor]

        return SpikeData(
            input_spikes=concatenated_inputs,
            target_spikes=visible_targets,
        )

    def reset_counter():
        chunk_counter[0] = 0

    return em_collate_fn, reset_counter


def transfer_scaling_factors(
    feedforward_model: FeedforwardConductanceLIFNetwork,
    recurrent_model: ConductanceLIFNetwork,
    n_ff_cell_types: int,
):
    """
    Transfer learned scaling factors from feedforward to recurrent model.

    The feedforward model has concatenated scaling factors [FF part, recurrent part].
    Split them and update the recurrent model's buffers.

    Args:
        feedforward_model: Trained feedforward model with learned scaling_factors_FF
        recurrent_model: Recurrent model to update (has optimisable=None)
        n_ff_cell_types: Number of feedforward cell types
    """
    # Get learned concatenated scaling factors
    learned_sf = feedforward_model.scaling_factors_FF.detach().cpu().numpy()

    # Split: [FF part, recurrent part]
    new_sf_FF = learned_sf[:n_ff_cell_types, :]  # (n_ff_types, n_rec_types)
    new_sf_rec = learned_sf[n_ff_cell_types:, :]  # (n_rec_types, n_rec_types)

    # Update recurrent model buffers (it has optimisable=None)
    with torch.no_grad():
        recurrent_model._buffers["_scaling_factors_FF_buffer"] = torch.from_numpy(
            new_sf_FF.astype(np.float32)
        ).to(recurrent_model.device)

        recurrent_model._buffers["_scaling_factors_buffer"] = torch.from_numpy(
            new_sf_rec.astype(np.float32)
        ).to(recurrent_model.device)

    # CRITICAL: Recompute weight caches with new scaling factors
    recurrent_model.set_timestep(float(recurrent_model.dt))


def main(
    input_dir,
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
):
    """Train hidden units using EM-like algorithm.

    E-step: Run recurrent network to infer hidden unit activities
    M-step: Train feedforward network on visible units using inferred hidden inputs

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(params_file, "r") as f:
        data = toml.load(f)

    # Load configuration sections
    simulation = StudentSimulationConfig(**data["simulation"])
    training = EMTrainingConfig(**data["training"])
    hyperparameters = StudentHyperparameters(**data["hyperparameters"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])
    scaling_factors = data.get("scaling_factors", {})
    em_config = data.get("em", {})

    # Extract EM parameters
    total_iterations = em_config.get("total_iterations", 10)
    epochs_per_update = em_config.get("epochs_per_update", 5)

    # Extract parameters into plain Python variables
    chunk_size = simulation.chunk_size
    seed = simulation.seed
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
    van_rossum_tau = hyperparameters.van_rossum_tau
    loss_weight_van_rossum = hyperparameters.loss_weight.van_rossum

    # Load hidden cell fraction
    hidden_cell_fraction = data["simulation"].get("hidden_cell_fraction", 0.5)

    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")

    # ================================
    # Load Network Structure from Disk
    # ================================

    network_structure = np.load(input_dir / "network_structure.npz")

    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    feedforward_cell_type_indices = network_structure["feedforward_cell_type_indices"]
    recurrent_mask = network_structure["recurrent_connectivity"]
    feedforward_mask = network_structure["feedforward_connectivity"]

    n_neurons_full = weights.shape[0]
    n_feedforward = feedforward_weights.shape[0]

    # ================================
    # Select Hidden and Visible Neurons
    # ================================

    n_hidden = int(n_neurons_full * hidden_cell_fraction)
    n_visible = n_neurons_full - n_hidden

    if n_hidden > 0:
        all_indices = np.arange(n_neurons_full)
        np.random.shuffle(all_indices)
        hidden_indices = np.sort(all_indices[:n_hidden])
        visible_indices = np.sort(all_indices[n_hidden:])
    else:
        hidden_indices = np.array([], dtype=int)
        visible_indices = np.arange(n_neurons_full)

    print("\n✓ Hidden cell configuration:")
    print(f"  - Total neurons: {n_neurons_full}")
    print(f"  - Hidden fraction: {hidden_cell_fraction:.1%}")
    print(f"  - Hidden neurons: {n_hidden}")
    print(f"  - Visible neurons: {n_visible}")

    # ===============================================
    # Get Cell and Synapse Parameters
    # ===============================================

    recurrent_cell_params = recurrent.get_cell_params()
    feedforward_cell_params = feedforward.get_cell_params()

    n_ff_cell_types = len(feedforward_cell_params)
    n_rec_cell_types = len(recurrent_cell_params)

    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()

    n_ff_synapse_types = len(feedforward_synapse_params)

    # ============================================
    # Concatenate and Perturb Scaling Factors
    # ============================================

    sf_feedforward = np.array(scaling_factors["feedforward"])
    sf_recurrent = np.array(scaling_factors["recurrent"])

    # Concatenate vertically: feedforward on top, recurrent on bottom
    concatenated_scaling_factors = np.concatenate(
        [sf_feedforward, sf_recurrent], axis=0
    )

    # Apply weight perturbation to create target scaling factors
    sigma = np.sqrt(weight_perturbation_variance)
    mu = -(sigma**2) / 2.0

    perturbation_factors = np.random.lognormal(
        mean=mu, sigma=sigma, size=concatenated_scaling_factors.shape
    )

    # Target scaling factors are reciprocals of perturbation
    target_scaling_factors_FF = 1.0 / perturbation_factors

    # ===============================================
    # Construct Combined Input Cell Types and Parameters
    # ===============================================

    # Concatenate cell type indices: feedforward + recurrent (with offset)
    concatenated_cell_type_indices = np.concatenate(
        [feedforward_cell_type_indices, cell_type_indices + n_ff_cell_types]
    )

    # Concatenate cell params with offset cell_ids
    combined_cell_params_FF = feedforward_cell_params.copy()
    for cell_params in recurrent_cell_params:
        offset_cell_params = cell_params.copy()
        offset_cell_params["cell_id"] = cell_params["cell_id"] + n_ff_cell_types
        combined_cell_params_FF.append(offset_cell_params)

    # Concatenate synapse params with offset indices
    combined_synapse_params_FF = feedforward_synapse_params.copy()
    for syn_params in recurrent_synapse_params:
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

    # ===============================================
    # Perturb Weights (Same Perturbation for Both Models)
    # ===============================================

    # Apply perturbation to ALL weights (full recurrent network)
    n_total_inputs = n_feedforward + n_neurons_full
    concatenated_weights_full = np.concatenate([feedforward_weights, weights], axis=0)

    perturbed_weights_full = concatenated_weights_full.copy()
    for input_idx in range(n_total_inputs):
        input_type = concatenated_cell_type_indices[input_idx]
        for output_idx in range(n_neurons_full):
            output_type = cell_type_indices[output_idx]
            perturbed_weights_full[input_idx, output_idx] *= perturbation_factors[
                input_type, output_type
            ]

    # Split back into FF and recurrent weights
    perturbed_ff_weights = perturbed_weights_full[:n_feedforward, :]
    perturbed_rec_weights = perturbed_weights_full[n_feedforward:, :]

    # ===============================================
    # Filter Weights to Visible Outputs Only (For Feedforward Model)
    # ===============================================

    # Perturbed weights filtered to visible outputs
    perturbed_weights_to_visible = perturbed_rec_weights[:, visible_indices]
    perturbed_ff_weights_to_visible = perturbed_ff_weights[:, visible_indices]

    # Concatenate for feedforward model
    concatenated_weights_visible = np.concatenate(
        [perturbed_ff_weights_to_visible, perturbed_weights_to_visible], axis=0
    )

    concatenated_mask_visible = np.concatenate(
        [feedforward_mask[:, visible_indices], recurrent_mask[:, visible_indices]],
        axis=0,
    )

    cell_type_indices_visible = cell_type_indices[visible_indices]

    print("\n✓ Feedforward network setup (all inputs -> visible outputs):")
    print(f"  - Output neurons: {n_visible} (visible only)")
    print(
        f"  - Total inputs per neuron: {n_total_inputs} ({n_feedforward} FF + {n_neurons_full} recurrent)"
    )
    print(f"  - Weight matrix shape: {concatenated_weights_visible.shape}")

    # Save targets and hidden neuron info
    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        targets_dir / "target_scaling_factors.npz",
        feedforward_scaling_factors=target_scaling_factors_FF,
    )

    np.savez(
        targets_dir / "hidden_neurons.npz",
        hidden_indices=hidden_indices,
        visible_indices=visible_indices,
        hidden_cell_fraction=hidden_cell_fraction,
        n_neurons_full=n_neurons_full,
    )

    # ======================
    # Load Dataset from Disk
    # ======================

    spike_dataset = PrecomputedSpikeDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
    )

    batch_size = spike_dataset.batch_size
    dt = spike_dataset.dt
    num_chunks = spike_dataset.num_chunks

    print(f"\n✓ Loaded {num_chunks} chunks × {batch_size} batch size")

    # ======================================
    # Initialize RECURRENT Model (Inference)
    # ======================================

    print("\n✓ Initializing recurrent model for inference (E-step)...")

    recurrent_model = ConductanceLIFNetwork(
        dt=dt,
        weights=perturbed_rec_weights,
        weights_FF=perturbed_ff_weights,
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=feedforward_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=feedforward_cell_params,
        synapse_params=recurrent_synapse_params,
        synapse_params_FF=feedforward_synapse_params,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        scaling_factors=sf_recurrent.copy(),  # Start with unperturbed
        scaling_factors_FF=sf_feedforward.copy(),  # Start with unperturbed
        optimisable=None,  # No optimization - just inference
        connectome_mask=recurrent_mask,
        feedforward_mask=feedforward_mask,
        track_variables=False,
        use_tqdm=False,
    )
    recurrent_model.to(device)

    print(f"  - Output neurons: {n_neurons_full}")
    print(f"  - {n_total_inputs} feedforward inputs per neuron")

    # ==============================================
    # Initialize FEEDFORWARD Model (Visible Outputs)
    # ==============================================

    print("\n✓ Initializing feedforward model for training (M-step)...")

    feedforward_model = FeedforwardConductanceLIFNetwork(
        dt=dt,
        weights_FF=concatenated_weights_visible,
        cell_type_indices=cell_type_indices_visible,
        cell_type_indices_FF=concatenated_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        scaling_factors_FF=concatenated_scaling_factors.copy(),
        optimisable=optimisable,
        feedforward_mask=concatenated_mask_visible,
        track_variables=False,
        use_tqdm=False,
    )
    feedforward_model.to(device)

    print(f"  - Output neurons: {n_visible} (visible only)")
    print(f"  - {n_total_inputs} feedforward inputs per neuron")
    print(f"  - Scaling factors shape: {feedforward_model.scaling_factors_FF.shape}")

    # ==============================
    # Setup Loss Function
    # ==============================

    van_rossum_loss_fn = VanRossumLoss(
        tau=van_rossum_tau,
        dt=dt,
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
        """Generate spike train comparison plot for subset of visible neurons."""
        n_plot = min(10, spikes.shape[2])

        interleaved = np.zeros((1, spikes.shape[1], 2 * n_plot))
        for i in range(n_plot):
            interleaved[0, :, 2 * i] = target_spikes[0, :, i]
            interleaved[0, :, 2 * i + 1] = spikes[0, :, i]

        cell_type_indices_plot = np.array([0, 1] * n_plot)
        cell_type_names_plot = ["Target", "Trained"]

        fig = plot_spike_trains(
            spikes=interleaved,
            dt=dt,
            cell_type_indices=cell_type_indices_plot,
            cell_type_names=cell_type_names_plot,
            n_neurons_plot=2 * n_plot,
            fraction=1.0,
            random_seed=None,
            title=f"EM Training: Target vs Trained (first {n_plot} visible neurons, {hidden_cell_fraction:.0%} hidden)",
            ylabel="Neuron",
            figsize=(14, 8),
        )

        return {"spike_comparison": fig}

    # ================================================
    # Create Stats Computer Function
    # ================================================

    def stats_computer(spikes, model_snapshot):
        """Compute summary statistics for visible neurons."""
        spikes_np = spikes[0, :, :]

        spike_counts = spikes_np.sum(axis=0)
        duration_s = spikes_np.shape[0] * dt / 1000.0
        firing_rates = spike_counts / duration_s

        stats = {
            "firing_rate/mean": float(firing_rates.mean()),
            "firing_rate/std": float(firing_rates.std()),
            "firing_rate/min": float(firing_rates.min()),
            "firing_rate/max": float(firing_rates.max()),
            "hidden_cells/fraction": hidden_cell_fraction,
            "hidden_cells/n_hidden": n_hidden,
            "hidden_cells/n_visible": n_visible,
        }

        current_sf = model_snapshot["scaling_factors_FF"]
        target_sf = target_scaling_factors_FF

        input_cell_type_names = (
            feedforward.cell_types.names + recurrent.cell_types.names
        )
        output_cell_type_names = recurrent.cell_types.names

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
    # EM Training Loop
    # ===================

    print("\n" + "=" * 60)
    print("Starting EM training:")
    print(f"  - Total EM iterations: {total_iterations}")
    print(f"  - Epochs per M-step: {epochs_per_update}")
    print(f"  - Chunks per epoch: {num_chunks}")
    print(f"  - Total chunks per M-step: {epochs_per_update * num_chunks}")
    print("=" * 60)

    # Track current EM iteration for stats_computer (mutable container for closure)
    current_em_iter = [0]

    # Wrap stats_computer to include EM iteration
    def stats_computer_with_em(spikes, model_snapshot):
        stats = stats_computer(spikes, model_snapshot)
        stats["em/iteration"] = current_em_iter[0] + 1
        return stats

    # Initialize wandb if config provided
    wandb_logger = None
    if wandb_config:
        wandb_config_dict = {
            **wandb_config,
            "config": {
                "simulation": simulation.model_dump(),
                "training": training.model_dump(),
                "hyperparameters": hyperparameters.model_dump(),
                "recurrent": recurrent.model_dump(),
                "feedforward": feedforward.model_dump(),
                "scaling_factors": scaling_factors,
                "em": em_config,
                "hidden_cell_fraction": hidden_cell_fraction,
                "n_hidden": n_hidden,
                "n_visible": n_visible,
                "output_dir": str(output_dir),
                "device": device,
            },
        }
        wandb_logger = wandb.init(
            name=output_dir.name,
            dir=str(output_dir),
            **wandb_config_dict,
        )

    best_loss_overall = float("inf")
    global_chunk_counter = 0

    for em_iter in range(total_iterations):
        # Update EM iteration counter for stats logging
        current_em_iter[0] = em_iter

        print(f"\n{'=' * 60}")
        print(f"EM Iteration {em_iter + 1}/{total_iterations}")
        print("=" * 60)

        # ===== E-STEP: INFERENCE =====
        print("\n--- E-Step: Running inference to get hidden unit activities ---")
        recurrent_model.reset_state(batch_size=batch_size)

        # Save inferred spikes to zarr for M-step
        inferred_spikes_path = output_dir / f"em_iter_{em_iter}_inferred_spikes.zarr"

        # Create dataloader for inference
        inference_dataloader = DataLoader(
            spike_dataset,
            batch_size=None,
            sampler=CyclicSampler(spike_dataset),
            num_workers=0,
        )

        # Run inference and save to zarr
        inference_runner = SNNInference(
            model=recurrent_model,
            dataloader=inference_dataloader,
            device=device,
            output_mode="zarr",
            zarr_path=inferred_spikes_path,
            save_tracked_variables=False,
            max_chunks=num_chunks,
            progress_bar=True,
        )

        inference_runner.run()
        print(f"  Inferred spikes saved to: {inferred_spikes_path}")

        # Load metadata to report statistics
        zarr_root = zarr.open_group(inferred_spikes_path, mode="r")
        inferred_spikes_zarr = zarr_root["output_spikes"]
        print(f"  Inferred spikes shape: {inferred_spikes_zarr.shape}")

        # Compute mean firing rate from a sample (to avoid loading all into memory)
        sample_size = min(10000, inferred_spikes_zarr.shape[1])
        sample_spikes = np.array(inferred_spikes_zarr[:, :sample_size, :])
        mean_rate = sample_spikes.mean() * 1000 / dt
        print(f"  Mean firing rate (sampled): {mean_rate:.2f} Hz")

        # ===== M-STEP: TRAINING =====
        print(
            f"\n--- M-Step: Training feedforward network for {epochs_per_update} epochs ---"
        )

        # Create EM collate function that reads inferred spikes from zarr
        em_collate_fn, reset_counter = make_em_collate_fn(
            visible_indices=visible_indices,
            hidden_indices=hidden_indices,
            inferred_spikes_zarr_path=inferred_spikes_path,
            chunk_size=chunk_size,
            n_neurons_full=n_neurons_full,
        )

        # Create fresh optimizer for this M-step
        optimiser = torch.optim.Adam(feedforward_model.parameters(), lr=learning_rate)
        scaler = GradScaler("cuda", enabled=mixed_precision and device == "cuda")

        # Reset loss function state
        van_rossum_loss_fn.reset_state()

        # Calculate number of chunks for this M-step
        num_chunks_m_step = epochs_per_update * num_chunks

        # Create dataloader with EM collate function
        spike_dataloader = DataLoader(
            spike_dataset,
            batch_size=None,
            sampler=CyclicSampler(spike_dataset),
            num_workers=0,
            collate_fn=em_collate_fn,
        )

        pbar = tqdm(
            range(num_chunks_m_step),
            desc=f"M-step (EM {em_iter + 1})",
            unit="chunk",
            total=num_chunks_m_step,
        )

        # Update feedforward cell type names for trainer
        feedforward_copy = FeedforwardLayerConfig(**data["feedforward"])
        feedforward_copy.cell_types.names = (
            feedforward_copy.cell_types.names + recurrent.cell_types.names
        )

        # Create trainer for this M-step
        # Use global step counting so wandb metrics don't overwrite across EM iterations
        trainer = SNNTrainer(
            model=feedforward_model,
            optimizer=optimiser,
            scaler=scaler,
            dataloader=spike_dataloader,
            loss_functions={
                "van_rossum": van_rossum_loss_fn,
            },
            loss_weights=loss_weights,
            device=device,
            num_epochs=global_chunk_counter + num_chunks_m_step,  # Global end
            chunks_per_update=chunks_per_update,
            log_interval=log_interval,
            checkpoint_interval=checkpoint_interval,
            plot_size=plot_size,
            mixed_precision=mixed_precision,
            grad_norm_clip=grad_norm_clip,
            wandb_config=None,  # Don't re-initialize wandb
            progress_bar=pbar,
            plot_generator=plot_generator,
            stats_computer=stats_computer_with_em,
            connectome_mask=None,
            feedforward_mask=torch.from_numpy(
                concatenated_mask_visible.astype(np.float32)
            ).to(device),
            chunks_per_data_epoch=num_chunks,
        )

        # Attach existing wandb logger so trainer logs scaling factors
        if wandb_logger:
            trainer.wandb_logger = wandb_logger

        # Set global starting epoch so wandb steps are globally unique
        trainer.current_epoch = global_chunk_counter

        # Reset counter at start of each M-step
        reset_counter()

        # Reset model state before training
        feedforward_model.reset_state(batch_size=batch_size)

        # Create output directory for this EM iteration
        em_iter_output_dir = output_dir / f"em_iter_{em_iter + 1:03d}"
        em_iter_output_dir.mkdir(parents=True, exist_ok=True)

        best_loss = trainer.train(output_dir=em_iter_output_dir)

        if best_loss < best_loss_overall:
            best_loss_overall = best_loss

        print(f"\n  M-step best loss: {best_loss:.6f}")

        # Close trainer resources
        if trainer.metrics_logger:
            trainer.metrics_logger.close()

        # ===== TRANSFER SCALING FACTORS =====
        print("\n--- Transferring learned scaling factors to recurrent model ---")

        transfer_scaling_factors(
            feedforward_model=feedforward_model,
            recurrent_model=recurrent_model,
            n_ff_cell_types=n_ff_cell_types,
        )

        # Log EM iteration metrics
        if wandb_logger:
            wandb.log(
                {
                    "em/iteration": em_iter + 1,
                    "em/m_step_best_loss": best_loss,
                    "em/best_loss_overall": best_loss_overall,
                }
            )

        # Update global counter
        global_chunk_counter += num_chunks_m_step

        # Clean up
        if device == "cuda":
            torch.cuda.empty_cache()

    # ========
    # Clean Up
    # ========

    print("\n" + "=" * 60)
    print("EM Training complete!")
    print(f"Total EM iterations: {total_iterations}")
    print(f"Best loss overall: {best_loss_overall:.6f}")
    print(f"Hidden neurons: {n_hidden} ({hidden_cell_fraction:.1%})")
    print("=" * 60)

    # Save final state
    final_state_dir = output_dir / "final_state"
    final_state_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving final network structure...")
    np.savez(
        final_state_dir / "network_structure.npz",
        feedforward_weights=feedforward_model.weights_FF.detach().cpu().numpy(),
        feedforward_connectivity=concatenated_mask_visible,
        cell_type_indices=cell_type_indices_visible,
        feedforward_cell_type_indices=concatenated_cell_type_indices,
        scaling_factors_FF=feedforward_model.scaling_factors_FF.detach().cpu().numpy(),
        # Recurrent model scaling factors
        scaling_factors_recurrent=recurrent_model.scaling_factors.detach()
        .cpu()
        .numpy(),
        scaling_factors_FF_recurrent=recurrent_model.scaling_factors_FF.detach()
        .cpu()
        .numpy(),
        # Save mapping back to full network
        visible_indices=visible_indices,
        hidden_indices=hidden_indices,
        n_neurons_full=n_neurons_full,
    )

    if wandb_logger is not None:
        wandb.finish()

    print(f"\n✓ EM iteration outputs: {output_dir / 'em_iter_*'}")
    print(f"✓ Final state: {final_state_dir / 'network_structure.npz'}")
    print(f"✓ Hidden neuron info: {targets_dir / 'hidden_neurons.npz'}")

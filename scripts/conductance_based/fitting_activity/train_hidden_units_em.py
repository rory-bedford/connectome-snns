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
    num_chunks: int,
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
        num_chunks: Number of chunks in the zarr file (for cycling)

    Returns:
        em_collate_fn: Collate function for dataloader
        reset_counter: Function to reset chunk counter
        chunk_counter: Mutable list with current chunk index (shared reference)
        set_model: Function to set model reference for state resets
    """
    visible_tensor = torch.from_numpy(visible_indices).long()
    hidden_tensor = torch.from_numpy(hidden_indices).long()
    chunk_counter = [0]  # Mutable to track chunk index
    model_ref = [None]  # Will hold reference to model for state resets

    # Open zarr file for reading
    zarr_root = zarr.open_group(inferred_spikes_zarr_path, mode="r")
    inferred_spikes_zarr = zarr_root["output_spikes"]

    def em_collate_fn(batch: SpikeData) -> SpikeData:
        # Reset model state when cycling back to chunk 0 (prevents artifacts)
        if chunk_counter[0] % num_chunks == 0 and chunk_counter[0] > 0:
            if model_ref[0] is not None:
                model_ref[0].reset_state(batch_size=batch.target_spikes.shape[0])

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

        # Hidden: inferred spikes from zarr file (cycle through zarr for multiple epochs)
        # Note: inference model now outputs only hidden neurons directly
        chunk_idx = chunk_counter[0] % num_chunks
        start_t = chunk_idx * time_steps
        end_t = start_t + time_steps
        # Read inferred hidden spikes directly (zarr already contains only hidden neurons)
        inferred_chunk = inferred_spikes_zarr[:, start_t:end_t, :]
        hidden_spikes_chunk = (
            torch.from_numpy(np.array(inferred_chunk)).float().to(device)
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

    def set_model(model):
        """Set model reference for state resets during cycling."""
        model_ref[0] = model

    return em_collate_fn, reset_counter, chunk_counter, set_model


def make_estep_collate_fn(
    visible_indices: np.ndarray,
):
    """
    Collate function for E-step inference that provides [FF, visible teacher] as inputs.

    The inference model has:
    - Inputs: [FF spikes, visible teacher spikes]
    - Recurrence: hidden → hidden only
    - Outputs: hidden neuron spikes

    Args:
        visible_indices: Array of indices for visible neurons
    """
    visible_tensor = torch.from_numpy(visible_indices).long()

    def estep_collate_fn(batch: SpikeData) -> SpikeData:
        # Extract visible teacher spikes
        visible_teacher_spikes = batch.target_spikes[:, :, visible_tensor].float()

        # Concatenate [FF inputs, visible teacher spikes] as the new input
        concatenated_inputs = torch.cat(
            [batch.input_spikes, visible_teacher_spikes], dim=2
        )

        # Target is not used during inference, but return teacher spikes for compatibility
        return SpikeData(
            input_spikes=concatenated_inputs,
            target_spikes=batch.target_spikes,
        )

    return estep_collate_fn


def transfer_scaling_factors(
    feedforward_model: FeedforwardConductanceLIFNetwork,
    inference_model: ConductanceLIFNetwork,
    n_ff_cell_types: int,
):
    """
    Transfer learned scaling factors from feedforward (M-step) to inference (E-step) model.

    The feedforward model has concatenated scaling factors [FF part, recurrent part].
    The inference model needs:
    - scaling_factors_FF: [FF part, recurrent part] for [FF, visible] → hidden
    - scaling_factors: recurrent part for hidden → hidden

    Args:
        feedforward_model: Trained feedforward model with learned scaling_factors_FF
        inference_model: Inference model for E-step (hidden-only recurrence)
        n_ff_cell_types: Number of feedforward cell types
    """
    # Get learned concatenated scaling factors from M-step model
    learned_sf = feedforward_model.scaling_factors_FF.detach().cpu().numpy()

    # Split: [FF part, recurrent part]
    new_sf_FF = learned_sf[:n_ff_cell_types, :]  # (n_ff_types, n_rec_types)
    new_sf_rec = learned_sf[n_ff_cell_types:, :]  # (n_rec_types, n_rec_types)

    # For inference model:
    # - scaling_factors_FF: [FF, recurrent] for [FF, visible] → hidden inputs
    # - scaling_factors: recurrent part for hidden → hidden recurrence
    new_inference_sf_FF = np.concatenate([new_sf_FF, new_sf_rec], axis=0)

    # Update inference model buffers
    with torch.no_grad():
        inference_model._buffers["_scaling_factors_FF_buffer"] = torch.from_numpy(
            new_inference_sf_FF.astype(np.float32)
        ).to(inference_model.device)

        inference_model._buffers["_scaling_factors_buffer"] = torch.from_numpy(
            new_sf_rec.astype(np.float32)
        ).to(inference_model.device)

    # CRITICAL: Recompute weight caches with new scaling factors
    inference_model.set_timestep(float(inference_model.dt))


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

    # ===============================================
    # Setup Inference Model Weights (Hidden-Only Recurrence)
    # ===============================================
    # For E-step: inputs are [FF, visible teacher], recurrence is hidden→hidden only
    # This conditions hidden inference on observed visible spikes

    # Weights for feedforward-style inputs → hidden outputs
    # FF → hidden
    ff_to_hidden_weights = perturbed_ff_weights[:, hidden_indices]
    ff_to_hidden_mask = feedforward_mask[:, hidden_indices]

    # Visible → hidden (treated as feedforward input)
    visible_to_hidden_weights = perturbed_rec_weights[visible_indices, :][
        :, hidden_indices
    ]
    visible_to_hidden_mask = recurrent_mask[visible_indices, :][:, hidden_indices]

    # Concatenate: [FF, visible] → hidden
    inference_weights_FF = np.concatenate(
        [ff_to_hidden_weights, visible_to_hidden_weights], axis=0
    )
    inference_mask_FF = np.concatenate(
        [ff_to_hidden_mask, visible_to_hidden_mask], axis=0
    )

    # Recurrent weights: hidden → hidden only
    inference_weights_rec = perturbed_rec_weights[hidden_indices, :][:, hidden_indices]
    inference_mask_rec = recurrent_mask[hidden_indices, :][:, hidden_indices]

    # Cell type indices for hidden outputs
    cell_type_indices_hidden = cell_type_indices[hidden_indices]

    # Cell type indices for [FF, visible] inputs
    # FF part: original feedforward indices
    # Visible part: recurrent cell types (with offset for combined params)
    inference_cell_type_indices_FF = np.concatenate(
        [
            feedforward_cell_type_indices,
            cell_type_indices[visible_indices] + n_ff_cell_types,
        ]
    )

    # Scaling factors for inference model
    # FF → hidden: same structure as original (n_ff_cell_types, n_rec_cell_types)
    # Visible → hidden: same as recurrent (n_rec_cell_types, n_rec_cell_types)
    # Concatenated: (n_ff_cell_types + n_rec_cell_types, n_rec_cell_types)
    inference_scaling_factors_FF = np.concatenate(
        [sf_feedforward, sf_recurrent], axis=0
    )
    # Hidden → hidden recurrence: (n_rec_cell_types, n_rec_cell_types)
    inference_scaling_factors_rec = sf_recurrent.copy()

    n_inference_inputs = n_feedforward + n_visible

    print("\n✓ Inference network setup (E-step with hidden-only recurrence):")
    print(f"  - Output neurons: {n_hidden} (hidden only)")
    print(
        f"  - FF inputs: [FF={n_feedforward}, visible={n_visible}] = {n_inference_inputs}"
    )
    print(f"  - Recurrent: hidden→hidden only ({n_hidden}×{n_hidden})")
    print(f"  - Weight matrix FF shape: {inference_weights_FF.shape}")
    print(f"  - Weight matrix rec shape: {inference_weights_rec.shape}")

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
    # Initialize INFERENCE Model (E-step)
    # ======================================
    # This model has:
    # - Inputs: [FF spikes, visible teacher spikes]
    # - Recurrence: hidden → hidden only
    # - Outputs: hidden neurons only

    print("\n✓ Initializing inference model for E-step (hidden-only recurrence)...")

    inference_model = ConductanceLIFNetwork(
        dt=dt,
        weights=inference_weights_rec,  # hidden → hidden only
        weights_FF=inference_weights_FF,  # [FF, visible] → hidden
        cell_type_indices=cell_type_indices_hidden,  # hidden neuron types
        cell_type_indices_FF=inference_cell_type_indices_FF,  # [FF, visible] input types
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,  # Combined [FF, recurrent] cell params
        synapse_params=recurrent_synapse_params,  # For hidden → hidden synapses
        synapse_params_FF=combined_synapse_params_FF,  # For [FF, visible] → hidden synapses
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        scaling_factors=inference_scaling_factors_rec.copy(),  # hidden → hidden
        scaling_factors_FF=inference_scaling_factors_FF.copy(),  # [FF, visible] → hidden
        optimisable=None,  # No optimization - just inference
        connectome_mask=inference_mask_rec,
        feedforward_mask=inference_mask_FF,
        track_variables=False,
        use_tqdm=False,
    )
    inference_model.to(device)

    print(f"  - Output neurons: {n_hidden} (hidden only)")
    print(
        f"  - Input neurons: {n_inference_inputs} ({n_feedforward} FF + {n_visible} visible)"
    )
    print(f"  - Recurrent connections: {n_hidden}×{n_hidden} (hidden→hidden)")

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
    # Factory Functions for Callbacks (with access to inferred spikes)
    # ================================================

    def make_plot_generator(inferred_spikes_zarr_path: Path, chunk_counter_ref: list):
        """
        Factory function that creates plot_generator with access to inferred spikes.

        Args:
            inferred_spikes_zarr_path: Path to zarr file with inferred spikes
            chunk_counter_ref: Reference to current chunk counter for indexing
        """
        # Open zarr for reading inferred spikes
        zarr_root = zarr.open_group(inferred_spikes_zarr_path, mode="r")
        inferred_spikes_zarr = zarr_root["output_spikes"]

        # Open original spike data for teacher hidden spikes
        teacher_zarr_root = zarr.open_group(input_dir / "spike_data.zarr", mode="r")
        teacher_spikes_zarr = teacher_zarr_root["output_spikes"]

        def plot_generator(
            spikes,
            input_spikes,
            target_spikes,
            **kwargs,
        ):
            """Generate spike train comparison plots for visible and hidden neurons."""
            figs = {}

            # === VISIBLE NEURONS PLOT ===
            n_plot = min(10, spikes.shape[2])

            # Randomly select neurons to plot
            plot_indices_visible = np.random.choice(
                spikes.shape[2], size=n_plot, replace=False
            )

            interleaved = np.zeros((1, spikes.shape[1], 2 * n_plot))
            for i, neuron_idx in enumerate(plot_indices_visible):
                interleaved[0, :, 2 * i] = target_spikes[0, :, neuron_idx]
                interleaved[0, :, 2 * i + 1] = spikes[0, :, neuron_idx]

            cell_type_indices_plot = np.array([0, 1] * n_plot)
            cell_type_names_plot = ["Target", "Trained"]

            fig_visible = plot_spike_trains(
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
            figs["spike_comparison_visible"] = fig_visible

            # === HIDDEN NEURONS PLOT (new) ===
            if n_hidden > 0:
                # Get current chunk index (approximate - use modulo for cycling)
                chunk_idx = (chunk_counter_ref[0] - 1) % num_chunks
                start_t = chunk_idx * chunk_size
                end_t = start_t + chunk_size

                # Get inferred hidden spikes from zarr
                # Note: inference model now outputs only hidden neurons directly
                inferred_hidden = np.array(inferred_spikes_zarr[:1, start_t:end_t, :])

                # Get teacher hidden spikes from original data
                # Extract all neurons first, then index (zarr fancy indexing issue)
                teacher_all_plot = np.array(teacher_spikes_zarr[:1, start_t:end_t, :])[
                    0
                ]  # (time, n_all_neurons)
                teacher_hidden = teacher_all_plot[:, hidden_indices][
                    np.newaxis, :, :
                ]  # (1, time, n_hidden)

                n_plot_hidden = min(10, n_hidden)

                # Randomly select hidden neurons to plot
                plot_indices_hidden = np.random.choice(
                    n_hidden, size=n_plot_hidden, replace=False
                )

                interleaved_hidden = np.zeros(
                    (1, inferred_hidden.shape[1], 2 * n_plot_hidden)
                )
                for i, neuron_idx in enumerate(plot_indices_hidden):
                    interleaved_hidden[0, :, 2 * i] = teacher_hidden[0, :, neuron_idx]
                    interleaved_hidden[0, :, 2 * i + 1] = inferred_hidden[
                        0, :, neuron_idx
                    ]

                cell_type_indices_hidden = np.array([0, 1] * n_plot_hidden)
                cell_type_names_hidden = ["Teacher", "Inferred"]

                fig_hidden = plot_spike_trains(
                    spikes=interleaved_hidden,
                    dt=dt,
                    cell_type_indices=cell_type_indices_hidden,
                    cell_type_names=cell_type_names_hidden,
                    n_neurons_plot=2 * n_plot_hidden,
                    fraction=1.0,
                    random_seed=None,
                    title=f"EM Training: Teacher vs Inferred (first {n_plot_hidden} hidden neurons, {hidden_cell_fraction:.0%} hidden)",
                    ylabel="Neuron",
                    figsize=(14, 8),
                )
                figs["spike_comparison_hidden"] = fig_hidden

            return figs

        return plot_generator

    def make_stats_computer(inferred_spikes_zarr_path: Path, chunk_counter_ref: list):
        """
        Factory function that creates stats_computer with access to inferred spikes.

        Args:
            inferred_spikes_zarr_path: Path to zarr file with inferred spikes
            chunk_counter_ref: Reference to current chunk counter for indexing
        """
        # Open zarr for reading inferred spikes
        zarr_root = zarr.open_group(inferred_spikes_zarr_path, mode="r")
        inferred_spikes_zarr = zarr_root["output_spikes"]

        # Open original spike data for teacher spikes
        teacher_zarr_root = zarr.open_group(input_dir / "spike_data.zarr", mode="r")
        teacher_spikes_zarr = teacher_zarr_root["output_spikes"]

        def stats_computer(spikes, model_snapshot):
            """Compute summary statistics for visible and hidden neurons (student and teacher)."""
            # Student visible: from model output (spikes accumulated over multiple chunks)
            student_visible = spikes[0, :, :]  # (time_accumulated, n_visible)
            n_timesteps_accumulated = student_visible.shape[0]

            duration_s = n_timesteps_accumulated * dt / 1000.0

            # Calculate how many zarr chunks this corresponds to
            n_chunks_accumulated = n_timesteps_accumulated // chunk_size

            # Get the ending chunk index to extract the corresponding teacher data
            current_chunk_idx = (chunk_counter_ref[0] - 1) % num_chunks
            start_chunk_idx = max(0, current_chunk_idx - n_chunks_accumulated + 1)

            start_t = start_chunk_idx * chunk_size
            end_t = start_t + n_timesteps_accumulated

            # Student hidden: from inferred spikes zarr (same time range as student_visible)
            student_hidden = np.array(inferred_spikes_zarr[:1, start_t:end_t, :])[
                0
            ]  # (time_accumulated, n_hidden)

            # Teacher spikes: extract all neurons first, then index
            # (zarr fancy indexing doesn't work the same as numpy)
            teacher_all = np.array(teacher_spikes_zarr[:1, start_t:end_t, :])[
                0
            ]  # (time_accumulated, n_all_neurons)

            teacher_visible = teacher_all[
                :, visible_indices
            ]  # (time_accumulated, n_visible)
            teacher_hidden = teacher_all[
                :, hidden_indices
            ]  # (time_accumulated, n_hidden)

            # Compute firing rates for all 4 categories
            def compute_firing_rate_stats(spike_data, prefix):
                """Compute firing rate statistics for a spike matrix."""
                spike_counts = spike_data.sum(axis=0)
                firing_rates = spike_counts / duration_s
                return {
                    f"{prefix}/mean": float(firing_rates.mean()),
                    f"{prefix}/std": float(firing_rates.std()),
                    f"{prefix}/min": float(firing_rates.min()),
                    f"{prefix}/max": float(firing_rates.max()),
                }

            stats = {}

            # Student firing rates
            stats.update(
                compute_firing_rate_stats(
                    student_visible, "firing_rate/student/visible"
                )
            )
            if n_hidden > 0:
                stats.update(
                    compute_firing_rate_stats(
                        student_hidden, "firing_rate/student/hidden"
                    )
                )

            # Teacher firing rates
            stats.update(
                compute_firing_rate_stats(
                    teacher_visible, "firing_rate/teacher/visible"
                )
            )
            if n_hidden > 0:
                stats.update(
                    compute_firing_rate_stats(
                        teacher_hidden, "firing_rate/teacher/hidden"
                    )
                )

            # Hidden cell configuration
            stats["hidden_cells/fraction"] = hidden_cell_fraction
            stats["hidden_cells/n_hidden"] = n_hidden
            stats["hidden_cells/n_visible"] = n_visible

            # Scaling factors
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

        return stats_computer

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

    # Track current EM iteration (mutable container for closure)
    current_em_iter = [0]

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

        # Create output directory for this EM iteration
        em_iter_output_dir = output_dir / f"em_iter_{em_iter + 1:03d}"
        em_iter_output_dir.mkdir(parents=True, exist_ok=True)

        # ===== E-STEP: INFERENCE =====
        # Infer hidden unit activities conditioned on visible teacher spikes
        # Model: [FF, visible] → hidden with hidden→hidden recurrence only
        print(
            "\n--- E-Step: Inferring hidden activities conditioned on visible spikes ---"
        )
        inference_model.reset_state(batch_size=batch_size)

        # Save inferred spikes to zarr for M-step
        inferred_spikes_path = em_iter_output_dir / "inferred_spikes.zarr"

        # Create collate function that provides [FF, visible teacher] as inputs
        estep_collate_fn = make_estep_collate_fn(visible_indices=visible_indices)

        # Create dataloader for inference with E-step collate function
        inference_dataloader = DataLoader(
            spike_dataset,
            batch_size=None,
            sampler=CyclicSampler(spike_dataset),
            num_workers=0,
            collate_fn=estep_collate_fn,
        )

        # Run inference and save to zarr
        inference_runner = SNNInference(
            model=inference_model,
            dataloader=inference_dataloader,
            device=device,
            output_mode="zarr",
            zarr_path=inferred_spikes_path,
            save_tracked_variables=False,
            max_chunks=num_chunks,
            progress_bar=True,
        )

        inference_runner.run()
        print(f"  Inferred hidden spikes saved to: {inferred_spikes_path}")

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
        em_collate_fn, reset_counter, chunk_counter_ref, set_model = make_em_collate_fn(
            visible_indices=visible_indices,
            hidden_indices=hidden_indices,
            inferred_spikes_zarr_path=inferred_spikes_path,
            chunk_size=chunk_size,
            n_neurons_full=n_neurons_full,
            num_chunks=num_chunks,
        )

        # Pass model reference to collate function so it can reset state when cycling
        set_model(feedforward_model)

        # Create callbacks with access to inferred spikes and chunk counter
        plot_generator = make_plot_generator(inferred_spikes_path, chunk_counter_ref)
        stats_computer = make_stats_computer(inferred_spikes_path, chunk_counter_ref)

        # Wrap stats_computer to include EM iteration
        def stats_computer_with_em(spikes, model_snapshot):
            stats = stats_computer(spikes, model_snapshot)
            stats["em/iteration"] = current_em_iter[0] + 1
            return stats

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

        best_loss = trainer.train(output_dir=em_iter_output_dir)

        if best_loss < best_loss_overall:
            best_loss_overall = best_loss

        print(f"\n  M-step best loss: {best_loss:.6f}")

        # Close trainer resources
        if trainer.metrics_logger:
            trainer.metrics_logger.close()

        # ===== TRANSFER SCALING FACTORS =====
        print("\n--- Transferring learned scaling factors to inference model ---")

        transfer_scaling_factors(
            feedforward_model=feedforward_model,
            inference_model=inference_model,
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
        # Inference model scaling factors (hidden-only recurrence)
        scaling_factors_inference_rec=inference_model.scaling_factors.detach()
        .cpu()
        .numpy(),
        scaling_factors_inference_FF=inference_model.scaling_factors_FF.detach()
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

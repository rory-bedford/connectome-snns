"""
Training all neurons with feedforward dynamics to match target activity.

This script is identical to train_feedforward.py, except that Gaussian noise
is added to the weight matrix before training. The noise is multiplicative
and clipped at zero:
    w_noisy = w * max(1 + noise_frac * N(0,1), 0)

After applying the clipped noise, an affine transformation is applied to the
non-zero weights so that both the mean and std exactly match the original
weight distribution. This ensures the noise perturbs individual weights while
preserving the overall distribution statistics.

TODO: Log teacher firing rate metrics (mean, std, min, max) to the training CSV.
      Currently we only log learned firing rates, which makes post-hoc analysis harder
      since we need to load the zarr file to compute teacher rates.
"""

import numpy as np
import matplotlib.pyplot as plt
import zarr
from dataloaders.supervised import (
    PrecomputedSpikeDataset,
    CyclicSampler,
    feedforward_collate_fn,
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


def apply_weight_noise(weights, noise_frac, rng=None, preserve_statistics=True):
    """Apply multiplicative Gaussian noise to weights with statistics preservation.

    Applies noise: w * max(1 + noise_frac * N(0,1), 0), then uses an affine
    transformation to exactly match the original mean and std of non-zero weights.

    Args:
        weights: Weight matrix (numpy array, may be sparse with zeros)
        noise_frac: Fraction of weight magnitude for noise (e.g., 0.1 for 10%)
        rng: Optional numpy random generator for reproducibility
        preserve_statistics: If True, apply affine transform to match original
            mean and std of non-zero weights (default: True)

    Returns:
        Noisy weights with same shape as input. Zero weights remain zero.
        Non-zero weights have the same mean and std as the original.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Identify non-zero weights (sparse connectivity)
    nonzero_mask = weights != 0

    # Original statistics on non-zero weights only
    orig_mean = weights[nonzero_mask].mean()
    orig_std = weights[nonzero_mask].std()

    # Multiplicative noise: w * (1 + noise_frac * N(0,1))
    # Clip at 0 only (no upper bound) to prevent negative weights before rescaling
    multiplier = 1 + noise_frac * rng.standard_normal(weights.shape)
    multiplier = np.maximum(multiplier, 0)

    noisy_weights = weights * multiplier

    if preserve_statistics and orig_std > 0:
        # Compute noisy statistics on non-zero positions only
        noisy_nz = noisy_weights[nonzero_mask]
        noisy_mean = noisy_nz.mean()
        noisy_std = noisy_nz.std()

        # Affine transform to exactly match original mean and std
        # new = orig_mean + (old - old_mean) * (orig_std / old_std)
        if noisy_std > 0:
            noisy_weights[nonzero_mask] = orig_mean + (noisy_nz - noisy_mean) * (
                orig_std / noisy_std
            )

    return noisy_weights


def main(
    input_dir,
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
):
    """Train all neurons to match target spike train activity using feedforward dynamics.

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

    # Load weight noise configuration
    weight_noise_config = data.get("weight_noise", {})
    noise_frac = weight_noise_config.get("noise_frac", 0.0)

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
    momentum = training.momentum
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

    # Create RNG for weight noise (uses same seed for reproducibility)
    weight_noise_rng = np.random.default_rng(seed)

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

    # ===============================================
    # Apply Gaussian Noise to Weights
    # ===============================================

    if noise_frac > 0:
        print(
            f"\nApplying {noise_frac * 100:.1f}% multiplicative Gaussian noise to weights (statistics-preserving)..."
        )
        # Compute statistics on non-zero weights only (sparse connectivity)
        nonzero_mask = concatenated_weights != 0
        original_weights_nz = concatenated_weights[nonzero_mask].copy()
        original_mean = original_weights_nz.mean()
        original_std = original_weights_nz.std()

        concatenated_weights = apply_weight_noise(
            concatenated_weights,
            noise_frac,
            rng=weight_noise_rng,
            preserve_statistics=True,
        )

        noisy_weights_nz = concatenated_weights[nonzero_mask]
        noisy_mean = noisy_weights_nz.mean()
        noisy_std = noisy_weights_nz.std()
        print(
            f"  - Original weights (non-zero): mean={original_mean:.6f}, std={original_std:.6f}"
        )
        print(
            f"  - Noisy weights (non-zero):    mean={noisy_mean:.6f}, std={noisy_std:.6f}"
        )
        print(
            f"  - Statistics preserved: mean change = {abs(noisy_mean - original_mean):.2e}, std change = {abs(noisy_std - original_std):.2e}"
        )

        # Create scatter plot comparing original vs noisy weights
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        ax = axes[0]
        max_points = 50000
        if len(original_weights_nz) > max_points:
            idx = np.random.default_rng(42).choice(
                len(original_weights_nz), max_points, replace=False
            )
        else:
            idx = np.arange(len(original_weights_nz))

        ax.scatter(original_weights_nz[idx], noisy_weights_nz[idx], alpha=0.2, s=1)
        lims = [
            min(0, noisy_weights_nz.min()),
            np.percentile(np.concatenate([original_weights_nz, noisy_weights_nz]), 95),
        ]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Original Weight")
        ax.set_ylabel("Noisy Weight")
        ax.set_aspect("equal")

        corr = np.corrcoef(original_weights_nz, noisy_weights_nz)[0, 1]
        stats_text = (
            f"Original: μ={original_mean:.6f}, σ={original_std:.6f}\n"
            f"Noisy:    μ={noisy_mean:.6f}, σ={noisy_std:.6f}\n"
            f"Corr: r={corr:.4f}"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax.set_title("Original vs Noisy Weights")

        # Histogram comparison
        ax = axes[1]
        min_val = min(0, noisy_weights_nz.min())
        max_val = np.percentile(
            np.concatenate([original_weights_nz, noisy_weights_nz]), 99
        )
        bins = np.linspace(min_val, max_val, 100)
        ax.hist(
            original_weights_nz,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Original",
            color="blue",
        )
        ax.hist(
            noisy_weights_nz,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Noisy",
            color="orange",
        )
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Weight")
        ax.set_ylabel("Density")
        ax.set_title("Distribution Comparison")
        ax.legend()

        fig.suptitle(
            f"Weight Noise: {noise_frac * 100:.1f}% (Clip [0, ∞) + Affine Rescale)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save plot
        plot_path = output_dir / "weight_noise_comparison.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  - Saved weight noise comparison plot to {plot_path}")
    else:
        print("\nNo weight noise applied (noise_frac = 0)")

    print("\n✓ Feedforward network setup:")
    print(f"  - Output neurons: {n_neurons}")
    print(
        f"  - Total inputs per neuron: {n_total_inputs} ({n_feedforward} FF + {n_neurons} recurrent)"
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
    # Sample TARGET scaling factors from log-normal distribution with mean 1
    sigma = np.sqrt(weight_perturbation_variance)
    mu = -(sigma**2) / 2.0  # Ensures E[target] = 1

    # Generate target scaling factors directly (mean 1, controllable variance)
    target_scaling_factors_FF = np.random.lognormal(
        mean=mu, sigma=sigma, size=concatenated_scaling_factors.shape
    )

    # Perturbation is reciprocal of target (so target * perturbation = 1)
    perturbation_factors = 1.0 / target_scaling_factors_FF

    # Apply perturbation to weights for all neurons
    perturbed_weights = concatenated_weights.copy()
    for input_idx in range(n_total_inputs):
        input_type = concatenated_cell_type_indices[input_idx]
        for output_idx in range(n_neurons):
            output_type = cell_type_indices[output_idx]
            perturbed_weights[input_idx, output_idx] *= perturbation_factors[
                input_type, output_type
            ]

    # Save targets
    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        targets_dir / "target_scaling_factors.npz",
        feedforward_scaling_factors=target_scaling_factors_FF,
    )

    # Also save the noise configuration for reproducibility
    np.savez(
        targets_dir / "weight_noise_config.npz",
        noise_frac=noise_frac,
        seed=seed,
    )

    # ======================
    # Load Dataset from Disk
    # ======================

    # Load precomputed spike data from zarr
    spike_dataset = PrecomputedSpikeDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
    )

    batch_size = spike_dataset.batch_size

    print(f"\n✓ Loaded {spike_dataset.num_chunks} chunks × {batch_size} batch size")

    # DataLoader with custom collate function for concatenation
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        num_workers=0,
        collate_fn=feedforward_collate_fn,
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

    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(momentum, 0.999)
    )
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
            title=f"Feedforward Network (noisy): Target vs Trained (first {n_plot} neurons)",
            ylabel="Neuron",
            figsize=(14, 8),
        )

        return {"spike_comparison": fig}

    # ================================================
    # Create Stats Computer Function
    # ================================================

    # Open teacher spike data for computing teacher firing rates
    teacher_zarr_root = zarr.open_group(input_dir / "spike_data.zarr", mode="r")
    teacher_spikes_zarr = teacher_zarr_root["output_spikes"]
    chunk_size = spike_dataset.chunk_size
    num_chunks = spike_dataset.num_chunks
    dt = spike_dataset.dt

    # Track chunk counter for accessing correct teacher data
    chunk_counter_ref = [0]

    def stats_computer(spikes, model_snapshot):
        """Compute summary statistics for all neurons (student and teacher)."""
        # spikes shape: (batch, time, n_neurons)
        student_spikes = spikes[0, :, :]  # (time, n_neurons)
        n_timesteps = student_spikes.shape[0]
        duration_s = n_timesteps * dt / 1000.0

        # Get corresponding teacher spikes
        n_chunks_accumulated = n_timesteps // chunk_size
        current_chunk_idx = chunk_counter_ref[0] % num_chunks
        start_chunk_idx = max(0, current_chunk_idx - n_chunks_accumulated + 1)
        start_t = start_chunk_idx * chunk_size
        end_t = start_t + n_timesteps
        teacher_spikes = np.array(teacher_spikes_zarr[:1, start_t:end_t, :])[0]
        chunk_counter_ref[0] += 1

        # Compute student firing rates
        student_spike_counts = student_spikes.sum(axis=0)
        student_firing_rates = student_spike_counts / duration_s

        # Compute teacher firing rates
        teacher_spike_counts = teacher_spikes.sum(axis=0)
        teacher_firing_rates = teacher_spike_counts / duration_s

        # Use flattened metric names so wandb glob pattern firing_rate/* matches all
        stats = {
            # Student aggregate stats
            "firing_rate/student_mean": float(student_firing_rates.mean()),
            "firing_rate/student_std": float(student_firing_rates.std()),
            "firing_rate/student_min": float(student_firing_rates.min()),
            "firing_rate/student_max": float(student_firing_rates.max()),
            # Teacher aggregate stats
            "firing_rate/teacher_mean": float(teacher_firing_rates.mean()),
            "firing_rate/teacher_std": float(teacher_firing_rates.std()),
            "firing_rate/teacher_min": float(teacher_firing_rates.min()),
            "firing_rate/teacher_max": float(teacher_firing_rates.max()),
        }

        # Add cell-type-specific firing rates (handles arbitrary cell types from teacher)
        output_cell_type_names = recurrent.cell_types.names
        for type_idx, type_name in enumerate(output_cell_type_names):
            type_mask = cell_type_indices == type_idx
            if type_mask.sum() > 0:
                # Student by cell type
                student_type_rates = student_firing_rates[type_mask]
                stats[f"firing_rate/student_{type_name}_mean"] = float(
                    student_type_rates.mean()
                )
                stats[f"firing_rate/student_{type_name}_std"] = float(
                    student_type_rates.std()
                )
                # Teacher by cell type
                teacher_type_rates = teacher_firing_rates[type_mask]
                stats[f"firing_rate/teacher_{type_name}_mean"] = float(
                    teacher_type_rates.mean()
                )
                stats[f"firing_rate/teacher_{type_name}_std"] = float(
                    teacher_type_rates.std()
                )

        # Add scaling factor tracking with proper cell type names
        current_sf = model_snapshot["scaling_factors_FF"]
        target_sf = target_scaling_factors_FF

        # Get cell type names for proper labeling
        # Combined input types: feedforward names + recurrent names
        input_cell_type_names = (
            feedforward.cell_types.names + recurrent.cell_types.names
        )
        output_cell_type_names = recurrent.cell_types.names

        # Log all scaling factor elements normalized so target=1
        # This makes it easy to see convergence (value should approach 1)
        # Use flattened names so wandb glob pattern scaling_factors/* matches all
        for source_idx in range(current_sf.shape[0]):
            source_type_name = input_cell_type_names[source_idx]
            for target_idx in range(current_sf.shape[1]):
                target_type_name = output_cell_type_names[target_idx]
                synapse_name = f"{source_type_name}_to_{target_type_name}"
                target_val = target_sf[source_idx, target_idx]
                if target_val != 0:
                    normalized_value = current_sf[source_idx, target_idx] / target_val
                else:
                    normalized_value = current_sf[source_idx, target_idx]
                stats[f"scaling_factors/{synapse_name}_value"] = float(normalized_value)
                stats[f"scaling_factors/{synapse_name}_target"] = 1.0

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
                "weight_noise": weight_noise_config,
                "output_dir": str(output_dir),
                "device": device,
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
    if noise_frac > 0:
        print(f"Weight noise: {noise_frac * 100:.1f}%")

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

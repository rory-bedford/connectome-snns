"""
Check loss when scaling factors are at target values (normalized = 1).

Same setup as train_feedforward_noisy.py but no training - just runs inference
for plot_size chunks with scaling factors initialized to targets, then plots
and logs metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
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
from training_utils.losses import VanRossumLoss
from configs import (
    StudentSimulationConfig,
    StudentTrainingConfig,
    StudentHyperparameters,
)
from configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig
from snn_runners import SNNInference
import toml
from visualization.neuronal_dynamics import plot_spike_trains


def apply_weight_noise(weights, noise_frac, rng=None, preserve_statistics=True):
    """Apply multiplicative Gaussian noise to weights with statistics preservation."""
    if rng is None:
        rng = np.random.default_rng()

    nonzero_mask = weights != 0
    orig_mean = weights[nonzero_mask].mean()
    orig_std = weights[nonzero_mask].std()

    multiplier = 1 + noise_frac * rng.standard_normal(weights.shape)
    multiplier = np.maximum(multiplier, 0)
    noisy_weights = weights * multiplier

    if preserve_statistics and orig_std > 0:
        noisy_nz = noisy_weights[nonzero_mask]
        noisy_mean = noisy_nz.mean()
        noisy_std = noisy_nz.std()
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
    """Run inference with scaling factors at target values.

    Args:
        input_dir (Path): Directory containing teacher data (network_structure.npz, spike_data.zarr)
        output_dir (Path): Directory where outputs will be saved
        params_file (Path): Path to the parameters TOML file
        wandb_config (dict, optional): W&B configuration (not used)
        resume_from (Path, optional): Not used
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load parameters
    with open(params_file, "r") as f:
        data = toml.load(f)

    simulation = StudentSimulationConfig(**data["simulation"])
    training = StudentTrainingConfig(**data["training"])
    hyperparameters = StudentHyperparameters(**data["hyperparameters"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])
    scaling_factors = data.get("scaling_factors", {})

    weight_noise_config = data.get("weight_noise", {})
    noise_frac = weight_noise_config.get("noise_frac", 0.0)

    chunk_size = simulation.chunk_size
    seed = simulation.seed
    plot_size = training.plot_size
    weight_perturbation_variance = training.weight_perturbation_variance

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    weight_noise_rng = np.random.default_rng(seed)

    # Load network structure
    network_structure = np.load(input_dir / "network_structure.npz")

    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    feedforward_cell_type_indices = network_structure["feedforward_cell_type_indices"]
    recurrent_mask = network_structure["recurrent_connectivity"]
    feedforward_mask = network_structure["feedforward_connectivity"]

    n_neurons = weights.shape[0]
    n_feedforward = feedforward_weights.shape[0]
    n_total_inputs = n_feedforward + n_neurons

    # Concatenate weights
    concatenated_weights = np.concatenate([feedforward_weights, weights], axis=0)
    concatenated_mask = np.concatenate([feedforward_mask, recurrent_mask], axis=0)

    # Apply weight noise
    if noise_frac > 0:
        print(f"\nApplying {noise_frac * 100:.1f}% multiplicative Gaussian noise...")
        concatenated_weights = apply_weight_noise(
            concatenated_weights,
            noise_frac,
            rng=weight_noise_rng,
            preserve_statistics=True,
        )

    # Setup cell parameters
    recurrent_cell_params = recurrent.get_cell_params()
    feedforward_cell_params = feedforward.get_cell_params()
    n_ff_cell_types = len(feedforward_cell_params)

    concatenated_cell_type_indices = np.concatenate(
        [feedforward_cell_type_indices, cell_type_indices + n_ff_cell_types]
    )

    combined_cell_params_FF = feedforward_cell_params.copy()
    for cell_params in recurrent_cell_params:
        offset_cell_params = cell_params.copy()
        offset_cell_params["cell_id"] = cell_params["cell_id"] + n_ff_cell_types
        combined_cell_params_FF.append(offset_cell_params)

    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()
    n_ff_synapse_types = len(feedforward_synapse_params)

    combined_synapse_params_FF = feedforward_synapse_params.copy()
    for syn_params in recurrent_synapse_params:
        offset_syn_params = syn_params.copy()
        offset_syn_params["cell_id"] = syn_params["cell_id"] + n_ff_cell_types
        offset_syn_params["synapse_id"] = syn_params["synapse_id"] + n_ff_synapse_types
        combined_synapse_params_FF.append(offset_syn_params)

    # Get base scaling factors
    sf_feedforward = np.array(scaling_factors["feedforward"])
    sf_recurrent = np.array(scaling_factors["recurrent"])
    concatenated_scaling_factors = np.concatenate(
        [sf_feedforward, sf_recurrent], axis=0
    )

    # Generate target scaling factors (same as training)
    sigma = np.sqrt(weight_perturbation_variance)
    mu = -(sigma**2) / 2.0
    target_scaling_factors_FF = np.random.lognormal(
        mean=mu, sigma=sigma, size=concatenated_scaling_factors.shape
    )

    # Perturbation is reciprocal of target
    perturbation_factors = 1.0 / target_scaling_factors_FF

    # Apply perturbation to weights
    perturbed_weights = concatenated_weights.copy()
    for input_idx in range(n_total_inputs):
        input_type = concatenated_cell_type_indices[input_idx]
        for output_idx in range(n_neurons):
            output_type = cell_type_indices[output_idx]
            perturbed_weights[input_idx, output_idx] *= perturbation_factors[
                input_type, output_type
            ]

    # Initialize scaling factors AT TARGET (so normalized = 1)
    scaling_factors_init = target_scaling_factors_FF.copy()

    # Load dataset
    spike_dataset = PrecomputedSpikeDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
    )
    batch_size = spike_dataset.batch_size
    dt = spike_dataset.dt

    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        num_workers=0,
        collate_fn=feedforward_collate_fn,
    )

    # Initialize model
    model = FeedforwardConductanceLIFNetwork(
        dt=dt,
        weights_FF=perturbed_weights,
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=concatenated_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=hyperparameters.surrgrad_scale,
        batch_size=batch_size,
        scaling_factors_FF=scaling_factors_init,
        optimisable=None,
        feedforward_mask=concatenated_mask,
        track_variables=False,
        use_tqdm=False,
    )
    model.to(device)
    model.eval()
    model.reset_state(batch_size=batch_size)

    # Run inference (2x plot_size, discard first half for burn-in)
    total_chunks = 2 * plot_size
    print(
        f"\nRunning inference for {total_chunks} chunks ({plot_size} burn-in + {plot_size} analysis)..."
    )
    inference_runner = SNNInference(
        model=model,
        dataloader=spike_dataloader,
        device=device,
        output_mode="memory",
        max_chunks=total_chunks,
        save_tracked_variables=False,
    )
    results = inference_runner.run()

    # Extract results and discard burn-in (first half)
    burnin_timesteps = plot_size * chunk_size
    output_spikes = results["output_spikes"][
        :, burnin_timesteps:, :
    ]  # (batch, time, neurons)
    target_spikes = results["target_spikes"][
        :, burnin_timesteps:, :
    ]  # (batch, time, neurons)

    # Compute loss
    van_rossum_loss_fn = VanRossumLoss(
        tau_rise=hyperparameters.van_rossum_tau_rise,
        tau_decay=hyperparameters.van_rossum_tau_decay,
        dt=dt,
        window_size=chunk_size,
        device=device,
    )

    output_spikes_t = torch.from_numpy(output_spikes).float().to(device)
    target_spikes_t = torch.from_numpy(target_spikes).float().to(device)
    loss = van_rossum_loss_fn(output_spikes_t, target_spikes_t).item()

    # Compute firing rates
    duration_s = output_spikes.shape[1] * dt / 1000.0
    student_fr = output_spikes[0].sum(axis=0) / duration_s
    teacher_fr = target_spikes[0].sum(axis=0) / duration_s

    exc_mask = cell_type_indices == 0
    inh_mask = cell_type_indices == 1

    metrics = {
        "loss": loss,
        "firing_rate/student_mean": float(student_fr.mean()),
        "firing_rate/student_std": float(student_fr.std()),
        "firing_rate/teacher_mean": float(teacher_fr.mean()),
        "firing_rate/teacher_std": float(teacher_fr.std()),
        "firing_rate/student_exc_mean": float(student_fr[exc_mask].mean()),
        "firing_rate/student_inh_mean": float(student_fr[inh_mask].mean()),
        "firing_rate/teacher_exc_mean": float(teacher_fr[exc_mask].mean()),
        "firing_rate/teacher_inh_mean": float(teacher_fr[inh_mask].mean()),
    }

    # Print metrics
    print("\n" + "=" * 60)
    print("METRICS (scaling factors at target)")
    print("=" * 60)
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")
    print("=" * 60)

    # Generate plot
    n_plot = min(10, output_spikes.shape[2])
    interleaved = np.zeros((1, output_spikes.shape[1], 2 * n_plot))
    for i in range(n_plot):
        interleaved[0, :, 2 * i] = target_spikes[0, :, i]
        interleaved[0, :, 2 * i + 1] = output_spikes[0, :, i]

    cell_type_indices_plot = np.array([0, 1] * n_plot)
    cell_type_names_plot = ["Target", "Student"]

    fig = plot_spike_trains(
        spikes=interleaved,
        dt=dt,
        cell_type_indices=cell_type_indices_plot,
        cell_type_names=cell_type_names_plot,
        n_neurons_plot=2 * n_plot,
        fraction=1.0,
        random_seed=None,
        title=f"Scaling Factors at Target (loss={loss:.4f})",
        ylabel="Neuron",
        figsize=(14, 8),
    )

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / "spike_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(output_dir / "metrics.npz", **metrics)

    print(f"\nSaved to {output_dir}")

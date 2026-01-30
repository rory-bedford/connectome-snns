"""Check if target scaling factors give better loss than learned ones.

This script loads an existing noisy weight experiment, reconstructs the network
with different scaling factors (target, learned, or ones), and runs inference
to compute loss and metrics. This helps diagnose whether:
1. The target scaling factors truly give optimal loss
2. Or the network found a different (possibly better) solution

The noisy weights experiment applies multiplicative noise to weights and then
trains scaling factors to recover the original network dynamics. This script
tests whether the "correct" scaling factors (target=1.0 when normalized)
actually minimize the loss.

No optimization is performed - just inference and evaluation.

Args:
    input_dir (Path): Path to a trained noisy weights experiment directory
        (e.g., .../noisy-weights/varying-noise/noise-0.40)
    output_dir (Path): Directory where results will be saved
    params_file (Path): Path to parameters TOML file
    wandb_config (dict, optional): W&B configuration
    resume_from (Path, optional): Not used by this script
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import toml
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import StudentSimulationConfig, StudentHyperparameters
from configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig
from dataloaders.supervised import (
    PrecomputedSpikeDataset,
    CyclicSampler,
    feedforward_collate_fn,
)
from network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from training_utils.losses import VanRossumLoss
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


def run_inference_with_scaling_factors(
    input_dir,
    output_dir,
    scaling_factor_mode,
    num_chunks,
    recurrent,
    feedforward,
    hyperparameters,
    simulation,
):
    """
    Run inference on a trained experiment with specified scaling factors.

    Args:
        input_dir: Path to experiment directory containing trained model
        output_dir: Path to save results
        scaling_factor_mode: "target" for correct SF, "learned" for trained SF, "ones" for all 1s
        num_chunks: Number of chunks to run inference on
        recurrent: RecurrentLayerConfig
        feedforward: FeedforwardLayerConfig
        hyperparameters: StudentHyperparameters
        simulation: StudentSimulationConfig
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Input experiment: {input_dir}")
    print(f"Scaling factor mode: {scaling_factor_mode}")

    # Load parameters from the original experiment
    params_file = input_dir / "parameters.toml"
    with open(params_file, "r") as f:
        data = toml.load(f)

    scaling_factors_config = data.get("scaling_factors", {})
    weight_noise_config = data.get("weight_noise", {})
    noise_frac = weight_noise_config.get("noise_frac", 0.0)

    chunk_size = simulation.chunk_size
    seed = simulation.seed

    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    weight_noise_rng = np.random.default_rng(seed)

    # Load network structure
    inputs_dir = input_dir / "inputs"
    network_structure = np.load(inputs_dir / "network_structure.npz")

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

    # Apply weight noise (same as training)
    if noise_frac > 0:
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
    sf_feedforward = np.array(scaling_factors_config["feedforward"])
    sf_recurrent = np.array(scaling_factors_config["recurrent"])
    np.concatenate([sf_feedforward, sf_recurrent], axis=0)

    # Load target scaling factors
    targets_dir = input_dir / "targets"
    target_data = np.load(targets_dir / "target_scaling_factors.npz")
    target_scaling_factors_FF = target_data["feedforward_scaling_factors"]

    # Compute perturbation (reciprocal of target)
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

    # Determine which scaling factors to use
    if scaling_factor_mode == "target":
        scaling_factors_to_use = target_scaling_factors_FF.copy()
        print("\nUsing TARGET scaling factors (should give optimal loss)")
    elif scaling_factor_mode == "learned":
        metrics_path = input_dir / "training_metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            final_row = df.iloc[-1]

            scaling_factors_to_use = np.zeros_like(target_scaling_factors_FF)

            input_cell_type_names = (
                feedforward.cell_types.names + recurrent.cell_types.names
            )
            output_cell_type_names = recurrent.cell_types.names

            for source_idx, source_name in enumerate(input_cell_type_names):
                for target_idx, target_name in enumerate(output_cell_type_names):
                    col_name = f"scaling_factors/{source_name}_to_{target_name}_value"
                    if col_name in final_row:
                        normalized_val = final_row[col_name]
                        scaling_factors_to_use[source_idx, target_idx] = (
                            normalized_val
                            * target_scaling_factors_FF[source_idx, target_idx]
                        )
            print("\nUsing LEARNED scaling factors from training")
        else:
            raise FileNotFoundError(f"No training metrics found at {metrics_path}")
    elif scaling_factor_mode == "ones":
        scaling_factors_to_use = np.ones_like(target_scaling_factors_FF)
        print("\nUsing ONES (no correction)")
    else:
        raise ValueError(f"Unknown scaling_factor_mode: {scaling_factor_mode}")

    print(f"Scaling factors shape: {scaling_factors_to_use.shape}")
    print(f"Scaling factors mean: {scaling_factors_to_use.mean():.4f}")

    # Load dataset
    spike_dataset = PrecomputedSpikeDataset(
        spike_data_path=inputs_dir / "spike_data.zarr",
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
        scaling_factors_FF=scaling_factors_to_use,
        optimisable=None,
        feedforward_mask=concatenated_mask,
        track_variables=False,
        use_tqdm=False,
    )
    model.to(device)
    model.eval()

    # Setup loss function
    van_rossum_loss_fn = VanRossumLoss(
        tau_rise=hyperparameters.van_rossum_tau_rise,
        tau_decay=hyperparameters.van_rossum_tau_decay,
        dt=dt,
        window_size=chunk_size,
        device=device,
    )

    # Run inference
    print(f"\nRunning inference for {num_chunks} chunks...")
    model.reset_state(batch_size=batch_size)
    van_rossum_loss_fn.reset_state()

    all_losses = []
    all_spikes = []
    all_targets = []
    student_firing_rates = []
    teacher_firing_rates = []

    data_iter = iter(spike_dataloader)

    with torch.no_grad():
        for chunk_idx in tqdm(range(num_chunks), desc="Inference"):
            batch = next(data_iter)
            input_spikes = batch.input_spikes
            target_spikes = batch.target_spikes

            output = model(input_spikes)
            spikes = output["spikes"]

            loss = van_rossum_loss_fn(spikes, target_spikes)
            all_losses.append(loss.item())

            if chunk_idx >= num_chunks - 5:
                all_spikes.append(spikes.cpu().numpy())
                all_targets.append(target_spikes.cpu().numpy())

            duration_s = chunk_size * dt / 1000.0
            student_fr = spikes[0].sum(dim=0).cpu().numpy() / duration_s
            teacher_fr = target_spikes[0].sum(dim=0).cpu().numpy() / duration_s
            student_firing_rates.append(student_fr)
            teacher_firing_rates.append(teacher_fr)

    # Compute statistics
    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)

    student_fr_all = np.concatenate(
        [fr.reshape(1, -1) for fr in student_firing_rates], axis=0
    )
    teacher_fr_all = np.concatenate(
        [fr.reshape(1, -1) for fr in teacher_firing_rates], axis=0
    )

    student_mean_fr = student_fr_all.mean()
    teacher_mean_fr = teacher_fr_all.mean()

    exc_mask = cell_type_indices == 0
    inh_mask = cell_type_indices == 1

    student_exc_fr = student_fr_all[:, exc_mask].mean()
    student_inh_fr = student_fr_all[:, inh_mask].mean()
    teacher_exc_fr = teacher_fr_all[:, exc_mask].mean()
    teacher_inh_fr = teacher_fr_all[:, inh_mask].mean()

    print("\n" + "=" * 60)
    print(f"RESULTS ({scaling_factor_mode} scaling factors)")
    print("=" * 60)
    print(f"Loss: {mean_loss:.6f} +/- {std_loss:.6f}")
    print("\nFiring rates:")
    print(f"  Student (all): {student_mean_fr:.3f} Hz")
    print(f"  Teacher (all): {teacher_mean_fr:.3f} Hz")
    print(f"  Student (exc): {student_exc_fr:.3f} Hz")
    print(f"  Teacher (exc): {teacher_exc_fr:.3f} Hz")
    print(f"  Student (inh): {student_inh_fr:.3f} Hz")
    print(f"  Teacher (inh): {teacher_inh_fr:.3f} Hz")
    print("=" * 60)

    # Generate plots
    mode_output_dir = output_dir / scaling_factor_mode
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    if all_spikes:
        spikes_concat = np.concatenate(all_spikes, axis=1)
        targets_concat = np.concatenate(all_targets, axis=1)

        n_plot = min(10, spikes_concat.shape[2])
        interleaved = np.zeros((1, spikes_concat.shape[1], 2 * n_plot))
        for i in range(n_plot):
            interleaved[0, :, 2 * i] = targets_concat[0, :, i]
            interleaved[0, :, 2 * i + 1] = spikes_concat[0, :, i]

        cell_type_indices_plot = np.array([0, 1] * n_plot)
        cell_type_names_plot = ["Teacher", "Student"]

        fig = plot_spike_trains(
            spikes=interleaved,
            dt=dt,
            cell_type_indices=cell_type_indices_plot,
            cell_type_names=cell_type_names_plot,
            n_neurons_plot=2 * n_plot,
            fraction=1.0,
            random_seed=None,
            title=f"Spike Comparison ({scaling_factor_mode} SF, loss={mean_loss:.4f})",
            ylabel="Neuron",
            figsize=(14, 8),
        )
        fig.savefig(
            mode_output_dir / "spike_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(all_losses, "b-", alpha=0.7)
    ax.axhline(mean_loss, color="r", linestyle="--", label=f"Mean: {mean_loss:.4f}")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss per Chunk ({scaling_factor_mode} scaling factors)")
    ax.legend()
    fig.savefig(mode_output_dir / "loss_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    results = {
        "scaling_factor_mode": scaling_factor_mode,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "student_mean_fr": student_mean_fr,
        "teacher_mean_fr": teacher_mean_fr,
        "student_exc_fr": student_exc_fr,
        "teacher_exc_fr": teacher_exc_fr,
        "student_inh_fr": student_inh_fr,
        "teacher_inh_fr": teacher_inh_fr,
        "num_chunks": num_chunks,
    }

    np.savez(mode_output_dir / "results.npz", **results)
    print(f"\nResults saved to {mode_output_dir}")

    return results


def main(
    input_dir,
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
):
    """Main execution function.

    Args:
        input_dir (Path): Path to trained noisy weights experiment directory
        output_dir (Path): Directory where results will be saved
        params_file (Path): Path to parameters TOML file (only used for analysis config;
            all other parameters loaded from the experiment's own parameters.toml)
        wandb_config (dict, optional): W&B configuration (not used)
        resume_from (Path, optional): Not used by this script
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    input_dir = Path(input_dir)

    # Load parameters from the experiment being analyzed
    experiment_params_file = input_dir / "parameters.toml"
    with open(experiment_params_file, "r") as f:
        data = toml.load(f)

    simulation = StudentSimulationConfig(**data["simulation"])
    hyperparameters = StudentHyperparameters(**data["hyperparameters"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])

    # Analysis-specific parameters from params_file (optional overrides)
    mode = "all"
    num_chunks = 50
    if params_file is not None:
        with open(params_file, "r") as f:
            analysis_data = toml.load(f)
        analysis_config = analysis_data.get("analysis", {})
        mode = analysis_config.get("mode", mode)
        num_chunks = analysis_config.get("num_chunks", num_chunks)

    # Set seed
    if simulation.seed is not None:
        np.random.seed(simulation.seed)
        torch.manual_seed(simulation.seed)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "all":
        results = {}
        for sf_mode in ["target", "learned", "ones"]:
            print(f"\n{'#' * 60}")
            print(f"# Testing {sf_mode.upper()} scaling factors")
            print(f"{'#' * 60}")
            results[sf_mode] = run_inference_with_scaling_factors(
                input_dir=input_dir,
                output_dir=output_dir,
                scaling_factor_mode=sf_mode,
                num_chunks=num_chunks,
                recurrent=recurrent,
                feedforward=feedforward,
                hyperparameters=hyperparameters,
                simulation=simulation,
            )

        # Print comparison
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Mode':<12} {'Loss':<20} {'Student FR':<15} {'Teacher FR':<15}")
        print("-" * 80)
        for sf_mode, res in results.items():
            print(
                f"{sf_mode:<12} {res['mean_loss']:.6f} +/- {res['std_loss']:.4f}  "
                f"{res['student_mean_fr']:.3f} Hz       {res['teacher_mean_fr']:.3f} Hz"
            )
        print("=" * 80)

        best_mode = min(results, key=lambda m: results[m]["mean_loss"])
        print(f"\nBest loss: {best_mode} ({results[best_mode]['mean_loss']:.6f})")

        if best_mode != "target":
            print(
                f"\n*** NOTE: {best_mode} scaling factors give BETTER loss than target! ***"
            )
            print("This suggests the loss landscape has a different optimum than SF=1.")

        # Save comparison summary
        summary = {
            "best_mode": best_mode,
            "modes": list(results.keys()),
            "losses": [results[m]["mean_loss"] for m in results],
        }
        np.savez(output_dir / "comparison_summary.npz", **summary)
    else:
        run_inference_with_scaling_factors(
            input_dir=input_dir,
            output_dir=output_dir,
            scaling_factor_mode=mode,
            num_chunks=num_chunks,
            recurrent=recurrent,
            feedforward=feedforward,
            hyperparameters=hyperparameters,
            simulation=simulation,
        )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    pass

"""
Compare trained student network activity against teacher network activity.

This script loads a trained student network from a checkpoint and generates
spike trains using the same input patterns as the teacher network. It then
compares the student's output spike trains with the teacher's target spike trains
using cross-correlation analysis and visualization to evaluate training success.

The script generates comparison plots including cross-correlation histograms
and scatter plots to quantify how well the student matches the teacher.
"""

import numpy as np
import torch
import toml
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from inputs.dataloaders import PrecomputedSpikeDataset
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from parameter_loaders import StudentTrainingParams
from visualization.firing_statistics import (
    plot_cross_correlation_histogram,
    plot_cross_correlation_scatter,
)


def main(experiment_dir, output_dir):
    """Generate comparison plots between student and teacher network activity.

    Args:
        experiment_dir (Path): Directory containing training run outputs
            Expected structure:
                - input/ : Contains teacher network_structure.npz and spike_data.zarr
                - checkpoints/ : Contains checkpoint_best.pt
                - parameters.toml : Training parameters
        output_dir (Path): Directory where comparison plots will be saved
    """

    # ======================================
    # Device Selection and Parameter Loading
    # ======================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load parameters from the training run
    params_file = experiment_dir / "parameters.toml"
    with open(params_file, "r") as f:
        data = toml.load(f)
    params = StudentTrainingParams(**data)

    simulation = params.simulation
    recurrent = params.recurrent
    feedforward = params.feedforward

    # Set random seed for reproducibility
    if simulation.seed is not None:
        np.random.seed(simulation.seed)
        torch.manual_seed(simulation.seed)
        print(f"Using seed: {simulation.seed}")

    # ================================
    # Load Original Network Structure
    # ================================

    input_dir = experiment_dir / "inputs"
    network_structure = np.load(input_dir / "network_structure.npz")

    # Extract network components (original unperturbed network)
    original_weights = network_structure["recurrent_weights"]
    original_feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    feedforward_cell_type_indices = network_structure["feedforward_cell_type_indices"]

    # Load assembly structure if it exists
    assembly_ids = network_structure.get("assembly_ids")
    if assembly_ids is not None:
        n_assemblies = len(np.unique(assembly_ids[assembly_ids >= 0]))
        print(
            f"✓ Loaded original network with {len(cell_type_indices)} neurons ({n_assemblies} assemblies)"
        )
    else:
        print(
            f"✓ Loaded original network with {len(cell_type_indices)} neurons (no assembly structure)"
        )

    # ======================
    # Load Dataset from Disk
    # ======================

    spike_dataset = PrecomputedSpikeDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=simulation.chunk_size,
        device=device,
    )

    # Calculate number of chunks to run based on training.plot_size
    training = params.training
    chunks_to_run = min(training.plot_size, spike_dataset.num_chunks)

    # Calculate simulation duration
    simulation_duration = chunks_to_run * simulation.chunk_size * spike_dataset.dt

    print(f"✓ Loaded spike dataset with {spike_dataset.num_chunks} chunks")
    print(
        f"  Running {chunks_to_run} chunks ({simulation_duration:.1f} ms) for comparison"
    )
    print(
        f"  Chunk size: {simulation.chunk_size:.1f} ms, Timestep: {spike_dataset.dt:.2f} ms"
    )

    # ====================================
    # Initialize Original Network and Run
    # ====================================

    print("\nInitializing original target network...")

    target_model = ConductanceLIFNetwork(
        dt=spike_dataset.dt,
        weights=original_weights,
        cell_type_indices=cell_type_indices,
        cell_params=recurrent.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        weights_FF=original_feedforward_weights,
        cell_type_indices_FF=feedforward_cell_type_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        surrgrad_scale=1.0,  # Not used for inference
        optimisable=None,  # No optimization for target network
        use_tqdm=False,
    ).to(device)

    print("  Running original network simulation...")

    # Initialize lists to accumulate results across batches and chunks
    all_target_spikes = []

    # Run inference in chunks, processing all batches and patterns
    with torch.inference_mode():
        for chunk_idx in tqdm(range(chunks_to_run), desc="Target network chunks"):
            # Get one chunk of input spikes from dataset (returns tuple)
            input_spikes_chunk, _ = spike_dataset[chunk_idx]

            # Dataset returns (batch, patterns, time, neurons)
            # Flatten batch and pattern dimensions: (batch * patterns, time, neurons)
            batch, patterns, time, n_inputs = input_spikes_chunk.shape
            input_spikes_flat = input_spikes_chunk.reshape(
                batch * patterns, time, n_inputs
            )

            # Initialize state variables for this chunk (all batch×pattern combinations)
            if chunk_idx == 0:
                initial_v = None
                initial_g = None
                initial_g_FF = None

            # Run one chunk of simulation
            (
                target_spikes_chunk,
                target_voltages,
                target_currents,
                target_currents_FF,
                target_currents_leak,
                target_conductances,
                target_conductances_FF,
            ) = target_model(
                input_spikes=input_spikes_flat,
                initial_v=initial_v,
                initial_g=initial_g,
                initial_g_FF=initial_g_FF,
            )

            # Store final states for next chunk
            initial_v = target_voltages[:, -1, :].clone()  # Last timestep voltages
            initial_g = target_conductances[
                :, -1, :, :, :
            ].clone()  # Last timestep conductances
            initial_g_FF = target_conductances_FF[
                :, -1, :, :, :
            ].clone()  # Last timestep FF conductances

            # Move to CPU and accumulate results
            if device == "cuda":
                all_target_spikes.append(target_spikes_chunk.cpu())
            else:
                all_target_spikes.append(target_spikes_chunk)

    # Concatenate all chunks along time dimension: (batch*patterns, total_time, n_neurons)
    target_spikes = torch.cat(all_target_spikes, dim=1)
    # Reshape back to (batch, patterns, total_time, n_neurons)
    target_spikes = target_spikes.reshape(batch, patterns, -1, target_spikes.shape[-1])

    print(f"✓ Generated target spikes: {target_spikes.shape}")

    # ====================================
    # Load Best Checkpoint and Run
    # ====================================

    checkpoint_path = experiment_dir / "checkpoints" / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found at {checkpoint_path}")

    print(f"\nLoading best checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Initialize trained model with same structure as target
    # The checkpoint will contain perturbed weights + learned scaling factors
    trained_model = ConductanceLIFNetwork(
        dt=spike_dataset.dt,
        weights=original_weights,  # Will be overwritten by checkpoint
        cell_type_indices=cell_type_indices,
        cell_params=recurrent.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        weights_FF=original_feedforward_weights,  # Will be overwritten by checkpoint
        cell_type_indices_FF=feedforward_cell_type_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        surrgrad_scale=1.0,  # Not used for inference
        optimisable="scaling_factors",  # Match training configuration
        use_tqdm=False,
    ).to(device)

    # Load trained weights from checkpoint
    trained_model.load_state_dict(checkpoint["model_state_dict"])

    print("  Running trained network simulation...")

    # Initialize lists to accumulate results across batches and chunks
    all_trained_spikes = []

    # Run inference in chunks, processing all batches and patterns
    with torch.inference_mode():
        for chunk_idx in tqdm(range(chunks_to_run), desc="Trained network chunks"):
            # Get one chunk of input spikes from dataset (returns tuple)
            input_spikes_chunk, _ = spike_dataset[chunk_idx]

            # Dataset returns (batch, patterns, time, neurons)
            # Flatten batch and pattern dimensions: (batch * patterns, time, neurons)
            batch, patterns, time, n_inputs = input_spikes_chunk.shape
            input_spikes_flat = input_spikes_chunk.reshape(
                batch * patterns, time, n_inputs
            )

            # Initialize state variables for this chunk (all batch×pattern combinations)
            if chunk_idx == 0:
                initial_v = None
                initial_g = None
                initial_g_FF = None

            # Run one chunk of simulation
            (
                trained_spikes_chunk,
                trained_voltages,
                trained_currents,
                trained_currents_FF,
                trained_currents_leak,
                trained_conductances,
                trained_conductances_FF,
            ) = trained_model(
                input_spikes=input_spikes_flat,
                initial_v=initial_v,
                initial_g=initial_g,
                initial_g_FF=initial_g_FF,
            )

            # Store final states for next chunk
            initial_v = trained_voltages[:, -1, :].clone()  # Last timestep voltages
            initial_g = trained_conductances[
                :, -1, :, :, :
            ].clone()  # Last timestep conductances
            initial_g_FF = trained_conductances_FF[
                :, -1, :, :, :
            ].clone()  # Last timestep FF conductances

            # Move to CPU and accumulate results
            if device == "cuda":
                all_trained_spikes.append(trained_spikes_chunk.cpu())
            else:
                all_trained_spikes.append(trained_spikes_chunk)

    # Concatenate all chunks along time dimension: (batch*patterns, total_time, n_neurons)
    trained_spikes = torch.cat(all_trained_spikes, dim=1)
    # Reshape back to (batch, patterns, total_time, n_neurons)
    trained_spikes = trained_spikes.reshape(
        batch, patterns, -1, trained_spikes.shape[-1]
    )

    print(f"✓ Generated trained spikes: {trained_spikes.shape}")

    # ====================================
    # Compare Outputs
    # ====================================

    print("\nGenerating comparison plots...")

    # Convert to numpy - now shape is (batch, patterns, total_time, n_neurons)
    target_spikes_np = target_spikes.cpu().numpy()
    trained_spikes_np = trained_spikes.cpu().numpy()

    # Flatten batch and pattern dimensions for plotting: (batch*patterns, total_time, n_neurons)
    target_spikes_np = target_spikes_np.reshape(
        -1, target_spikes_np.shape[2], target_spikes_np.shape[3]
    )
    trained_spikes_np = trained_spikes_np.reshape(
        -1, trained_spikes_np.shape[2], trained_spikes_np.shape[3]
    )

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Cross-correlation histogram (assembly-pooled, short windows)
    print("  Creating cross-correlation histogram...")
    fig_hist = plot_cross_correlation_histogram(
        spike_trains_trial1=target_spikes_np,
        spike_trains_trial2=trained_spikes_np,
        window_size=0.05,  # 50ms windows (matches generate_teacher_activity.py)
        dt=spike_dataset.dt,
        bin_size=0.1,  # 0.1 Hz bins (matches generate_teacher_activity.py)
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        excitatory_idx=0,
        title="Target vs Trained Network Activity (Assembly-pooled)",
        x_label="Target Network Firing Rate (Hz)",
        y_label="Trained Network Firing Rate (Hz)",
    )

    if fig_hist is not None:
        fig_hist.savefig(
            output_dir / "cross_correlation_histogram.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig_hist)
        print(f"  ✓ Saved: {output_dir / 'cross_correlation_histogram.png'}")

    # Plot 2: Cross-correlation scatter plot (all neurons, long windows)
    print("  Creating cross-correlation scatter plot...")
    fig_scatter = plot_cross_correlation_scatter(
        spike_trains_trial1=target_spikes_np,
        spike_trains_trial2=trained_spikes_np,
        window_size=10.0,  # 10 second windows (matches generate_teacher_activity.py)
        dt=spike_dataset.dt,
        title="Target vs Trained Network Activity (Per Neuron)",
        x_label="Target Network Firing Rate (Hz)",
        y_label="Trained Network Firing Rate (Hz)",
    )

    if fig_scatter is not None:
        fig_scatter.savefig(
            output_dir / "cross_correlation_scatter.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig_scatter)
        print(f"  ✓ Saved: {output_dir / 'cross_correlation_scatter.png'}")

    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare trained network activity with target network activity"
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment directory containing input/, checkpoints/, and parameters/",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to directory where comparison plots will be saved",
    )

    args = parser.parse_args()

    main(
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
    )

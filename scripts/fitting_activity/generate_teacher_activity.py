"""
Generating spike trains from a synthetic connectome

This script generates spiketrains from a predefined synthetic connectome
to be used as teacher activity for fitting recurrent networks.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import toml
from inputs.dataloaders import (
    PoissonSpikeDataset,
    collate_pattern_batches,
    generate_odour_firing_rates,
)
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from torch.utils.data import DataLoader
from parameter_loaders import TeacherActivityParams
from tqdm import tqdm
from visualization import plot_input_firing_rate_histogram


def main(input_dir, output_dir, params_file):
    """Main execution function for Dp network simulation.

    Args:
        input_dir (Path, optional): Directory containing input data files (may be None)
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
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
    params = TeacherActivityParams(**data)

    # Extract commonly used parameter groups
    simulation = params.simulation
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

    print("Loading network structure from disk...")
    network_structure = np.load(input_dir / "network_structure.npz")

    # Extract network components
    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    input_source_indices = network_structure["feedforward_cell_type_indices"]

    print(f"✓ Loaded network with {len(cell_type_indices)} neurons")
    print(f"✓ Loaded recurrent weights: {weights.shape}")
    print(f"✓ Loaded feedforward weights: {feedforward_weights.shape}")

    # =========================
    # Create Feedforward Inputs
    # =========================

    # Generate odour-modulated firing rate patterns
    input_firing_rates = generate_odour_firing_rates(
        n_input_neurons=feedforward_weights.shape[0],
        input_source_indices=input_source_indices,
        cell_type_names=feedforward.cell_types.names,
        odour_configs=params.odours,
        n_patterns=simulation.num_odours,
    )

    n_patterns = simulation.num_odours
    batch_size = simulation.batch_size

    # Create Poisson dataset for multiple patterns
    # Dataset cycles through patterns indefinitely
    spike_dataset = PoissonSpikeDataset(
        firing_rates=input_firing_rates,
        chunk_size=simulation.chunk_size,
        dt=simulation.dt,
        device=device,
    )

    # DataLoader: batch_size * n_patterns items fetched per iteration
    # Result shape: (batch_size, n_patterns, n_steps, n_neurons)
    # batch_size = number of repeats/trials, n_patterns = number of different patterns
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=batch_size * n_patterns,  # Fetch batch_size repeats of all patterns
        shuffle=False,
        collate_fn=collate_pattern_batches,
        num_workers=0,  # Keep 0 for GPU generation
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
        surrgrad_scale=1.0,  # Not used for inference, but required parameter
        weights_FF=feedforward_weights,
        cell_type_indices_FF=input_source_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        optimisable=None,  # No optimization for inference
        use_tqdm=False,  # Disable model's internal progress bar
    )

    # Move model to device for GPU acceleration
    model.to(device)
    print(f"Model moved to device: {device}")

    # Jit compile the model for faster execution
    model.compile_step()
    print("✓ Model JIT compiled for faster execution")

    print("\n" + "=" * len("STARTING CHUNKED NETWORK SIMULATION"))
    print("STARTING CHUNKED NETWORK SIMULATION")
    print("=" * len("STARTING CHUNKED NETWORK SIMULATION"))

    # ==============================
    # Run Chunked Network Simulation
    # ==============================

    print(f"Running simulation with {n_patterns} patterns × {batch_size} repeats")
    print(f"Processing {simulation.num_chunks} chunks...")
    print(f"Total simulation duration: {simulation.total_duration_s:.2f} s")
    print(
        f"DataLoader returns (batch_size={batch_size}, n_patterns={n_patterns}, chunk_size, neurons) per chunk"
    )

    # Initialize storage with explicit batch/pattern dimensions
    # Shape: (batch_size, n_patterns, time, neurons)
    n_neurons = len(cell_type_indices)
    n_input_neurons = feedforward_weights.shape[0]

    # Storage for chunks
    all_output_spikes = []
    all_input_spikes = []

    # Initialize state variables: (batch_size, n_patterns, ...)
    initial_v = None
    initial_g = None
    initial_g_FF = None

    # Run inference in chunks using DataLoader
    with torch.inference_mode():
        dataloader_iter = iter(spike_dataloader)

        for chunk_idx in tqdm(range(simulation.num_chunks), desc="Processing chunks"):
            # Get next chunk: (batch_size, n_patterns, n_steps, n_input_neurons)
            input_spikes_chunk, pattern_indices = next(dataloader_iter)

            # Flatten batch and pattern dimensions for SNN forward pass
            # From (batch_size, n_patterns, n_steps, n_input_neurons) to (batch_size*n_patterns, n_steps, n_input_neurons)
            input_spikes_flat = input_spikes_chunk.reshape(
                batch_size * n_patterns, -1, n_input_neurons
            )

            # Run one chunk of simulation
            (
                output_spikes_chunk_flat,
                output_voltages_chunk_flat,
                output_currents_chunk_flat,
                output_currents_FF_chunk_flat,
                output_currents_leak_chunk_flat,
                output_conductances_chunk_flat,
                output_conductances_FF_chunk_flat,
            ) = model.forward(
                input_spikes=input_spikes_flat,
                initial_v=initial_v,
                initial_g=initial_g,
                initial_g_FF=initial_g_FF,
            )

            # Unflatten outputs back to (batch_size, n_patterns, ...)
            n_steps_out = output_spikes_chunk_flat.shape[1]
            output_spikes_chunk = output_spikes_chunk_flat.reshape(
                batch_size, n_patterns, n_steps_out, n_neurons
            )

            # Store final states for next chunk: (batch_size*n_patterns, neurons) or (batch_size*n_patterns, neurons, syn_types)
            initial_v = output_voltages_chunk_flat[:, -1, :].clone()
            initial_g = output_conductances_chunk_flat[:, -1, :, :].clone()
            initial_g_FF = output_conductances_FF_chunk_flat[:, -1, :, :].clone()

            # Move to CPU and accumulate results
            if device == "cuda":
                all_output_spikes.append(output_spikes_chunk.bool().cpu())
                all_input_spikes.append(input_spikes_chunk.cpu())
            else:
                all_output_spikes.append(output_spikes_chunk.bool())
                all_input_spikes.append(input_spikes_chunk)

    # Concatenate chunks along time dimension: (batch_size, n_patterns, time, neurons)
    output_spikes = torch.cat(all_output_spikes, dim=2).numpy()
    input_spikes = torch.cat(all_input_spikes, dim=2).numpy()

    print("\n✓ Network simulation completed!")
    print(f"Final output shape: {output_spikes.shape}")
    print(
        f"  (batch_size={output_spikes.shape[0]}, n_patterns={output_spikes.shape[1]}, time={output_spikes.shape[2]}, neurons={output_spikes.shape[3]})"
    )

    # =====================================
    # Save Output Data for Further Analysis
    # =====================================
    print("Saving simulation data...")

    # Create results directory if it doesn't exist
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save spike trains and patterns
    # input_spikes: (batch_size, n_patterns, time, n_input_neurons)
    # output_spikes: (batch_size, n_patterns, time, neurons)
    # input_firing_rates: (n_patterns, n_input_neurons) - defines the patterns
    np.savez_compressed(
        results_dir / "spike_data.npz",
        input_spikes=input_spikes.astype(np.bool_),
        output_spikes=output_spikes.astype(np.bool_),
        input_firing_rates=input_firing_rates,
    )

    print(f"✓ Saved data to {results_dir}")

    # Free GPU memory before generating dashboards
    if device == "cuda":
        del model
        torch.cuda.empty_cache()
        print("✓ Freed GPU memory")

    # ========================================
    # Generate Visualizations - Input Patterns
    # ========================================
    print("\nGenerating input pattern visualizations...")

    # Create figures directory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot input firing rate histogram
    fig = plot_input_firing_rate_histogram(
        input_spikes=input_spikes,
        dt=simulation.dt,
    )
    fig.savefig(
        figures_dir / "input_firing_rate_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"✓ Saved input firing rate histogram to {figures_dir}")

    # ========
    # Clean Up
    # ========

    print("\n" + "=" * len("SIMULATION COMPLETE!"))
    print("SIMULATION COMPLETE!")
    print("=" * len("SIMULATION COMPLETE!"))
    print(f"✓ Results saved to {output_dir / 'results'}")
    print(f"✓ Figures saved to {output_dir / 'figures'}")
    print(f"✓ Statistics saved to {output_dir / 'network_statistics.csv'}")
    print("=" * len("SIMULATION COMPLETE!"))

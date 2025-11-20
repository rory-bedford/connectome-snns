"""
Generating spike trains from a synthetic connectome - CONTROL EXPERIMENT

This script generates spiketrains from a predefined synthetic connectome
with unmodulated (constant baseline) inputs as a control condition.
Unlike the main experiment, inputs do not have odour-specific modulation.
"""

import numpy as np
import torch
import toml
import zarr
from inputs.dataloaders import (
    PoissonSpikeDataset,
    collate_pattern_batches,
    generate_baseline_firing_rates,
)
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from torch.utils.data import DataLoader
from parameter_loaders import TeacherActivityParams
from tqdm import tqdm


def main(input_dir, output_dir, params_file):
    """Main execution function for control experiment with unmodulated inputs.

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

    # Generate baseline (unmodulated) firing rates - CONTROL CONDITION
    input_firing_rates = generate_baseline_firing_rates(
        n_input_neurons=feedforward_weights.shape[0],
        input_source_indices=input_source_indices,
        cell_type_names=feedforward.cell_types.names,
        odour_configs=params.odours,
    )

    print("CONTROL EXPERIMENT: Using unmodulated baseline inputs")
    print(f"  Input firing rates shape: {input_firing_rates.shape}")
    print("  All neurons set to baseline rate (no modulation)")

    # Single pattern, same batch size as main experiment
    n_patterns = 1
    batch_size = simulation.batch_size

    # Create Poisson dataset with single baseline pattern
    spike_dataset = PoissonSpikeDataset(
        firing_rates=input_firing_rates,
        chunk_size=simulation.chunk_size,
        dt=simulation.dt,
        device=device,
    )

    # DataLoader: batch_size * n_patterns items fetched per iteration
    # Result shape: (batch_size, n_patterns, n_steps, n_neurons)
    # n_patterns = 1 for control (no pattern variation)
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=batch_size
        * n_patterns,  # Fetch batch_size repeats of single pattern
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

    # ======================================
    # Initialize Zarr Storage for Spike Data
    # ======================================

    # Initialize storage with explicit batch/pattern dimensions
    # Shape: (batch_size, n_patterns, time, neurons)
    n_neurons = len(cell_type_indices)
    n_input_neurons = feedforward_weights.shape[0]

    # Create results directory if it doesn't exist
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Zarr arrays with compression
    total_steps = simulation.num_chunks * simulation.chunk_size

    # Create zarr group (directory-based storage) - CONTROL VERSION
    root = zarr.open_group(results_dir / "spike_data_control.zarr", mode="w")

    # Create datasets with chunking optimized for both writing and reading
    # Chunks span all (batch, pattern) combinations for one time slice
    output_spikes_zarr = root.create_dataset(
        "output_spikes",
        shape=(batch_size, n_patterns, total_steps, n_neurons),
        chunks=(batch_size, n_patterns, simulation.chunk_size, n_neurons),
        dtype=np.bool_,
    )

    input_spikes_zarr = root.create_dataset(
        "input_spikes",
        shape=(batch_size, n_patterns, total_steps, n_input_neurons),
        chunks=(batch_size, n_patterns, simulation.chunk_size, n_input_neurons),
        dtype=np.bool_,
    )

    chunk_size_mb = (batch_size * n_patterns * simulation.chunk_size * n_neurons) / (
        1024**2
    )
    print(
        f"Zarr chunk shape: ({batch_size}, {n_patterns}, {simulation.chunk_size}, {n_neurons})"
    )
    print(f"  Chunk size: ~{chunk_size_mb:.1f} MB uncompressed")

    # ==============================
    # Run Chunked Network Simulation
    # ==============================

    print("\n" + "=" * len("STARTING CONTROL EXPERIMENT SIMULATION"))
    print("STARTING CONTROL EXPERIMENT SIMULATION")
    print("=" * len("STARTING CONTROL EXPERIMENT SIMULATION"))

    print(
        f"Running simulation with {n_patterns} pattern (baseline) × {batch_size} repeats"
    )
    print(f"Processing {simulation.num_chunks} chunks...")
    print(f"Total simulation duration: {simulation.total_duration_s:.2f} s")
    print(
        f"DataLoader returns (batch_size={batch_size}, n_patterns={n_patterns}, chunk_size, neurons) per chunk"
    )

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

            # Write directly to disk using Zarr - NO RAM accumulation
            start_idx = chunk_idx * simulation.chunk_size
            end_idx = start_idx + n_steps_out

            if device == "cuda":
                output_spikes_zarr[:, :, start_idx:end_idx, :] = (
                    output_spikes_chunk.bool().cpu().numpy()
                )
                input_spikes_zarr[:, :, start_idx:end_idx, :] = (
                    input_spikes_chunk.cpu().numpy()
                )
            else:
                output_spikes_zarr[:, :, start_idx:end_idx, :] = (
                    output_spikes_chunk.bool().numpy()
                )
                input_spikes_zarr[:, :, start_idx:end_idx, :] = (
                    input_spikes_chunk.numpy()
                )

    print("\n✓ Network simulation completed!")
    print(f"Final output shape: {output_spikes_zarr.shape}")
    print(
        f"  (batch_size={output_spikes_zarr.shape[0]}, n_patterns={output_spikes_zarr.shape[1]}, time={output_spikes_zarr.shape[2]}, neurons={output_spikes_zarr.shape[3]})"
    )

    # =====================================
    # Save Output Data for Further Analysis
    # =====================================
    print("Saving input firing rates...")

    # Spike data already saved incrementally to Zarr during simulation
    # Save input_firing_rates separately as a Zarr array
    root.create_dataset(
        "input_firing_rates",
        shape=input_firing_rates.shape,
        dtype=input_firing_rates.dtype,
        data=input_firing_rates,
    )

    print(f"✓ Saved spike data to {results_dir / 'spike_data_control.zarr'}")
    print(f"  - output_spikes: {output_spikes_zarr.shape}")
    print(f"  - input_spikes: {input_spikes_zarr.shape}")
    print(f"  - input_firing_rates: {input_firing_rates.shape}")

    # Free GPU memory
    if device == "cuda":
        del model
        torch.cuda.empty_cache()
        print("✓ Freed GPU memory")

    # ========
    # Clean Up
    # ========

    print("\n" + "=" * len("CONTROL EXPERIMENT COMPLETE!"))
    print("CONTROL EXPERIMENT COMPLETE!")
    print("=" * len("CONTROL EXPERIMENT COMPLETE!"))
    print(f"✓ Spike data saved to {results_dir / 'spike_data_control.zarr'}")
    print(
        f"✓ Data can be loaded with: zarr.open('{results_dir / 'spike_data_control.zarr'}', mode='r')"
    )
    print("=" * len("CONTROL EXPERIMENT COMPLETE!"))

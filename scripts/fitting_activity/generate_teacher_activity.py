"""
Generating teacher activity from optimized connectome

This script loads a pre-trained connectome (typically from homeostatic_plasticity
training) and generates teacher spike train patterns using odour-modulated inputs.
These patterns serve as targets for student network training.

Workflow position: Stage 3 (after network_inference → homeostatic_plasticity)
"""

import numpy as np
import torch
import toml
import zarr
import matplotlib.pyplot as plt
from src.network_inputs.unsupervised import (
    InhomogeneousPoissonSpikeDataLoader,
)
from src.network_inputs.rate_processes import OrnsteinUhlenbeckRateProcess
from src.network_inputs.odourants import (
    generate_odour_firing_rates,
)
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from parameter_loaders import TeacherActivityParams
from tqdm import tqdm
from visualization.dashboards import (
    create_connectivity_dashboard,
    create_activity_dashboard,
    create_assembly_activity_dashboard,
)


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
    assembly_ids = network_structure["assembly_ids"]

    print(f"✓ Loaded network with {len(cell_type_indices)} neurons")
    print(f"✓ Loaded recurrent weights: {weights.shape}")
    print(f"✓ Loaded feedforward weights: {feedforward_weights.shape}")
    print(
        f"✓ Loaded assembly IDs: {len(assembly_ids)} neurons in {len(np.unique(assembly_ids[assembly_ids >= 0]))} assemblies"
    )

    # Derive connectivity masks from weight matrices
    connectome_mask = (weights != 0).astype(np.bool_)
    feedforward_mask = (feedforward_weights != 0).astype(np.bool_)

    # =========================
    # Create Feedforward Inputs
    # =========================

    # Calculate number of odour patterns from assemblies
    num_odours = len(np.unique(assembly_ids[assembly_ids >= 0]))

    # Generate odour-modulated firing rate patterns (one per assembly)
    # Target excitatory cells (index 0) when computing connection strengths
    input_firing_rates_odour = generate_odour_firing_rates(
        feedforward_weights=feedforward_weights,
        input_source_indices=input_source_indices,
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        target_cell_type_idx=0,
        cell_type_names=feedforward.cell_types.names,
        odour_configs=params.get_odour_configs_dict(),
    )

    print(f"✓ Generated {num_odours} odour patterns")
    print(f"  Total patterns: {input_firing_rates_odour.shape[0]}")

    batch_size = simulation.batch_size

    # Create Ornstein-Uhlenbeck rate process in pattern space
    # This modulates the odour patterns dynamically over time
    rate_process = OrnsteinUhlenbeckRateProcess(
        patterns=input_firing_rates_odour,  # Shape: (n_patterns, n_input_neurons)
        chunk_size=int(simulation.chunk_size),
        dt=simulation.dt,
        tau=params.tau,  # Time constant for mean reversion (ms)
        temperature=params.temperature,  # Softmax temperature for pattern mixing
        sigma=params.sigma,  # Noise amplitude
        a_init=None,  # Initialize to ones
        return_rates=True,  # Return rates for storage and diagnostics
    )

    # Create inhomogeneous Poisson spike dataloader with OU-driven rates
    spike_dataloader = InhomogeneousPoissonSpikeDataLoader(
        rate_process=rate_process,
        batch_size=batch_size,
        device=device,
        return_rates=True,  # Enable rate output
    )

    print("✓ Created inhomogeneous Poisson spike dataloader with OU rate process")

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

    # Create zarr group (directory-based storage)
    root = zarr.open_group(results_dir / "spike_data.zarr", mode="w")

    # Store dt as a metadata attribute
    root.attrs["dt"] = simulation.dt

    # Save input firing rates immediately (the base patterns before OU modulation)
    root.create_dataset(
        "input_firing_rates",
        shape=input_firing_rates_odour.shape,
        dtype=input_firing_rates_odour.dtype,
        data=input_firing_rates_odour,
    )
    print("✓ Saved input firing rates to zarr")
    print(f"  - input_firing_rates (odour patterns): {input_firing_rates_odour.shape}")

    # Create datasets for OU-modulated spike data
    # Shape: (batch_size, total_steps, n_neurons) since OU process gives independent temporal dynamics per batch
    output_spikes = root.create_dataset(
        "output_spikes",
        shape=(batch_size, total_steps, n_neurons),
        chunks=(batch_size, simulation.chunk_size, n_neurons),
        dtype=np.bool_,
    )

    input_spikes = root.create_dataset(
        "input_spikes",
        shape=(batch_size, total_steps, n_input_neurons),
        chunks=(batch_size, simulation.chunk_size, n_input_neurons),
        dtype=np.bool_,
    )

    # Create datasets for OU-modulated firing rates
    input_rates = root.create_dataset(
        "input_rates",
        shape=(batch_size, total_steps, n_input_neurons),
        chunks=(batch_size, simulation.chunk_size, n_input_neurons),
        dtype=np.float32,
    )

    print(f"Zarr chunk shape: ({batch_size}, {simulation.chunk_size}, {n_neurons})")

    # ==============================
    # Run Chunked Network Simulation
    # ==============================

    print("\n" + "=" * len("STARTING CHUNKED NETWORK SIMULATION"))
    print("STARTING CHUNKED NETWORK SIMULATION")
    print("=" * len("STARTING CHUNKED NETWORK SIMULATION"))

    print(f"Running simulation with {batch_size} independent OU-modulated trajectories")
    print(f"Processing {simulation.num_chunks} chunks...")
    print(f"Total simulation duration: {simulation.total_duration_s:.2f} s")
    print(
        f"DataLoader returns (batch_size={batch_size}, chunk_size, n_input_neurons) per iteration"
    )

    # Initialize state variables: (batch_size, n_patterns, ...)
    initial_v = None
    initial_g = None
    initial_g_FF = None

    # Variables to store traces from last plot_size chunks for visualization
    viz_chunks = simulation.plot_size
    viz_duration_s = viz_chunks * simulation.chunk_duration_s
    viz_start_chunk = max(0, simulation.num_chunks - viz_chunks)

    # Store pattern 0 (first odour) for dashboards
    viz_voltages = []
    viz_currents = []
    viz_currents_FF = []
    viz_currents_leak = []
    viz_conductances = []
    viz_conductances_FF = []
    viz_output_spikes = []
    viz_input_spikes = []
    viz_input_rates = []

    # Store pattern 0 from batch 1 (second repeat) for cross-correlation
    viz_output_spikes_signal_repeat = []

    # Run inference in chunks using DataLoader
    with torch.inference_mode():
        for chunk_idx, batch_data in enumerate(
            tqdm(
                spike_dataloader, total=simulation.num_chunks, desc="Processing chunks"
            )
        ):
            # Unpack batch data: (spikes, rates, weights)
            input_spikes_chunk, input_rates_chunk, _ = batch_data

            # Run one chunk of simulation
            (
                output_spikes_chunk,
                output_voltages_chunk,
                output_currents_chunk,
                output_currents_FF_chunk,
                output_currents_leak_chunk,
                output_conductances_chunk,
                output_conductances_FF_chunk,
            ) = model.forward(
                input_spikes=input_spikes_chunk,
                initial_v=initial_v,
                initial_g=initial_g,
                initial_g_FF=initial_g_FF,
            )

            # Store traces from last few chunks for visualization
            if chunk_idx >= viz_start_chunk:
                # Extract first batch item for visualization
                viz_voltages.append(output_voltages_chunk[0:1, :, :].clone())
                viz_currents.append(output_currents_chunk[0:1, :, :, :].clone())
                viz_currents_FF.append(output_currents_FF_chunk[0:1, :, :, :].clone())
                viz_currents_leak.append(output_currents_leak_chunk[0:1, :, :].clone())
                viz_conductances.append(output_conductances_chunk[0:1, :, :, :].clone())
                viz_conductances_FF.append(
                    output_conductances_FF_chunk[0:1, :, :, :].clone()
                )
                viz_output_spikes.append(output_spikes_chunk[0:1, :, :].clone())
                viz_input_spikes.append(input_spikes_chunk[0:1, :, :].clone())
                viz_input_rates.append(input_rates_chunk[0:1, :, :].clone())

                # Store second batch item for repeatability comparison
                if batch_size > 1:
                    viz_output_spikes_signal_repeat.append(
                        output_spikes_chunk[1:2, :, :].clone()
                    )

            # Store final states for next chunk
            initial_v = output_voltages_chunk[:, -1, :].clone()
            initial_g = output_conductances_chunk[:, -1, :, :].clone()
            initial_g_FF = output_conductances_FF_chunk[:, -1, :, :].clone()

            # Write directly to zarr arrays - stream to disk, no RAM accumulation!
            start_idx = chunk_idx * simulation.chunk_size
            end_idx = start_idx + output_spikes_chunk.shape[1]

            if device == "cuda":
                output_spikes[:, start_idx:end_idx, :] = (
                    output_spikes_chunk.bool().cpu().numpy()
                )
                input_spikes[:, start_idx:end_idx, :] = input_spikes_chunk.cpu().numpy()
                input_rates[:, start_idx:end_idx, :] = input_rates_chunk.cpu().numpy()
            else:
                output_spikes[:, start_idx:end_idx, :] = (
                    output_spikes_chunk.bool().numpy()
                )
                input_spikes[:, start_idx:end_idx, :] = input_spikes_chunk.numpy()
                input_rates[:, start_idx:end_idx, :] = input_rates_chunk.numpy()

    print("\n✓ Network simulation completed!")
    print(f"\n✓ All data saved to {results_dir / 'spike_data.zarr'}")
    print(f"  - output_spikes: {output_spikes.shape}")
    print(f"  - input_spikes: {input_spikes.shape}")
    print(f"  - input_rates: {input_rates.shape}")

    # Free GPU memory
    if device == "cuda":
        del model
        torch.cuda.empty_cache()
        print("\n✓ Freed GPU memory")

    # ========
    # Clean Up
    # ========

    print("\n" + "=" * len("SIMULATION COMPLETE!"))
    print("SIMULATION COMPLETE!")
    print("=" * len("SIMULATION COMPLETE!"))
    print(f"✓ Spike data saved to {results_dir / 'spike_data.zarr'}")
    print(
        f"✓ Data can be loaded with: zarr.open('{results_dir / 'spike_data.zarr'}', mode='r')"
    )
    print("=" * len("SIMULATION COMPLETE!"))

    # =====================================
    # Generate Dashboards for Visualization
    # =====================================

    print("\n" + "=" * len("GENERATING DASHBOARDS"))
    print("GENERATING DASHBOARDS")
    print("=" * len("GENERATING DASHBOARDS"))

    # Concatenate visualization traces
    print("Preparing data for dashboards...")
    viz_voltages = torch.cat(viz_voltages, dim=1)
    viz_currents = torch.cat(viz_currents, dim=1)
    viz_currents_FF = torch.cat(viz_currents_FF, dim=1)
    viz_currents_leak = torch.cat(viz_currents_leak, dim=1)
    viz_conductances = torch.cat(viz_conductances, dim=1)
    viz_conductances_FF = torch.cat(viz_conductances_FF, dim=1)
    viz_output_spikes = torch.cat(viz_output_spikes, dim=1)
    viz_input_spikes = torch.cat(viz_input_spikes, dim=1)
    viz_input_rates = torch.cat(viz_input_rates, dim=1)

    # Concatenate signal repeat data (pattern 0 from batch 1)
    viz_output_spikes_signal_repeat = torch.cat(viz_output_spikes_signal_repeat, dim=1)

    # Move to CPU and convert to numpy
    if device == "cuda":
        viz_voltages = viz_voltages.cpu()
        viz_currents = viz_currents.cpu()
        viz_currents_FF = viz_currents_FF.cpu()
        viz_currents_leak = viz_currents_leak.cpu()
        viz_conductances = viz_conductances.cpu()
        viz_conductances_FF = viz_conductances_FF.cpu()
        viz_output_spikes = viz_output_spikes.cpu()
        viz_input_spikes = viz_input_spikes.cpu()
        viz_input_rates = viz_input_rates.cpu()
        viz_output_spikes_signal_repeat = viz_output_spikes_signal_repeat.cpu()

    viz_voltages = viz_voltages.numpy().astype(np.float32)
    viz_currents = viz_currents.numpy().astype(np.float32)
    viz_currents_FF = viz_currents_FF.numpy().astype(np.float32)
    viz_currents_leak = viz_currents_leak.numpy().astype(np.float32)
    viz_conductances = viz_conductances.numpy().astype(np.float32)
    viz_conductances_FF = viz_conductances_FF.numpy().astype(np.float32)
    viz_output_spikes = viz_output_spikes.numpy().astype(np.int32)
    viz_input_spikes = viz_input_spikes.numpy().astype(np.int32)
    viz_input_rates = viz_input_rates.numpy().astype(np.float32)

    # Convert signal repeat to numpy
    viz_output_spikes_signal_repeat = viz_output_spikes_signal_repeat.numpy().astype(
        np.int32
    )

    print(
        f"✓ Prepared {viz_voltages.shape[1]} time steps for visualization (last ~{viz_duration_s:.1f}s)"
    )

    # Calculate mean membrane potential by cell type from voltage traces
    recurrent_V_mem_by_type = {}
    for i, cell_type_name in enumerate(recurrent.cell_types.names):
        cell_mask = cell_type_indices == i
        if cell_mask.sum() > 0:
            # Average over batch, time, and neurons of this type
            recurrent_V_mem_by_type[cell_type_name] = float(
                viz_voltages[:, :, cell_mask].mean()
            )

    # Generate connectivity dashboard
    print("Generating connectivity dashboard...")
    connectivity_fig = create_connectivity_dashboard(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_source_indices,
        cell_type_names=recurrent.cell_types.names,
        input_cell_type_names=feedforward.cell_types.names,
        connectome_mask=connectome_mask,
        feedforward_mask=feedforward_mask,
    )

    # Generate activity dashboard
    print("Generating activity dashboard...")
    activity_fig = create_activity_dashboard(
        output_spikes=viz_output_spikes,
        input_spikes=viz_input_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=recurrent.cell_types.names,
        dt=simulation.dt,
        voltages=viz_voltages,
        neuron_types=cell_type_indices,
        neuron_params=recurrent.get_neuron_params_for_plotting(),
        recurrent_currents=viz_currents,
        feedforward_currents=viz_currents_FF,
        leak_currents=viz_currents_leak,
        recurrent_conductances=viz_conductances,
        feedforward_conductances=viz_conductances_FF,
        input_cell_type_names=feedforward.cell_types.names,
        recurrent_synapse_names=recurrent.get_synapse_names(),
        feedforward_synapse_names=feedforward.get_synapse_names(),
        window_size=50.0,
        n_neurons_plot=20,
        fraction=1.0,
        random_seed=42,
        assembly_ids=assembly_ids,
    )

    # Generate assembly activity dashboard
    print("Generating assembly activity dashboard...")
    assembly_fig = create_assembly_activity_dashboard(
        output_spikes=viz_output_spikes,
        input_rates=viz_input_rates,
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        dt=simulation.dt,
        excitatory_idx=0,
    )

    # Save dashboards
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    connectivity_fig.savefig(
        figures_dir / "connectivity_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close(connectivity_fig)

    activity_fig.savefig(
        figures_dir / "activity_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close(activity_fig)

    assembly_fig.savefig(
        figures_dir / "assembly_activity_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close(assembly_fig)

    print(f"✓ Saved dashboard plots to {figures_dir}")
    print("=" * len("GENERATING DASHBOARDS"))

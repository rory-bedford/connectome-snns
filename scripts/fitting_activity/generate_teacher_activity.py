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
    HomogeneousPoissonSpikeDataset,
    collate_pattern_batches,
)
from src.network_inputs.odourants import (
    generate_odour_firing_rates,
    generate_baseline_firing_rates,
)
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from torch.utils.data import DataLoader
from parameter_loaders import TeacherActivityParams
from tqdm import tqdm
from visualization.dashboards import (
    create_connectivity_dashboard,
    create_activity_dashboard,
)
from visualization.firing_statistics import (
    plot_assembly_population_activity,
    plot_cross_correlation_scatter,
    plot_cross_correlation_histogram,
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

    # Generate baseline (control) firing rate pattern
    input_firing_rates_baseline = generate_baseline_firing_rates(
        n_input_neurons=feedforward_weights.shape[0],
        input_source_indices=input_source_indices,
        cell_type_names=feedforward.cell_types.names,
        odour_configs=params.get_odour_configs_dict(),
    )

    # Concatenate odour patterns with baseline pattern
    # This appends the control as the last pattern
    input_firing_rates = np.concatenate(
        [input_firing_rates_odour, input_firing_rates_baseline], axis=0
    )

    print(f"✓ Generated {num_odours} odour patterns + 1 baseline (control) pattern")
    print(f"  Total patterns: {input_firing_rates.shape[0]}")

    n_patterns = num_odours + 1  # Include control pattern
    batch_size = simulation.batch_size

    # Create Poisson dataset for multiple patterns
    # Dataset cycles through patterns indefinitely
    spike_dataset = HomogeneousPoissonSpikeDataset(
        firing_rates=input_firing_rates,
        chunk_size=simulation.chunk_size,
        dt=simulation.dt,
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

    # Save input firing rates immediately
    root.create_dataset(
        "input_firing_rates",
        shape=input_firing_rates_odour.shape,
        dtype=input_firing_rates_odour.dtype,
        data=input_firing_rates_odour,
    )

    root.create_dataset(
        "input_firing_rates_control",
        shape=input_firing_rates_baseline.shape,
        dtype=input_firing_rates_baseline.dtype,
        data=input_firing_rates_baseline,
    )
    print("✓ Saved input firing rates to zarr")
    print(f"  - input_firing_rates (odour): {input_firing_rates_odour.shape}")
    print(f"  - input_firing_rates_control: {input_firing_rates_baseline.shape}")

    # Create separate datasets for odour patterns and control pattern
    # Stream directly to final arrays during simulation - no temp storage!
    # Odour patterns: indices 0 to num_odours-1
    output_spikes_odour = root.create_dataset(
        "output_spikes",
        shape=(batch_size, num_odours, total_steps, n_neurons),
        chunks=(batch_size, num_odours, simulation.chunk_size, n_neurons),
        dtype=np.bool_,
    )

    output_spikes_control = root.create_dataset(
        "output_spikes_control",
        shape=(batch_size, 1, total_steps, n_neurons),
        chunks=(batch_size, 1, simulation.chunk_size, n_neurons),
        dtype=np.bool_,
    )

    input_spikes_odour = root.create_dataset(
        "input_spikes",
        shape=(batch_size, num_odours, total_steps, n_input_neurons),
        chunks=(batch_size, num_odours, simulation.chunk_size, n_input_neurons),
        dtype=np.bool_,
    )

    input_spikes_control = root.create_dataset(
        "input_spikes_control",
        shape=(batch_size, 1, total_steps, n_input_neurons),
        chunks=(batch_size, 1, simulation.chunk_size, n_input_neurons),
        dtype=np.bool_,
    )

    chunk_size_mb = (batch_size * num_odours * simulation.chunk_size * n_neurons) / (
        1024**2
    )
    print(
        f"Zarr chunk shape (odour): ({batch_size}, {num_odours}, {simulation.chunk_size}, {n_neurons})"
    )
    print(f"  Chunk size: ~{chunk_size_mb:.1f} MB uncompressed")

    # ==============================
    # Run Chunked Network Simulation
    # ==============================

    print("\n" + "=" * len("STARTING CHUNKED NETWORK SIMULATION"))
    print("STARTING CHUNKED NETWORK SIMULATION")
    print("=" * len("STARTING CHUNKED NETWORK SIMULATION"))

    print(f"Running simulation with {n_patterns} patterns × {batch_size} repeats")
    print(f"  ({num_odours} odour patterns + 1 baseline control)")
    print(f"Processing {simulation.num_chunks} chunks...")
    print(f"Total simulation duration: {simulation.total_duration_s:.2f} s")
    print(
        f"DataLoader returns (batch_size={batch_size}, n_patterns={n_patterns}, chunk_size, neurons) per chunk"
    )

    # Initialize state variables: (batch_size, n_patterns, ...)
    initial_v = None
    initial_g = None
    initial_g_FF = None

    # Variables to store traces from last ~10 seconds for visualization
    viz_duration_s = 10.0  # Save last 10 seconds
    viz_chunks = int(np.ceil(viz_duration_s / simulation.chunk_duration_s))
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

    # Also store control pattern spikes for assembly comparison
    viz_output_spikes_control = []
    viz_input_spikes_control = []

    # Store pattern 0 from batch 1 (second repeat) for cross-correlation
    viz_output_spikes_signal_repeat = []

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

            # Store traces from last few chunks for visualization
            if chunk_idx >= viz_start_chunk:
                # Extract first batch, pattern 0 (first odour) for visualization
                viz_voltages.append(output_voltages_chunk_flat[0:1, :, :].clone())
                viz_currents.append(output_currents_chunk_flat[0:1, :, :, :].clone())
                viz_currents_FF.append(
                    output_currents_FF_chunk_flat[0:1, :, :, :].clone()
                )
                viz_currents_leak.append(
                    output_currents_leak_chunk_flat[0:1, :, :].clone()
                )
                viz_conductances.append(
                    output_conductances_chunk_flat[0:1, :, :, :].clone()
                )
                viz_conductances_FF.append(
                    output_conductances_FF_chunk_flat[0:1, :, :, :].clone()
                )
                viz_output_spikes.append(output_spikes_chunk_flat[0:1, :, :].clone())
                viz_input_spikes.append(input_spikes_flat[0:1, :, :].clone())

                # Also store control pattern (last) for assembly comparison
                viz_output_spikes_control.append(
                    output_spikes_chunk_flat[n_patterns - 1 : n_patterns, :, :].clone()
                )
                viz_input_spikes_control.append(
                    input_spikes_flat[n_patterns - 1 : n_patterns, :, :].clone()
                )

                # Store pattern 0 from batch 1 (second repeat)
                viz_output_spikes_signal_repeat.append(
                    output_spikes_chunk_flat[n_patterns : n_patterns + 1, :, :].clone()
                )

            # Store final states for next chunk: (batch_size*n_patterns, neurons) or (batch_size*n_patterns, neurons, syn_types)
            initial_v = output_voltages_chunk_flat[:, -1, :].clone()
            initial_g = output_conductances_chunk_flat[:, -1, :, :].clone()
            initial_g_FF = output_conductances_FF_chunk_flat[:, -1, :, :].clone()

            # Write directly to separate zarr arrays - stream to disk, no RAM accumulation!
            start_idx = chunk_idx * simulation.chunk_size
            end_idx = start_idx + n_steps_out

            if device == "cuda":
                # Split and write odour patterns (all except last)
                output_spikes_odour[:, :, start_idx:end_idx, :] = (
                    output_spikes_chunk[:, :-1, :, :].bool().cpu().numpy()
                )
                input_spikes_odour[:, :, start_idx:end_idx, :] = (
                    input_spikes_chunk[:, :-1, :, :].cpu().numpy()
                )
                # Write control pattern (last pattern only)
                output_spikes_control[:, :, start_idx:end_idx, :] = (
                    output_spikes_chunk[:, -1:, :, :].bool().cpu().numpy()
                )
                input_spikes_control[:, :, start_idx:end_idx, :] = (
                    input_spikes_chunk[:, -1:, :, :].cpu().numpy()
                )
            else:
                # Split and write odour patterns (all except last)
                output_spikes_odour[:, :, start_idx:end_idx, :] = (
                    output_spikes_chunk[:, :-1, :, :].bool().numpy()
                )
                input_spikes_odour[:, :, start_idx:end_idx, :] = input_spikes_chunk[
                    :, :-1, :, :
                ].numpy()
                # Write control pattern (last pattern only)
                output_spikes_control[:, :, start_idx:end_idx, :] = (
                    output_spikes_chunk[:, -1:, :, :].bool().numpy()
                )
                input_spikes_control[:, :, start_idx:end_idx, :] = input_spikes_chunk[
                    :, -1:, :, :
                ].numpy()

    print("\n✓ Network simulation completed!")
    print(f"\n✓ All data saved to {results_dir / 'spike_data.zarr'}")
    print(f"  - output_spikes (odour): {output_spikes_odour.shape}")
    print(f"  - output_spikes_control: {output_spikes_control.shape}")
    print(f"  - input_spikes (odour): {input_spikes_odour.shape}")
    print(f"  - input_spikes_control: {input_spikes_control.shape}")

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

    # Concatenate control pattern data
    viz_output_spikes_control = torch.cat(viz_output_spikes_control, dim=1)
    viz_input_spikes_control = torch.cat(viz_input_spikes_control, dim=1)

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
        viz_output_spikes_control = viz_output_spikes_control.cpu()
        viz_input_spikes_control = viz_input_spikes_control.cpu()
        viz_output_spikes_signal_repeat = viz_output_spikes_signal_repeat.cpu()

    viz_voltages = viz_voltages.numpy().astype(np.float32)
    viz_currents = viz_currents.numpy().astype(np.float32)
    viz_currents_FF = viz_currents_FF.numpy().astype(np.float32)
    viz_currents_leak = viz_currents_leak.numpy().astype(np.float32)
    viz_conductances = viz_conductances.numpy().astype(np.float32)
    viz_conductances_FF = viz_conductances_FF.numpy().astype(np.float32)
    viz_output_spikes = viz_output_spikes.numpy().astype(np.int32)
    viz_input_spikes = viz_input_spikes.numpy().astype(np.int32)

    # Convert control pattern to numpy
    viz_output_spikes_control = viz_output_spikes_control.numpy().astype(np.int32)
    viz_input_spikes_control = viz_input_spikes_control.numpy().astype(np.int32)

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

    # Generate assembly population activity plot (side-by-side comparison)
    print("Generating assembly population activity plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot pattern 0 (odour)
    plot_assembly_population_activity(
        spike_trains=viz_output_spikes,
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        window_size=200.0,  # Gaussian kernel std in ms
        dt=simulation.dt,
        excitatory_idx=0,  # Assuming excitatory neurons are type 0
        title="Excitatory Assembly Activity (Pattern 0)",
        ax=ax1,
    )

    # Plot control pattern
    plot_assembly_population_activity(
        spike_trains=viz_output_spikes_control,
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        window_size=200.0,  # Gaussian kernel std in ms
        dt=simulation.dt,
        excitatory_idx=0,  # Assuming excitatory neurons are type 0
        title="Excitatory Assembly Activity (Control Pattern)",
        ax=ax2,
    )

    plt.tight_layout()
    fig.savefig(
        figures_dir / "assembly_population_activity.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Generate cross-correlation scatter plot subplot
    print("Generating cross-correlation scatter plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Pattern 0 (batch 0) vs Pattern 0 (batch 1) - signal repeatability
    plot_cross_correlation_scatter(
        spike_trains_trial1=viz_output_spikes,
        spike_trains_trial2=viz_output_spikes_signal_repeat,
        window_size=10.0,  # 10 second windows
        dt=simulation.dt,
        cell_indices=None,  # All cells in network
        ax=ax1,
        title="Pattern 0 vs Pattern 0 (Repeat)",
        x_label="Pattern 0 (Trial 1) Rate (Hz)",
        y_label="Pattern 0 (Trial 2) Rate (Hz)",
    )

    # Right: Pattern 0 vs Control - signal vs noise
    plot_cross_correlation_scatter(
        spike_trains_trial1=viz_output_spikes,
        spike_trains_trial2=viz_output_spikes_control,
        window_size=10.0,  # 10 second windows
        dt=simulation.dt,
        cell_indices=None,  # All cells in network
        ax=ax2,
        title="Pattern 0 vs Control",
        x_label="Pattern 0 Rate (Hz)",
        y_label="Control Rate (Hz)",
    )

    # Share axis limits across subplots
    x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    fig.savefig(
        figures_dir / "cross_correlation_scatter.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Generate cross-correlation 2D histogram subplot (assembly-pooled)
    print("Generating cross-correlation 2D histogram (assembly-pooled)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Pattern 0 (batch 0) vs Pattern 0 (batch 1) - signal repeatability
    plot_cross_correlation_histogram(
        spike_trains_trial1=viz_output_spikes,
        spike_trains_trial2=viz_output_spikes_signal_repeat,
        window_size=0.05,  # 50 ms windows
        dt=simulation.dt,
        bin_size=0.1,  # 0.1 Hz bins
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        excitatory_idx=0,  # Assuming excitatory neurons are type 0
        ax=ax1,
        title="Pattern 0 vs Pattern 0 (Repeat)",
        x_label="Pattern 0 (Trial 1) Rate (Hz)",
        y_label="Pattern 0 (Trial 2) Rate (Hz)",
    )

    # Right: Pattern 0 vs Control - signal vs noise
    plot_cross_correlation_histogram(
        spike_trains_trial1=viz_output_spikes,
        spike_trains_trial2=viz_output_spikes_control,
        window_size=0.05,  # 50 ms windows
        dt=simulation.dt,
        bin_size=0.1,  # 0.1 Hz bins
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        excitatory_idx=0,  # Assuming excitatory neurons are type 0
        ax=ax2,
        title="Pattern 0 vs Control",
        x_label="Pattern 0 Rate (Hz)",
        y_label="Control Rate (Hz)",
    )

    # Share axis limits across subplots
    x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    fig.savefig(
        figures_dir / "cross_correlation_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"✓ Saved dashboard plots to {figures_dir}")
    print("=" * len("GENERATING DASHBOARDS"))

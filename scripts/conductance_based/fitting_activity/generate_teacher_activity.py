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
from src.dataloaders.unsupervised import (
    InhomogeneousPoissonSpikeDataLoader,
)
from src.dataloaders.rate_processes import OrnsteinUhlenbeckRateProcess
from src.dataloaders.odourants import (
    generate_odour_firing_rates,
)
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from snn_runners import SNNInference
from configs import SimulationConfig
from configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig
from configs.odours import OdourInputConfig
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

    # Load network parameters from TOML file
    with open(params_file, "r") as f:
        data = toml.load(f)

    # Load configuration sections
    simulation = SimulationConfig(**data["simulation"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])

    # Parse odour configs
    odours_data = data["odours"]
    tau = odours_data.pop("tau")
    temperature = odours_data.pop("temperature")
    sigma = odours_data.pop("sigma")
    odours = {name: OdourInputConfig(**config) for name, config in odours_data.items()}

    # Extract parameters into plain Python variables
    dt = simulation.dt
    chunk_size = simulation.chunk_size
    num_chunks = simulation.num_chunks
    batch_size = simulation.batch_size
    seed = simulation.seed
    plot_size = simulation.plot_size

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")
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
        odour_configs={name: cfg.to_dict() for name, cfg in odours.items()},
    )

    print(f"✓ Generated {num_odours} odour patterns")
    print(f"  Total patterns: {input_firing_rates_odour.shape[0]}")

    batch_size = batch_size

    # Create Ornstein-Uhlenbeck rate process in pattern space
    # This modulates the odour patterns dynamically over time
    rate_process = OrnsteinUhlenbeckRateProcess(
        patterns=input_firing_rates_odour,  # Shape: (n_patterns, n_input_neurons)
        chunk_size=int(chunk_size),
        dt=dt,
        tau=tau,  # Time constant for mean reversion (ms)
        temperature=temperature,  # Softmax temperature for pattern mixing
        sigma=sigma,  # Noise amplitude
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
        dt=dt,
        weights=weights,
        cell_type_indices=cell_type_indices,
        cell_params=recurrent.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        surrgrad_scale=1.0,  # Not used for inference, but required parameter
        batch_size=batch_size,
        weights_FF=feedforward_weights,
        cell_type_indices_FF=input_source_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        optimisable=None,  # No optimization for inference
        track_variables=False,  # Will enable only for visualization chunks
        track_batch_idx=0,  # When tracking enabled, only track first batch element
        use_tqdm=False,  # Disable model's internal progress bar
    )

    # Move model to device for GPU acceleration
    model.to(device)
    print(f"Model moved to device: {device}")

    # Jit compile the model for faster execution
    model.compile_step()
    print("✓ Model JIT compiled for faster execution")

    # ======================================
    # Save Static Data and Run Inference
    # ======================================

    # Create results directory if it doesn't exist
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = results_dir / "spike_data.zarr"

    # Save static odourant patterns (base patterns before OU modulation)
    # This is done separately since it doesn't come from the inference
    root = zarr.open_group(zarr_path, mode="w")
    root.create_dataset(
        "odourant_patterns",
        shape=input_firing_rates_odour.shape,
        dtype=input_firing_rates_odour.dtype,
        data=input_firing_rates_odour,
    )
    print("✓ Saved odourant patterns to zarr")
    print(f"  - odourant_patterns: {input_firing_rates_odour.shape}")

    # ==============================
    # Run Inference with SNNInference
    # ==============================

    print("\n" + "=" * len("STARTING NETWORK INFERENCE"))
    print("STARTING NETWORK INFERENCE")
    print("=" * len("STARTING NETWORK INFERENCE"))

    total_duration_s = num_chunks * chunk_size * dt / 1000.0
    print(f"Running inference with {batch_size} independent OU-modulated trajectories")
    print(f"Processing {num_chunks} chunks...")
    print(f"Total simulation duration: {total_duration_s:.2f} s")

    # Run inference and save to zarr
    # Note: We're NOT saving tracked variables yet (too large), we'll do a separate run for visualization
    inference_runner = SNNInference(
        model=model,
        dataloader=spike_dataloader,
        device=device,
        output_mode="zarr",
        zarr_path=zarr_path,
        save_tracked_variables=False,  # Don't save tracked vars (too large for full simulation)
        max_chunks=num_chunks,
        progress_bar=True,
    )

    _ = inference_runner.run()  # Run inference and save to zarr

    print("\n✓ Network inference completed!")
    print(f"✓ Data saved to {zarr_path}")

    # Reopen zarr to inspect what was saved
    root = zarr.open_group(zarr_path, mode="r")
    for key in root.keys():
        print(f"  - {key}: {root[key].shape}")

    # =========================================================
    # Run Small Inference for Visualization (with tracking)
    # =========================================================

    print("\n" + "=" * len("GENERATING VISUALIZATION DATA"))
    print("GENERATING VISUALIZATION DATA")
    print("=" * len("GENERATING VISUALIZATION DATA"))
    print(f"Running {plot_size} chunks with variable tracking for visualization...")

    # Reset model state for clean inference
    model.reset_state(batch_size=batch_size)
    model.track_variables = True  # Enable tracking for visualization

    # Create a new dataloader for visualization chunks
    viz_rate_process = OrnsteinUhlenbeckRateProcess(
        patterns=input_firing_rates_odour,
        chunk_size=int(chunk_size),
        dt=dt,
        tau=tau,
        temperature=temperature,
        sigma=sigma,
        a_init=None,
        return_rates=True,
    )

    viz_dataloader = InhomogeneousPoissonSpikeDataLoader(
        rate_process=viz_rate_process,
        batch_size=batch_size,
        device=device,
        return_rates=True,
    )

    # Run inference in memory mode to get tracked variables
    viz_inference_runner = SNNInference(
        model=model,
        dataloader=viz_dataloader,
        device=device,
        output_mode="memory",
        save_tracked_variables=True,  # Save tracked variables for visualization
        max_chunks=plot_size,
        progress_bar=False,
    )

    viz_result = viz_inference_runner.run()

    # Extract visualization data (first batch only)
    viz_output_spikes = viz_result["output_spikes"][0:1, ...]  # (1, time, neurons)
    viz_input_spikes = viz_result["input_spikes"][0:1, ...]
    viz_ou_weights = viz_result["weights"][0:1, ...]
    viz_voltages = viz_result["voltages"][0:1, ...]
    viz_currents = viz_result["currents_recurrent"][0:1, ...]
    viz_currents_FF = viz_result["currents_feedforward"][0:1, ...]
    viz_currents_leak = viz_result["currents_leak"][0:1, ...]
    viz_conductances = viz_result["conductances_recurrent"][0:1, ...]
    viz_conductances_FF = viz_result["conductances_feedforward"][0:1, ...]

    print(f"✓ Visualization data generated ({plot_size} chunks)")

    # Free GPU memory
    if device == "cuda":
        del model
        torch.cuda.empty_cache()
        print("\n✓ Freed GPU memory")

    # ========
    # Clean Up
    # ========

    print("\n" + "=" * len("INFERENCE COMPLETE!"))
    print("INFERENCE COMPLETE!")
    print("=" * len("INFERENCE COMPLETE!"))
    print(f"✓ Spike data saved to {zarr_path}")
    print(f"✓ Data can be loaded with: zarr.open('{zarr_path}', mode='r')")
    print("=" * len("INFERENCE COMPLETE!"))

    # =====================================
    # Generate Dashboards for Visualization
    # =====================================

    print("\n" + "=" * len("GENERATING DASHBOARDS"))
    print("GENERATING DASHBOARDS")
    print("=" * len("GENERATING DASHBOARDS"))

    # Data is already in numpy format from inference runner (batch=1)
    # Convert to appropriate dtypes for visualization
    viz_voltages = viz_voltages.astype(np.float32)
    viz_currents = viz_currents.astype(np.float32)
    viz_currents_FF = viz_currents_FF.astype(np.float32)
    viz_currents_leak = viz_currents_leak.astype(np.float32)
    viz_conductances = viz_conductances.astype(np.float32)
    viz_conductances_FF = viz_conductances_FF.astype(np.float32)
    viz_output_spikes = viz_output_spikes.astype(np.int32)
    viz_input_spikes = viz_input_spikes.astype(np.int32)
    viz_ou_weights = viz_ou_weights.astype(np.float32)

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
        dt=dt,
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
        ou_process_weights=viz_ou_weights,
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        dt=dt,
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

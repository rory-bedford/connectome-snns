"""
Generating spike trains from a loaded connectome

This script loads a predefined synthetic connectome from disk and generates
spiketrains using conductance-based LIF neurons with Poisson inputs.

Useful for running multiple simulations with the same connectivity structure
but different input patterns, seeds, or simulation parameters.

Overview:
1. Load pre-generated connectome structure (weights, connectivity, cell types)
   from disk.
2. Generate Poisson input spike trains with specified firing rates.
3. Run network simulation with loaded connectivity.
4. Save outputs (spike trains, dynamics) for analysis.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from configs import SimulationConfig
from configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig
import toml
from dataloaders.unsupervised import HomogeneousPoissonSpikeDataLoader
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from snn_runners import SNNInference
from analysis import compute_network_statistics
from visualization.dashboards import (
    create_connectivity_dashboard,
    create_activity_dashboard,
)


def main(input_dir, output_dir, params_file):
    """Main execution function for loaded connectome simulation.

    Args:
        input_dir (Path): Directory containing input data files (network_structure.npz)
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
    weights = network_structure["weights"]
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

    # ==========================================
    # Create Feedforward Input Spike Generator
    # ==========================================

    print(f"Setting up {feedforward_weights.shape[0]} feedforward inputs...")

    # Calculate simulation parameters
    print(
        f"Simulation: {simulation.duration:.1f} ms in {simulation.num_chunks} chunks of {simulation.chunk_size:.1f} ms each"
    )
    print(
        f"Timestep: {simulation.dt:.2f} ms ({int(simulation.chunk_size / simulation.dt)} steps per chunk)"
    )

    # Create firing rates array for input neurons
    input_firing_rates = np.zeros(feedforward_weights.shape[0])
    for ct_idx, ct_name in enumerate(feedforward.cell_types.names):
        mask = input_source_indices == ct_idx
        input_firing_rates[mask] = feedforward.activity[ct_name].firing_rate

    # Create Poisson spike generator dataloader
    spike_dataloader = HomogeneousPoissonSpikeDataLoader(
        firing_rates=input_firing_rates,
        chunk_size=simulation.chunk_size,
        dt=simulation.dt,
        batch_size=1,  # Inference mode - single sample at a time
        device=device,
    )

    # ======================
    # Initialize LIF Network
    # ======================

    print(
        f"Initializing conductance-based LIF network with {len(recurrent.get_cell_params())} cell types and {len(recurrent.get_synapse_params())} synapse types..."
    )

    # Initialize conductance-based LIF network model
    model = ConductanceLIFNetwork(
        dt=simulation.dt,
        weights=weights,
        cell_type_indices=cell_type_indices,
        cell_params=recurrent.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        surrgrad_scale=1.0,  # Not used for inference, but required parameter
        batch_size=1,
        weights_FF=feedforward_weights,
        cell_type_indices_FF=input_source_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        optimisable=None,  # No optimization for inference
        track_variables=True,  # Enable tracking for visualization
        use_tqdm=False,  # Disable model's internal progress bar (SNNInference has its own)
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

    print(f"Running simulation in {simulation.num_chunks} chunks...")

    # Run inference using SNNInference
    inference_runner = SNNInference(
        model=model,
        dataloader=spike_dataloader,
        device=device,
        output_mode="memory",  # Load all results into memory for visualization
        save_tracked_variables=True,  # Save voltages, currents, conductances
        max_chunks=simulation.num_chunks,
        progress_bar=True,
    )

    result = inference_runner.run()

    # Extract results (already in numpy format, batch dimension is first)
    output_spikes = result["output_spikes"]  # (batch=1, time, neurons)
    output_voltages = result["voltages"]
    output_currents = result["currents_recurrent"]
    output_currents_FF = result["currents_feedforward"]
    output_currents_leak = result["currents_leak"]
    output_conductances = result["conductances_recurrent"]
    output_conductances_FF = result["conductances_feedforward"]
    input_spikes = result["input_spikes"]

    print("\n✓ Network simulation completed!")

    # ============================================
    # Save Output Data for Further Analysis
    # ============================================

    print("Saving simulation data...")

    # Create results directory if it doesn't exist
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save spike trains (the main output for analysis)
    np.savez_compressed(
        results_dir / "spike_data.npz",
        output_spikes=output_spikes.astype(np.bool_),
        input_spikes=input_spikes.astype(np.bool_),
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_source_indices,
    )

    # Save connectivity/weights (needed for network analysis)
    # Note: This copies the loaded network structure to the output directory
    np.savez_compressed(
        results_dir / "network_structure.npz",
        weights=weights.astype(np.float32),
        feedforward_weights=feedforward_weights.astype(np.float32),
        cell_type_indices=cell_type_indices,
        feedforward_cell_type_indices=input_source_indices,
        assembly_ids=assembly_ids,
    )

    print(f"✓ Saved data to {results_dir} (spikes + connectivity)")

    # =============================================
    # Generate All Plots and Visualizations
    # =============================================

    print("Generating dashboards...")

    # Calculate mean membrane potential by cell type from voltage traces
    recurrent_V_mem_by_type = {}
    for i, cell_type_name in enumerate(recurrent.cell_types.names):
        cell_mask = cell_type_indices == i
        if cell_mask.sum() > 0:
            # Average over batch, time, and neurons of this type
            recurrent_V_mem_by_type[cell_type_name] = float(
                output_voltages[:, :, cell_mask].mean()
            )

    # Derive connectivity masks from weight matrices
    connectome_mask = (weights != 0).astype(np.bool_)
    feedforward_mask = (feedforward_weights != 0).astype(np.bool_)

    # Generate connectivity dashboard
    connectivity_fig = create_connectivity_dashboard(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_source_indices,
        cell_type_names=recurrent.cell_types.names,
        input_cell_type_names=feedforward.cell_types.names,
        connectome_mask=connectome_mask,
        feedforward_mask=feedforward_mask,
        num_assemblies=recurrent.topology.num_assemblies,
    )

    # Generate activity dashboard
    activity_fig = create_activity_dashboard(
        output_spikes=output_spikes,
        input_spikes=input_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=recurrent.cell_types.names,
        dt=simulation.dt,
        voltages=output_voltages,
        neuron_types=cell_type_indices,
        neuron_params=recurrent.get_neuron_params_for_plotting(),
        recurrent_currents=output_currents,
        feedforward_currents=output_currents_FF,
        leak_currents=output_currents_leak,
        recurrent_conductances=output_conductances,
        feedforward_conductances=output_conductances_FF,
        input_cell_type_names=feedforward.cell_types.names,
        recurrent_synapse_names=recurrent.get_synapse_names(),
        feedforward_synapse_names=feedforward.get_synapse_names(),
        window_size=50.0,
        n_neurons_plot=20,
        fraction=1.0,
        random_seed=42,
    )

    # Save the two dashboards
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

    print(f"✓ Saved dashboard plots to {figures_dir}")

    # ============================================
    # Compute and Save Combined Statistics to CSV
    # ============================================

    print("Computing network statistics...")
    stats_df = compute_network_statistics(
        output_spikes=output_spikes,
        output_voltages=output_voltages,
        cell_type_indices=cell_type_indices,
        cell_type_names=recurrent.cell_types.names,
        duration=simulation.duration,
        dt=simulation.dt,
    )

    stats_csv_path = output_dir / "network_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"✓ Saved network statistics to {stats_csv_path}")

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

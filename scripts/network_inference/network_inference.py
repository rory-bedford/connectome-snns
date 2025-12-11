"""
Generating spike trains from a synthetic connectome

This script generates spiketrains from a recurrent E-I network with conductance-based
LIF neurons, and assembly connectivity inspired by zebrafish Dp.

Overview:
1. First we generate a biologically plausible recurrent weight matrix with a
   Dp-inspired assembly structure.
2. Next we generate excitatory mitral cell inputs from the OB with Poisson
   statistics and sparse projections.
3. Then we initialise our network with parameters adapted from
   Meissner-Bernard et al. (2025) https://doi.org/10.1016/j.celrep.2025.115330
4. Finally we run our network and examine the output dynamics.
"""

import numpy as np
from synthetic_connectome import (
    topology_generators,
    weight_assigners,
    cell_types,
)
from odourants.dataloaders import PoissonSpikeDataset
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
import torch
import matplotlib.pyplot as plt
from parameter_loaders import ConductanceBasedParams
import toml
from analysis import compute_network_statistics
from visualization.dashboards import (
    create_connectivity_dashboard,
    create_activity_dashboard,
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
    params = ConductanceBasedParams(**data)

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

    # ===================================
    # Create Recurrent Connectivity Graph
    # ===================================

    print(
        f"Generating {recurrent.topology.num_neurons} neuron recurrent network with {recurrent.topology.num_assemblies} assemblies..."
    )

    # Assign cell types to recurrent layer
    cell_type_indices = cell_types.assign_cell_types(
        num_neurons=recurrent.topology.num_neurons,
        cell_type_proportions=recurrent.cell_types.proportion,
    )

    # Generate assembly-based connectivity graph
    connectivity_graph, assembly_ids = topology_generators.assembly_generator(
        source_cell_types=cell_type_indices,
        target_cell_types=cell_type_indices,  # Same for recurrent connections
        num_assemblies=recurrent.topology.num_assemblies,
        conn_within=recurrent.topology.conn_within,
        conn_between=recurrent.topology.conn_between,
        method="configuration",  # Ensures exact in-degree/out-degree distributions
    )

    # =======================
    # Assign Synaptic Weights
    # =======================

    # Assign log-normal weights to connectivity graph (no signs for conductance-based)
    weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=connectivity_graph,
        source_cell_indices=cell_type_indices,
        target_cell_indices=cell_type_indices,  # Same for recurrent connections
        cell_type_signs=None,  # No signs for conductance-based model
        w_mu_matrix=recurrent.weights.w_mu,
        w_sigma_matrix=recurrent.weights.w_sigma,
        parameter_space="linear",
    )

    # ==========================================
    # Create Feedforward Connections and Weights
    # ==========================================

    print(f"Generating {feedforward.topology.num_neurons} feedforward inputs...")

    # Calculate simulation parameters
    print(
        f"Simulation: {simulation.duration:.1f} ms in {simulation.num_chunks} chunks of {simulation.chunk_size:.1f} ms each"
    )
    print(
        f"Timestep: {simulation.dt:.2f} ms ({int(simulation.chunk_size / simulation.dt)} steps per chunk)"
    )

    # Assign cell types to input layer
    input_source_indices = cell_types.assign_cell_types(
        num_neurons=feedforward.topology.num_neurons,
        cell_type_proportions=feedforward.cell_types.proportion,
    )

    # Create firing rates array for input neurons
    input_firing_rates = np.zeros(feedforward.topology.num_neurons)
    for ct_idx, ct_name in enumerate(feedforward.cell_types.names):
        mask = input_source_indices == ct_idx
        input_firing_rates[mask] = feedforward.activity[ct_name].firing_rate

    # Create Poisson spike generator dataset
    spike_dataset = PoissonSpikeDataset(
        firing_rates=input_firing_rates,
        chunk_size=simulation.chunk_size,
        dt=simulation.dt,
        device=device,
    )

    # Generate feedforward connectivity graph
    feedforward_connectivity_graph = topology_generators.sparse_graph_generator(
        source_cell_types=input_source_indices,
        target_cell_types=cell_type_indices,  # Connectome layer cell assignments
        conn_matrix=feedforward.topology.conn_inputs,  # N_input_types x N_recurrent_types matrix
        allow_self_loops=True,  # Allow for feedforward connections
        method="configuration",  # Ensures exact in-degree/out-degree distributions
    )

    # Assign log-normal weights to feedforward connectivity (no signs for conductance-based)
    feedforward_weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=feedforward_connectivity_graph,
        source_cell_indices=input_source_indices,
        target_cell_indices=cell_type_indices,
        cell_type_signs=None,  # No signs for conductance-based model
        w_mu_matrix=feedforward.weights.w_mu,
        w_sigma_matrix=feedforward.weights.w_sigma,
        parameter_space="linear",
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
        weights_FF=feedforward_weights,
        cell_type_indices_FF=input_source_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        optimisable=None,  # No optimization for inference
        use_tqdm=True,  # Enable model's internal progress bar for each chunk
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

    # Initialize lists to accumulate results across chunks
    all_output_spikes = []
    all_output_voltages = []
    all_output_currents = []
    all_output_currents_FF = []
    all_output_currents_leak = []
    all_output_conductances = []
    all_output_conductances_FF = []
    all_input_spikes = []

    # Initialize state variables (will be passed between chunks)
    initial_v = None
    initial_g = None
    initial_g_FF = None

    # Run inference in chunks
    with torch.inference_mode():
        for chunk_idx in range(simulation.num_chunks):
            print(f"Processing chunk {chunk_idx + 1}/{simulation.num_chunks}...")

            # Get one chunk of input spikes from dataset (returns tuple)
            input_spikes_chunk, _ = spike_dataset[chunk_idx]
            input_spikes_chunk = input_spikes_chunk.unsqueeze(0)  # Add batch dimension

            # Run one chunk of simulation
            (
                output_spikes,
                output_voltages,
                output_currents,
                output_currents_FF,
                output_currents_leak,
                output_conductances,
                output_conductances_FF,
            ) = model.forward(
                input_spikes=input_spikes_chunk,
                initial_v=initial_v,
                initial_g=initial_g,
                initial_g_FF=initial_g_FF,
            )

            # Store final states for next chunk
            initial_v = output_voltages[:, -1, :].clone()  # Last timestep voltages
            initial_g = output_conductances[
                :, -1, :, :
            ].clone()  # Last timestep conductances
            initial_g_FF = output_conductances_FF[
                :, -1, :, :
            ].clone()  # Last timestep FF conductances

            # Move to CPU and accumulate results
            if device == "cuda":
                all_output_spikes.append(output_spikes.bool().cpu())
                all_output_voltages.append(output_voltages.cpu())
                all_output_currents.append(output_currents.cpu())
                all_output_currents_FF.append(output_currents_FF.cpu())
                all_output_currents_leak.append(output_currents_leak.cpu())
                all_output_conductances.append(output_conductances.cpu())
                all_output_conductances_FF.append(output_conductances_FF.cpu())
                all_input_spikes.append(input_spikes_chunk.cpu())
            else:
                all_output_spikes.append(output_spikes.bool())
                all_output_voltages.append(output_voltages)
                all_output_currents.append(output_currents)
                all_output_currents_FF.append(output_currents_FF)
                all_output_currents_leak.append(output_currents_leak)
                all_output_conductances.append(output_conductances)
                all_output_conductances_FF.append(output_conductances_FF)
                all_input_spikes.append(input_spikes_chunk)

    # Concatenate all chunks along time dimension and convert to numpy
    output_spikes = torch.cat(all_output_spikes, dim=1).numpy()
    output_voltages = torch.cat(all_output_voltages, dim=1).numpy()
    output_currents = torch.cat(all_output_currents, dim=1).numpy()
    output_currents_FF = torch.cat(all_output_currents_FF, dim=1).numpy()
    output_currents_leak = torch.cat(all_output_currents_leak, dim=1).numpy()
    output_conductances = torch.cat(all_output_conductances, dim=1).numpy()
    output_conductances_FF = torch.cat(all_output_conductances_FF, dim=1).numpy()
    input_spikes = torch.cat(all_input_spikes, dim=1).numpy()

    print("✓ Network simulation completed!")

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
    np.savez_compressed(
        results_dir / "network_structure.npz",
        weights=weights.astype(np.float32),
        feedforward_weights=feedforward_weights.astype(np.float32),
        cell_type_indices=cell_type_indices,
        feedforward_cell_type_indices=input_source_indices,
        assembly_ids=assembly_ids,
        recurrent_connectivity=connectivity_graph,
        feedforward_connectivity=feedforward_connectivity_graph,
    )

    print(f"✓ Saved data to {results_dir} (spikes + connectivity)")

    # =============================================
    # Generate All Plots and Visualizations
    # =============================================

    print("Generating dashboards...")

    # Calculate mean membrane potential by cell type from voltage traces
    recurrent_V_mem_by_type = {}
    for i, cell_type_name in enumerate(params.recurrent.cell_types.names):
        cell_mask = cell_type_indices == i
        if cell_mask.sum() > 0:
            # Average over batch, time, and neurons of this type
            recurrent_V_mem_by_type[cell_type_name] = float(
                output_voltages[:, :, cell_mask].mean()
            )

    # Generate connectivity dashboard
    connectivity_fig = create_connectivity_dashboard(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_source_indices,
        cell_type_names=params.recurrent.cell_types.names,
        input_cell_type_names=params.feedforward.cell_types.names,
        num_assemblies=params.recurrent.topology.num_assemblies,
    )

    # Compute total excitatory and inhibitory currents from recurrent and feedforward
    # Generate activity dashboard
    activity_fig = create_activity_dashboard(
        output_spikes=output_spikes,
        input_spikes=input_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=params.recurrent.cell_types.names,
        dt=params.simulation.dt,
        voltages=output_voltages,
        neuron_types=cell_type_indices,
        neuron_params=params.recurrent.get_neuron_params_for_plotting(),
        recurrent_currents=output_currents,
        feedforward_currents=output_currents_FF,
        leak_currents=output_currents_leak,
        recurrent_conductances=output_conductances,
        feedforward_conductances=output_conductances_FF,
        input_cell_type_names=params.feedforward.cell_types.names,
        recurrent_synapse_names=params.recurrent.get_synapse_names(),
        feedforward_synapse_names=params.feedforward.get_synapse_names(),
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
        cell_type_names=params.recurrent.cell_types.names,
        duration=params.simulation.duration,
        dt=params.simulation.dt,
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

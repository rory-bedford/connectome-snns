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
from inputs.dataloaders import PoissonSpikeDataset
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
import torch
import sys
from pathlib import Path
from network_simulators.conductance_based.parameter_loader import (
    ConductanceBasedParams,
)
import toml

# Import local scripts
sys.path.append(str(Path(__file__).resolve().parent))
import conductance_based_Dp_plots


def main(output_dir, params_file):
    """Main execution function for Dp network simulation.

    Args:
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
    connectivity_graph = topology_generators.assembly_generator(
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

    # Calculate number of timesteps
    n_steps = int(simulation.duration / simulation.dt)
    print(
        f"Simulation: {simulation.duration:.1f} ms ({n_steps} timesteps at {simulation.dt:.2f} ms resolution)"
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
        chunk_size=n_steps,
        dt=simulation.dt,
        device=device,
    )

    # Generate input spikes (batch_size=1)
    input_spikes = spike_dataset[0].unsqueeze(0)  # Shape: (1, n_steps, num_neurons)

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
        optimisable="weights",
        use_tqdm=True,  # Enable progress bar for simulation
    )

    # Move model to device for GPU acceleration
    model.to(device)
    print(f"Model moved to device: {device}")

    # Jit compile the model for faster execution
    model._step = torch.jit.script(model._step)
    print("✓ Model JIT compiled for faster execution")

    print("\n" + "=" * len("STARTING NETWORK SIMULATION"))
    print("STARTING NETWORK SIMULATION")
    print("=" * len("STARTING NETWORK SIMULATION"))

    # =================
    # Run Network Simulation
    # =================

    print("Running network simulation...")

    # Run inference
    with torch.inference_mode():
        (
            output_spikes,
            output_voltages,
            output_currents,
            output_currents_FF,
            output_conductances,
            output_conductances_FF,
        ) = model.forward(
            input_spikes=input_spikes,
        )

    # Move tensors to CPU and convert to numpy arrays in one step
    if device == "cuda":
        output_spikes = output_spikes.cpu().numpy()
        output_voltages = output_voltages.cpu().numpy()
        output_currents = output_currents.cpu().numpy()
        output_currents_FF = output_currents_FF.cpu().numpy()
        output_conductances = output_conductances.cpu().numpy()
        output_conductances_FF = output_conductances_FF.cpu().numpy()
        input_spikes = input_spikes.cpu().numpy()
    else:
        output_spikes = output_spikes.numpy()
        output_voltages = output_voltages.numpy()
        output_currents = output_currents.numpy()
        output_currents_FF = output_currents_FF.numpy()
        output_conductances = output_conductances.numpy()
        output_conductances_FF = output_conductances_FF.numpy()
        input_spikes = input_spikes.numpy()

    print("✓ Network simulation completed!")

    # ============================================
    # Save Output Data for Further Analysis
    # ============================================

    print("Saving simulation data...")

    # Create results directory if it doesn't exist
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save output arrays to results/
    np.save(results_dir / "output_spikes.npy", output_spikes)
    np.save(results_dir / "output_voltages.npy", output_voltages)
    np.save(results_dir / "output_currents.npy", output_currents)
    np.save(results_dir / "input_currents.npy", output_currents_FF)
    np.save(results_dir / "output_conductances.npy", output_conductances)
    np.save(
        results_dir / "input_conductances.npy",
        output_conductances_FF,
    )

    # Save input data and network structure to results/
    np.save(results_dir / "input_spikes.npy", input_spikes)
    np.save(results_dir / "cell_type_indices.npy", cell_type_indices)
    np.save(results_dir / "input_cell_type_indices.npy", input_source_indices)
    np.save(results_dir / "connectivity_graph.npy", connectivity_graph)
    np.save(results_dir / "weights.npy", weights)
    np.save(results_dir / "feedforward_weights.npy", feedforward_weights)

    print(f"✓ Saved all arrays to {results_dir}")

    # =============================================
    # Generate All Plots and Visualizations
    # =============================================

    print("Generating plots and analysis...")

    # Call the plotting script to generate all visualizations
    conductance_based_Dp_plots.main(output_dir)

    # ========
    # Clean Up
    # ========

    print("\n" + "=" * len("SIMULATION COMPLETE!"))
    print("SIMULATION COMPLETE!")
    print("=" * len("SIMULATION COMPLETE!"))
    print(f"✓ Results saved to {output_dir / 'results'}")
    print(f"✓ Figures saved to {output_dir / 'figures'}")
    print(f"✓ Analysis saved to {output_dir / 'analysis'}")
    print("=" * len("SIMULATION COMPLETE!"))

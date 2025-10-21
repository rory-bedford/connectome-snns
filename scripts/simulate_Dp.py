"""
Generating spike trains from a synthetic connectome

This script generates spiketrains from a recurrent E-I network with current-based
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
import pandas as pd
import toml
from synthetic_connectome import topology_generators, weight_assigners
from network_simulators.current_lif_network import CurrentLIFNetwork
import torch
import simulate_Dp_plots


def main(output_dir, params_file):
    """Main execution function for Dp network simulation.

    Args:
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
    """
    # ================================================================================================
    # SETUP: Device selection and parameter loading
    # ================================================================================================

    # Select device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all network parameters from TOML file
    with open(params_file, 'r') as f:
        params = toml.load(f)

    # Extract simulation parameters
    delta_t = params['simulation']['delta_t']
    duration = params['simulation']['duration']
    
    # Extract connectome topology parameters
    num_neurons = int(params['connectome']['topology']['num_neurons'])
    num_assemblies = int(params['connectome']['topology']['num_assemblies'])
    conn_within = params['connectome']['topology']['conn_within']  # Matrix of connection probabilities
    conn_between = params['connectome']['topology']['conn_between']  # Matrix of connection probabilities
    
    # Extract connectome cell type information
    cell_type_names = params['connectome']['cell_types']['names']
    cell_type_signs = params['connectome']['cell_types']['signs']  # 1 for excitatory, -1 for inhibitory
    cell_type_proportions = params['connectome']['cell_types']['proportion']
    
    # Extract connectome weight parameters (matrices)
    w_mu = params['connectome']['weights']['w_mu']  # Matrix of log-space means
    w_sigma = params['connectome']['weights']['w_sigma']  # Matrix of log-space std devs
    
    # Extract input layer parameters
    input_num_neurons = int(params['inputs']['topology']['num_neurons'])
    input_cell_type_names = params['inputs']['cell_types']['names']
    input_firing_rates = params['inputs']['activity']['firing_rates']
    input_conn_inputs = params['inputs']['topology']['conn_inputs']  # Matrix of connection probabilities to connectome
    input_w_mu = params['inputs']['weights']['w_mu']  # Matrix of input weight means
    input_w_sigma = params['inputs']['weights']['w_sigma']  # Matrix of input weight std devs
    input_w_mu_FF = params['inputs']['weights']['w_mu_FF']  # Scalar for feedforward weights
    input_w_sigma_FF = params['inputs']['weights']['w_sigma_FF']  # Scalar for feedforward weights
    
    # Extract scaling parameters (matrices)
    scaling_factors = params['optimisation']['scaling_factors']  # Matrix of scaling factors
    scaling_factors_FF = params['optimisation']['scaling_factors_FF']  # Matrix for feedforward

    # ================================================================================================
    # STEP 1: Generate Assembly-Based Topology and Visualization
    # ================================================================================================

    # Generate assembly-based connectivity graph and cell type assignments
    connectivity_graph, cell_type_indices = topology_generators.assembly_generator(
        num_neurons=num_neurons,
        num_assemblies=num_assemblies,
        conn_within=conn_within,
        conn_between=conn_between,
        cell_type_proportions=cell_type_proportions,
    )
    
    # Map cell type indices to +1/-1 based on cell_type_signs from config
    neuron_types = np.array([cell_type_signs[idx] for idx in cell_type_indices], dtype=np.int_)

    # ================================================================================================
    # STEP 2: Assign Synaptic Weights and Analyze Connectivity
    # ================================================================================================

    # Assign log-normal weights to connectivity graph
    weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=connectivity_graph,
        source_cell_indices=cell_type_indices,
        target_cell_indices=cell_type_indices,  # Same for recurrent connections
        cell_type_signs=cell_type_signs,
        w_mu_matrix=w_mu,
        w_sigma_matrix=w_sigma,
    )

    # ================================================================================================
    # STEP 3: Create Feedforward Inputs from Mitral Cells
    # ================================================================================================

    # Generate Poisson spike trains for mitral cells
    n_steps = int(duration / delta_t)
    shape = (1, n_steps, input_num_neurons)
    p_spike = input_firing_rates[0] * delta_t * 1e-3  # rate in Hz, delta_t in ms
    input_spikes = np.random.rand(*shape) < p_spike

    # Generate feedforward connectivity graph using cross-layer connectivity
    feedforward_connectivity_graph = topology_generators.generate_cross_layer_connectivity(
        n_source=input_num_neurons,
        n_target=num_neurons,
        source_cell_proportions=[1.0],  # Input layer: all mitral cells
        target_cell_indices=cell_type_indices,  # Connectome layer cell assignments
        conn_matrix=input_conn_inputs,  # 1x2 matrix: [mitral->excitatory, mitral->inhibitory]
    )

    # Assign log-normal weights to feedforward connectivity 
    # Source: input layer (all mitral cells = type 0), Target: connectome layer
    input_source_indices = np.zeros(input_num_neurons, dtype=int)  # All mitral cells are type 0
    feedforward_weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=feedforward_connectivity_graph,
        source_cell_indices=input_source_indices,
        target_cell_indices=cell_type_indices,
        cell_type_signs=[1],  # Input mitral cells are excitatory (sign +1)
        w_mu_matrix=input_w_mu,
        w_sigma_matrix=input_w_sigma,
    )

    # ================================================================================================
    # STEP 4: Initialize and Run LIF Network Simulation
    # ================================================================================================

    # Initialize LIF network model
    model = CurrentLIFNetwork(
        params_file=params_file,
        neuron_types=neuron_types,
        recurrent_weights=weights,
        feedforward_weights=feedforward_weights,
    )

    # Move model to device for GPU acceleration
    model.to(device)

    # Run inference
    with torch.inference_mode():
        output_spikes, output_voltages, output_I_exc, output_I_inh = model.forward(
            n_steps=n_steps,
            delta_t=delta_t,
            inputs=input_spikes,
        )
    
    # Move tensors to CPU for further processing and saving
    output_spikes = output_spikes.cpu()
    output_voltages = output_voltages.cpu()
    output_I_exc = output_I_exc.cpu()
    output_I_inh = output_I_inh.cpu()

    # ================================================================================================
    # STEP 5: Save Output Data for Further Analysis
    # ================================================================================================

    # Save output arrays
    np.save(output_dir / "output_spikes.npy", output_spikes.numpy())
    np.save(output_dir / "output_voltages.npy", output_voltages.numpy())
    np.save(output_dir / "output_I_exc.npy", output_I_exc.numpy())
    np.save(output_dir / "output_I_inh.npy", output_I_inh.numpy())
    np.save(output_dir / "input_spikes.npy", input_spikes)
    np.save(output_dir / "neuron_types.npy", neuron_types)
    np.save(output_dir / "cell_type_indices.npy", cell_type_indices)
    np.save(output_dir / "connectivity_graph.npy", connectivity_graph)
    np.save(output_dir / "weights.npy", weights)
    np.save(output_dir / "feedforward_weights.npy", feedforward_weights)

    # ================================================================================================
    # STEP 6: Generate All Plots and Visualizations
    # ================================================================================================
    
    # Call the plotting script to generate all visualizations
    simulate_Dp_plots.main(output_dir)

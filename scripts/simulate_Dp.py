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
import toml
from synthetic_connectome import topology_generators, weight_assigners
from synthetic_connectome.cell_types import assign_cell_types
from network_simulators.current_lif_network import CurrentLIFNetwork
import torch
import sys
from pathlib import Path

# Import local scripts
sys.path.append(str(Path(__file__).resolve().parent))
import simulate_Dp_plots


def main(output_dir, params_file):
    """Main execution function for Dp network simulation.

    Args:
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
    """
    # ============================================
    # SETUP: Device selection and parameter loading
    # ==========================================

    # Select device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all network parameters from TOML file
    with open(params_file, "r") as f:
        params = toml.load(f)

    # Extract and convert all parameters at the top
    dt = float(params["simulation"]["dt"])
    duration = float(params["simulation"]["duration"])
    seed = params["simulation"].get("seed", None)

    num_neurons = int(params["connectome"]["topology"]["num_neurons"])
    num_assemblies = int(params["connectome"]["topology"]["num_assemblies"])
    conn_within = np.array(params["connectome"]["topology"]["conn_within"])
    conn_between = np.array(params["connectome"]["topology"]["conn_between"])

    cell_type_names = params["connectome"]["cell_types"]["names"]
    cell_type_signs = np.array(params["connectome"]["cell_types"]["signs"], dtype=int)
    cell_type_proportions = np.array(
        params["connectome"]["cell_types"]["proportion"], dtype=float
    )

    w_mu = np.array(params["connectome"]["weights"]["w_mu"], dtype=float)
    w_sigma = np.array(params["connectome"]["weights"]["w_sigma"], dtype=float)

    input_num_neurons = int(params["inputs"]["topology"]["num_neurons"])
    input_cell_type_names = params["inputs"]["cell_types"]["names"]
    input_firing_rates = np.array(
        params["inputs"]["activity"]["firing_rates"], dtype=float
    )
    input_conn_inputs = np.array(
        params["inputs"]["topology"]["conn_inputs"], dtype=float
    )
    input_w_mu = np.array(params["inputs"]["weights"]["w_mu"], dtype=float)
    input_w_sigma = np.array(params["inputs"]["weights"]["w_sigma"], dtype=float)

    # Correctly structure physiology_params as a nested dictionary
    physiology_params = {
        ct: {
            "tau_mem": float(params["physiology"][ct]["tau_mem"]),
            "tau_syn": float(params["physiology"][ct]["tau_syn"]),
            "R": float(params["physiology"][ct]["R"]),
            "U_rest": float(params["physiology"][ct]["U_rest"]),
            "theta": float(params["physiology"][ct]["theta"]),
            "U_reset": float(params["physiology"][ct]["U_reset"]),
        }
        for ct in cell_type_names
    }

    scaling_factors = np.array(params["optimisation"]["scaling_factors"], dtype=float)
    scaling_factors_FF = np.array(
        params["optimisation"]["scaling_factors_FF"], dtype=float
    )

    surrgrad_scale = float(params["hyperparameters"]["surrgrad_scale"])

    # Set global random seed for reproducibility (only if specified in config)
    if seed is not None:
        np.random.seed(seed)
        print(f"Using seed: {seed}")
    else:
        print("No seed specified - using random initialization")

    # ==========================================================
    # STEP 1: Generate Assembly-Based Topology and Visualization
    # ==========================================================

    # First assign cell types to source and target neurons (same for recurrent)
    cell_type_indices = assign_cell_types(
        num_neurons=num_neurons,
        cell_type_proportions=cell_type_proportions,
    )

    # Generate assembly-based connectivity graph
    connectivity_graph = topology_generators.assembly_generator(
        source_cell_types=cell_type_indices,
        target_cell_types=cell_type_indices,  # Same for recurrent connections
        num_assemblies=num_assemblies,
        conn_within=conn_within,
        conn_between=conn_between,
    )

    # Map cell type indices to +1/-1 based on cell_type_signs from config
    neuron_types = np.array(
        [cell_type_signs[idx] for idx in cell_type_indices], dtype=np.int_
    )

    # ========================================================
    # STEP 2: Assign Synaptic Weights and Analyze Connectivity
    # ========================================================

    # Assign log-normal weights to connectivity graph
    weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=connectivity_graph,
        source_cell_indices=cell_type_indices,
        target_cell_indices=cell_type_indices,  # Same for recurrent connections
        cell_type_signs=cell_type_signs,
        w_mu_matrix=w_mu,
        w_sigma_matrix=w_sigma,
    )

    # ===================================================
    # STEP 3: Create Feedforward Inputs from Mitral Cells
    # ===================================================

    # Generate Poisson spike trains for mitral cells
    n_steps = int(duration / dt)
    shape = (1, n_steps, input_num_neurons)
    p_spike = input_firing_rates[0] * dt * 1e-3  # rate in Hz, dt in ms
    input_spikes = np.random.rand(*shape) < p_spike

    # Assign cell types to input layer (all mitral cells are type 0)
    input_source_indices = assign_cell_types(
        num_neurons=input_num_neurons,
        cell_type_proportions=[1.0],  # Input layer: all mitral cells
    )

    # Generate feedforward connectivity graph
    feedforward_connectivity_graph = topology_generators.sparse_graph_generator(
        source_cell_types=input_source_indices,
        target_cell_types=cell_type_indices,  # Connectome layer cell assignments
        conn_matrix=input_conn_inputs,  # 1x2 matrix: [mitral->excitatory, mitral->inhibitory]
        allow_self_loops=True,  # Allow for feedforward connections
    )

    # Assign log-normal weights to feedforward connectivity
    # Source: input layer (all mitral cells = type 0), Target: connectome layer
    feedforward_weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=feedforward_connectivity_graph,
        source_cell_indices=input_source_indices,
        target_cell_indices=cell_type_indices,
        cell_type_signs=[1],  # Input mitral cells are excitatory (sign +1)
        w_mu_matrix=input_w_mu,
        w_sigma_matrix=input_w_sigma,
    )

    # =================================================
    # STEP 4: Initialize and Run LIF Network Simulation
    # =================================================

    # Initialize LIF network model with corrected arguments
    model = CurrentLIFNetwork(
        weights=weights,
        cell_types=cell_type_names,
        cell_type_indices=cell_type_indices,
        physiology_params=physiology_params,
        scaling_factors=scaling_factors,
        surrgrad_scale=surrgrad_scale,
        weights_FF=feedforward_weights,
        cell_types_FF=input_cell_type_names,
        cell_type_indices_FF=input_source_indices,
        scaling_factors_FF=scaling_factors_FF,
    )

    # Move model to device for GPU acceleration
    model.to(device)

    # Run inference
    with torch.inference_mode():
        output_spikes, output_voltages, output_I_exc, output_I_inh = model.forward(
            n_steps=n_steps,
            dt=dt,
            inputs=input_spikes,
        )

    # Move tensors to CPU for further processing and saving
    output_spikes = output_spikes.cpu()
    output_voltages = output_voltages.cpu()
    output_I_exc = output_I_exc.cpu()
    output_I_inh = output_I_inh.cpu()

    # ============================================
    # STEP 5: Save Output Data for Further Analysis
    # ============================================

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

    # =============================================
    # STEP 6: Generate All Plots and Visualizations
    # =============================================

    # Call the plotting script to generate all visualizations
    simulate_Dp_plots.main(output_dir)
